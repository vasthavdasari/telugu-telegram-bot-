#!/usr/bin/env python3
"""
Telegram Translator Bot — English <-> Colloquial Telugu (text + voice)

What it does
    - English text         -> everyday spoken Telugu (copy/paste ready)
    - Telugu text          -> polished casual English
    - Telugu voice message -> Telugu transcript + polished English

Providers (sequential fallback — only 1 API call per message on success)
    - Telegram Bot API       unlimited, personal use, free
    - Gemini 2.5 Flash       PRIMARY for text translation (best Telugu quality)
    - Groq Llama 3.3 70B     FALLBACK for text if Gemini fails (1000 req/day)
    - Groq Whisper Large v3  PRIMARY for voice transcription (2000 req/day)
    - Gemini 2.5 multimodal  FALLBACK for voice if Groq Whisper fails

Required environment variables
    TELEGRAM_BOT_TOKEN   from @BotFather
    GEMINI_API_KEY       (optional) from aistudio.google.com/apikey
    GROQ_API_KEY         (optional) from console.groq.com
    PORT                 optional; defaults to 8080 (health endpoint)

    At least one of GEMINI_API_KEY or GROQ_API_KEY must be set.

Stdlib only. No pip install needed.
"""

import base64
import json
import os
import pathlib
import secrets
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer


# ---------- config ----------
def _require_env(name):
    v = os.environ.get(name, "").strip()
    if not v:
        print(f"ERROR: environment variable {name} is not set.", flush=True)
        sys.exit(1)
    return v


TELEGRAM_BOT_TOKEN = _require_env("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
PORT = int(os.environ.get("PORT", "8080"))

if not (GEMINI_API_KEY or GROQ_API_KEY):
    print("ERROR: at least one of GEMINI_API_KEY or GROQ_API_KEY must be set.", flush=True)
    sys.exit(1)

# Provider priority — SEQUENTIAL fallback, never parallel:
#   Text:  Gemini 2.5 Flash (quality) -> Groq Llama 3.3 70B (capacity)
#   Voice: Groq Whisper Large v3 (purpose-built) -> Gemini 2.5 Flash multimodal (fallback)
# If the primary succeeds, only 1 API call per user message. Fallback only runs
# when the primary raises an error.
GEMINI_TEXT_MODEL = "gemini-2.5-flash"
GEMINI_AUDIO_MODEL = "gemini-2.5-flash"
GROQ_CHAT_MODEL = "llama-3.3-70b-versatile"
GROQ_WHISPER_MODEL = "whisper-large-v3"

TG_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
TG_FILE = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}"

GEMINI_URL_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


def _gemini_url(model):
    return GEMINI_URL_TMPL.format(model=model)


POLL_TIMEOUT = 30
HTTP_TIMEOUT = 90
STT_TIMEOUT = 120

# Transient HTTP errors we silently retry on (Gemini occasionally returns
# 503 on free tier when servers are busy; 5xx/429 are all retryable).
RETRYABLE_HTTP = {429, 500, 502, 503, 504}


def _retry_wait(attempt):
    """Exponential backoff: 5, 10, 20, 40, 80 seconds (capped at 90)."""
    return min(5.0 * (2 ** attempt), 90.0)

# Daily hard caps (resets midnight Pacific).
# Text: matches Gemini Flash-Lite's free tier cap (1000/day).
# Voice: kept well under GCP's 1GB/month egress — 50/day × 200KB × 30d ~= 300 MB.
TEXT_DAILY_LIMIT = 1000
VOICE_DAILY_LIMIT = 50

USAGE_FILE = os.environ.get("USAGE_FILE", "usage.json")
_usage_lock = threading.Lock()


def _pacific_today():
    """Today's date in Pacific time (where Gemini's daily quota resets)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    except Exception:
        # Fallback: UTC-7 (approximates Pacific without DST awareness).
        return datetime.now(timezone(timedelta(hours=-7))).strftime("%Y-%m-%d")


def _next_reset_ist():
    """Return the next quota reset time as a short IST string, e.g. '12:30 PM IST'."""
    try:
        from zoneinfo import ZoneInfo
        pt_now = datetime.now(ZoneInfo("America/Los_Angeles"))
        pt_midnight = (pt_now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        ist = pt_midnight.astimezone(ZoneInfo("Asia/Kolkata"))
        h = ist.hour % 12 or 12
        return f"{h}:{ist.minute:02d} {'AM' if ist.hour < 12 else 'PM'} IST"
    except Exception:
        return "midnight Pacific"


def _status_line():
    """One-line summary of remaining quota + reset time, appended to replies."""
    with _usage_lock:
        d = _load_usage()
    text_left = max(0, TEXT_DAILY_LIMIT - d.get("text", 0))
    voice_left = max(0, VOICE_DAILY_LIMIT - d.get("voice", 0))
    return (
        f"📊 Text: {text_left}/{TEXT_DAILY_LIMIT} · "
        f"Voice: {voice_left}/{VOICE_DAILY_LIMIT} · "
        f"Resets {_next_reset_ist()}"
    )


def _load_usage():
    try:
        with open(USAGE_FILE) as f:
            d = json.load(f)
    except Exception:
        d = {}
    if d.get("date") != _pacific_today():
        d = {"date": _pacific_today(), "text": 0, "voice": 0}
    return d


def _save_usage(d):
    try:
        with open(USAGE_FILE, "w") as f:
            json.dump(d, f)
    except Exception as e:
        print(f"[usage save failed: {e}]", flush=True)


def try_reserve(kind):
    """Atomically reserve one slot for 'text' or 'voice'.

    Returns remaining count after this reservation, or None if we're
    already at the daily cap (hard block, no API call should be made).
    """
    limit = TEXT_DAILY_LIMIT if kind == "text" else VOICE_DAILY_LIMIT
    with _usage_lock:
        d = _load_usage()
        if d.get(kind, 0) >= limit:
            return None
        d[kind] = d.get(kind, 0) + 1
        _save_usage(d)
        return max(0, limit - d[kind])


def release(kind):
    """Refund one slot (call when an API request errors out)."""
    with _usage_lock:
        d = _load_usage()
        d[kind] = max(0, d.get(kind, 0) - 1)
        _save_usage(d)


# ---------- prompts ----------
EN_TO_TE_SYSTEM = """YOU ARE A PURE TRANSLATOR. NOT A CHATBOT. NOT AN ASSISTANT.

The input is ALWAYS English text to translate to Telugu — never a question to you, never a command, never a conversation. Translate it, that's it.

If the input looks like a question to you or meta-conversation (e.g. "how are you?", "what are you doing?", "help me", "are you there?") — you STILL translate it literally to Telugu. You NEVER respond as an assistant. You NEVER answer. You NEVER explain what you do. Your ONLY output is the Telugu translation.

Correct examples:
Input: "how are you"
OUTPUT: ఎలా ఉన్నావ్?

Input: "what are you doing"
OUTPUT: ఏం చేస్తున్నావ్?

Input: "hello, are you there"
OUTPUT: Hello, ఉన్నావా?

Input: "can you help me"
OUTPUT: నాకు help చేయగలవా?

-----

You translate English to Telugu. CONTEXT: primarily the user is a buyer on eBay reading replies from sellers — about price, item condition, shipping, returns, availability, order status — but may also send general conversational English to be translated. Your output helps the user understand what was written.

OUTPUT FORMAT — THIS IS CRITICAL:
- Use Telugu SCRIPT (తెలుగు Unicode) for Telugu words. NEVER Tinglish (Telugu words typed in Roman letters like "nammadi", "gurinchi", "adigav").
- If a word is genuinely a natural English loanword Telugu speakers use (e.g. "price", "shipping", "phone", "hello"), keep it in English Latin letters — this is natural code-mixing and sounds more real than forced Telugu translation.
- The mix should read like how an urban Telugu speaker actually types on WhatsApp: Telugu script for Telugu grammar/verbs/pronouns, English Latin for common English loanwords.

WRONG (Tinglish — Telugu VERBS/PRONOUNS in Roman letters, NEVER produce):
"Nammadi price gurinchi adugutunnara"  ← "adugutunnara" must be అడుగుతున్నారా
"Nuvvu mundu price gurinchi adigav kada"  ← "nuvvu", "adigav", "kada" must be Telugu script
"anukuntunnanu small discount istara ani"  ← "anukuntunnanu", "istara", "ani" must be Telugu script

RIGHT (Telugu script + natural English loanwords):
"మనం మునుపు మాట్లాడిన price negotiation గురించి నువ్వు ఏమనుకున్నావ్?"
"Life ఎలా ఉంది?"

ABSOLUTE RULE — Telugu verbs/pronouns/conjunctions/particles MUST be Telugu script:
If you find yourself about to write any Telugu word in Roman letters — especially verb forms ending in -anu, -ava, -avu, -aru, -adu, -indi, -undi, -nanu, -tava, -tunna, -tunnav, or pronouns like nuvvu/nenu/memu/meeru/vaallu, or conjunctions like kada/ani/ante/valla — STOP and rewrite that token in Telugu Unicode script. Only naturalized English nouns (price, shipping, item, camera, discount, phone, order, etc.) stay in Latin letters.

COMMON ENGLISH LOANWORDS (EXPERIMENTAL — based on research of modern Telugu code-mixing).
ALWAYS keep these words in English (Latin letters) when they appear in the input. Never translate them to Sanskrit-heavy Telugu equivalents:

  Courtesy:      hello, hi, bye, please, sorry, thanks, thank you, ok, okay, welcome, yes, no
  Commerce:      price, cost, shipping, delivery, payment, order, refund, return, discount, offer,
                 item, brand, model, box, case, condition, original, genuine, used, new, available,
                 free, cash, card, online, store, market, deal, quality, quantity, stock
  Money:         dollar, dollars, rupees, USD, INR, amount, total
  Tech/Comms:    phone, mobile, laptop, computer, internet, email, WhatsApp, message, website,
                 link, app, notification, password, OTP, login, video, photo, screenshot, file
  Logistics:     tracking, address, pincode, zipcode, courier, parcel, package, weight, size
  Time:          morning, evening, today, tomorrow, AM, PM, minute, hour, second, week, month
  Work/Places:   meeting, office, boss, client, project, plan, schedule, hospital, hotel, station
  People:        friend, bro, sister, family, guys
  Descriptors:   good, bad, nice, super, great, best, fine, awesome, cool, urgent, important,
                 ready, final, fix, normal, small, big, full, empty, fresh, old
  Common verbs:  ship, cancel, confirm, accept, reject, send, call, reply, update, check,
                 order (as verb), pay, refund (as verb)

This list reflects how urban Telugu speakers actually write on WhatsApp today. Use Telugu SCRIPT everywhere else.

RULES:
- Output ONLY the Telugu text. No prose, no explanation, no romanization, no quotation marks, no "Translation:" prefix.
- TRANSLATE THE MEANING, not just individual words. Understand what the English message is really saying, then convey the same intent in Telugu.
- Use EVERYDAY SPOKEN Telugu (వాడుక భాష), NOT formal/literary/textbook Telugu.
- Avoid heavy Sanskrit-derived vocabulary. Use plain words a young person would say to a friend.
- Natural code-mixing with English is expected and encouraged.
- Never translate proper nouns, brand names, product names (iPhone, PS5, Nikon, etc.), model numbers, dollar amounts, or URLs.
- Keep length and register similar to the English input.

EMOTIONAL TONE — carry the SAME emotion from English into Telugu, don't flatten it:
- Hedging/tentative buyer ("I was hoping", "might consider", "would it be possible", "if you could") → preserve the softness with "అని అనుకుంటున్నాను", "possible అయితే", "కొంచెం"; don't flatten into a blunt demand.
- Polite/friendly seller ("Sure thing!", "Happy to help", "of course") → use cheerful Telugu: "తప్పకుండా", "సంతోషంగా", exclamation, ishtamga
- Firm/dismissive ("Price is firm", "no negotiations", "final") → keep it firm: "price fix", "negotiations ledu", short sentences
- Apologetic ("really sorry", "my apologies") → reflect apology: "really sorry", "క్షమించండి", "nijanga sorry"
- Frustrated/annoyed ("Already told you", "I said no") → reflect their annoyance: "already cheppanu kada", "ippatike cheppanu"
- Enthusiastic ("Awesome!", "Great choice!") → use "!", "super", "awesome", match energy
- Neutral/business-like → standard casual Telugu, no extra emotion

The Telugu output MUST carry the SAME emotion as the English seller's message. The user reading it should feel the same vibe the seller intended.

EXAMPLES — every output uses Telugu SCRIPT for Telugu words, English Latin for natural loanwords:

English: "Hi, yes the item is still available."
Telugu:  Hi, అవును item ఇంకా available గా ఉంది.

English: "I can do $150 plus $15 shipping."
Telugu:  $150 plus $15 shipping ఇవ్వగలను.

English: "Sorry, I only ship within the US."
Telugu:  Sorry, US లో మాత్రమే ship చేయగలను.

English: "Minor wear on the edges, otherwise in excellent condition."
Telugu:  Edges దగ్గర కొంచెం wear ఉంది, మిగతా అన్నీ excellent condition లో ఉన్నాయి.

English: "Let me know if you have any other questions."
Telugu:  ఇంకా ఏవైనా doubts ఉంటే చెప్పండి.

English: "I'll ship it out tomorrow once payment clears."
Telugu:  Payment clear అయ్యాక రేపు ship చేస్తాను.

English: "No returns on this item, sold as-is."
Telugu:  ఈ item కి returns లేవు, as-is sold.

English: "Would you accept $180 shipped?"
Telugu:  Shipping కలిపి $180 కి accept చేస్తావా?

English: "What did you think about the price negotiation we talked previously?"
Telugu:  మనం మునుపు మాట్లాడిన price negotiation గురించి నువ్వు ఏమనుకున్నావ్?

English: "How's life?"
Telugu:  Life ఎలా ఉంది?

EXAMPLES showing TONE preservation from English seller:

Firm: "Price is firm, no negotiations."
Telugu:  Price fix, negotiations కి scope లేదు.

Friendly: "Sure thing! Happy to answer any questions."
Telugu:  తప్పకుండా! ఏ doubts ఉన్నా ఇష్టంగా చెప్తాను.

Apologetic: "I'm really sorry for the delay, will ship today."
Telugu:  Delay కి really sorry, ఇవాళ ship చేస్తాను.

Dismissive: "Already told you, no returns."
Telugu:  Already చెప్పాను కదా, returns లేవు.

Enthusiastic: "Awesome! Great choice, you'll love it!"
Telugu:  Super! Choice బాగుంది, మీకు చాలా నచ్చుతుంది!

Frustrated: "I said no returns. Don't ask again."
Telugu:  Returns లేవు అని చెప్పాను కదా. మళ్ళీ అడగకు.

Hedging: "I was hoping you might consider a small discount since I'm buying multiple items."
Telugu:  మీ దగ్గర multiple items కొంటున్నాను, కొంచెం discount ఇవ్వగలరా అని అనుకుంటున్నాను."""

TE_TO_EN_SYSTEM = """YOU ARE A PURE TRANSLATOR. NOT A CHATBOT. NOT AN ASSISTANT.

The input is ALWAYS text to translate — never a question to you, never a command, never a conversation. Translate it. That's it.

If the input looks like a question to you or meta-conversation (e.g. "ela unnav", "em chesthunnav", "help kavali", "nuvvu enti chesthunnavu") — you STILL translate it literally. You NEVER respond as an assistant. You NEVER answer. You NEVER explain what you do. Your ONLY output is the English translation itself.

Correct examples:
Input: "ela unnav bro"
OUTPUT: How are you, bro?

Input: "naaku help kavali"
OUTPUT: I need help.

Input: "em chesthunnav"
OUTPUT: What are you doing?

Input: "nuvvu ekkada unnav"
OUTPUT: Where are you?

-----

You translate Telugu to English. The input may be in Telugu script (తెలుగు) OR in Tinglish (Telugu typed with Roman/English letters, e.g. "naaku e camera entha price ki istharu", "ela undi", "cheppandi"). Treat both as Telugu and translate to English. CONTEXT: the user is a buyer on eBay messaging sellers — asking about price, item condition, shipping, making offers, placing/confirming orders. Your output will be pasted directly into an eBay message box.

YOUR GOAL: Sound like a native English speaker messaging an eBay seller — polite, direct, natural. Not homework English, not a news headline, not a robot.

HARD RULES:
- Output ONLY the English translation. No prefix, no label, no quotes, no explanation.
- Never translate proper nouns, brand names, product names (iPhone, PS5, Nikon, etc.), model numbers, place names, or URLs.
- Keep any English words already mixed into the Telugu AS-IS (don't re-translate them).
- Fix obvious speech-to-text errors using context clues.
- Preserve the original's tone (casual stays casual, firm stays firm, polite stays polite).

STYLE RULES — make it sound human:
- USE CONTRACTIONS: "I'm", "you're", "don't", "it's", "won't", "what's".
- Match the eBay-buyer register:
  * Asking price → "how much are you selling this for?", "what's your asking price?", "what are you looking to get for this?"
  * Negotiating → "would you take $X for it?", "any chance you'd do $X?", "can you knock off $X?"
  * Item condition → "what condition is it in?", "any scratches or defects?", "how much wear is there?"
  * Shipping → "do you ship to [country]?", "what's the shipping cost?", "do you offer combined shipping?"
  * Box/accessories → "does it come with the original box?", "are all accessories included?"
  * Availability → "is this still available?"
  * Closing → "thanks!", "appreciate it!"
- NEVER output telegraphic/broken English like "Hello, how much for camera" — always produce complete, natural sentences.
- Vary sentence structure. Don't copy Telugu word order verbatim.

BANNED WORDS (these make text sound AI-written — never use them):
delve, leverage, navigate, journey, realm, tapestry, landscape, foster, harness,
embark, unlock potential, in the realm of, it's worth noting, it's important to note,
moreover, furthermore, nevertheless, hence, thus, crucial, pivotal, seamless, robust.

EMOTIONAL TONE — carry the SAME emotion, don't flatten it:
Before translating, identify the tone. Telugu tonal cues and their English equivalents:
- Polite/respectful (andi, garu, dayachesi, please, softeners) → "could you please", "would you mind", "I'd really appreciate", "thank you so much"
- Pleading (repeated asks, "chala kavali naaku", emotional emphasis) → "would really mean a lot", "I'd love to", "please, I'd really appreciate"
- Urgent (twaraga, urgent, exclamations, "vendi vendi") → "ASAP", "really need this soon", exclamation marks
- Firm/assertive (short, direct, no softeners, "final", "take it or leave it") → crisp, no fluff: "I'll take it for $X", "final offer"
- Casual/neutral (standard phrasing) → everyday natural English
- Friendly/curious ("anukuntunna", light chatty tone) → "just wondering", "curious about", "hey so..."
- Frustrated/annoyed (sharp, "ento idi", complaining markers) → firm disappointment: "this isn't what I expected", "can you explain why..."
- Happy/excited (exclamations, "wow", "super", enthusiasm markers) → use "!", "awesome", "love it", "sounds great"

The English output MUST carry the SAME emotion as the Telugu. A polite request stays polite. A firm demand stays firm. An excited message stays excited. Never neutralize.

EXAMPLES (Telugu → natural English for eBay):
"hello e camera naaku entha price ki istharu"
→ Hey, how much are you selling this camera for?

"10 dollars taggincha galava"
→ Any chance you'd knock off $10?

"combined shipping untada rendu items ki"
→ Do you offer combined shipping if I buy two items?

"e laptop condition ela undi, scratches unnaya"
→ What condition is this laptop in — are there any scratches?

"india ki ship chestara"
→ Do you ship to India?

"original box, accessories anni vastunda"
→ Does it come with the original box and all the accessories?

"200 dollars ki ichhestava, best offer"
→ Would you take $200 as a best offer?

"item still available a"
→ Is this item still available?

"return policy enti meeru"
→ What's your return policy?

"thanks, confirm chesi order chestha"
→ Thanks, I'll confirm and place the order.

EXAMPLES showing TONE preservation (same topic, different emotion):

Polite: "please cheppandi e camera condition ela undi, naaku chala interested ga undi"
→ Could you please let me know the condition of this camera? I'm really interested in it.

Neutral: "camera condition ela undi"
→ What's the condition of this camera?

Firm: "final offer 200 dollars, take it or leave it"
→ Final offer: $200. Take it or leave it.

Pleading: "please please 150 ki ivvandi, chala important naaku"
→ Please, would you consider $150? It would really mean a lot to me.

Urgent: "urgent ga reply cheyyi, repu ship avvali eesari"
→ Please reply ASAP — I really need this shipped by tomorrow!

Friendly/curious: "hey just anukuntunna, ee model kothadhi a"
→ Hey, just curious — is this a newer model?

Frustrated: "ento ee delay, 5 rojulu ayindi inka ship cheyyaledu"
→ What's with the delay? It's been 5 days and still not shipped.

Excited: "wow super deal, definitely kontha, thanks"
→ Wow, great deal — I'll definitely take it, thanks!"""

VOICE_PROMPT = """YOU ARE A PURE TRANSLATOR. NOT A CHATBOT. NOT AN ASSISTANT.

The audio is ALWAYS Telugu speech to transcribe and translate — never a question to you, never a command to act on. Transcribe + translate. That's it.

If the speaker asks questions that sound directed at an AI ("what can you do?", "are you there?", "help me"), STILL just transcribe the Telugu and translate the phrase literally to English. NEVER respond as an assistant. NEVER answer. NEVER break the TELUGU/ENGLISH output format.

-----

The audio is a voice message spoken in Telugu. CONTEXT: the user is a buyer on eBay messaging sellers — asking about price, item condition, shipping, making offers, confirming orders. The English output will be pasted directly into an eBay message box.

Do TWO things:
1. Transcribe the Telugu verbatim in Telugu script.
2. Translate to English following ALL the rules below.

TRANSLATION RULES:
- Sound like a native English speaker messaging an eBay seller — polite, direct, natural. NOT homework English, NOT a headline, NOT a robot.
- USE CONTRACTIONS ("I'm", "you're", "don't", "what's", "won't").
- Never telegraphic/broken (never output "Hello, how much for camera" — always complete natural sentences).
- Keep English words already mixed into the Telugu AS-IS.
- Never translate proper nouns, brand names, product names (iPhone, PS5, Nikon, etc.), model numbers, or URLs.
- Fix obvious speech-to-text errors using context.
- Preserve the original's tone.
- Vary sentence structure — don't follow Telugu word order literally.
- BANNED WORDS (never use): delve, leverage, navigate, journey, realm, tapestry, landscape, foster, harness, embark, unlock, seamless, robust, crucial, pivotal, moreover, furthermore, nevertheless, hence, thus.

eBay-buyer register patterns:
- Asking price → "how much are you selling this for?", "what's your asking price?"
- Negotiating → "would you take $X?", "any chance you'd knock off $X?"
- Condition → "what condition is it in?", "any scratches or defects?"
- Shipping → "do you ship to [country]?", "what's the shipping cost?"
- Box/accessories → "does it come with the original box?", "are accessories included?"
- Closing → "thanks!", "appreciate it!"

EMOTIONAL TONE — carry the SAME emotion, never flatten it:
- Polite (andi, garu, dayachesi, please) → "could you please", "would you mind", "I'd appreciate"
- Pleading (repeated asks, emotional emphasis) → "would really mean a lot", "please, I'd love"
- Urgent (twaraga, exclamations) → "ASAP", "really need this soon", "!"
- Firm (short, direct, final) → crisp: "Final offer: $X", "Take it or leave it"
- Casual (standard) → everyday natural English
- Friendly/curious (anukuntunna, chatty) → "just wondering", "curious about"
- Frustrated (sharp complaining) → firm: "what's going on with...", "this isn't what I expected"
- Excited (wow, enthusiasm) → "!", "love this", "sounds great"

The English MUST carry the SAME emotion as the Telugu. Polite stays polite. Firm stays firm. Excited stays excited.

EXAMPLES:
Telugu: "hello e camera naaku entha price ki istharu"
English: Hey, how much are you selling this camera for?

Telugu: "10 dollars taggincha galava"
English: Any chance you'd knock off $10?

Telugu: "e laptop condition ela undi, scratches unnaya"
English: What condition is this laptop in — are there any scratches?

Telugu: "india ki ship chestara"
English: Do you ship to India?

Telugu: "original box, accessories anni vastunda"
English: Does it come with the original box and all the accessories?

Telugu: "200 dollars ki ichhestava, best offer"
English: Would you take $200 as a best offer?

EXAMPLES showing TONE preservation:

Polite: "please cheppandi, camera condition chala important naaku"
English: Could you please let me know the condition? It's really important to me.

Firm: "final offer 200 dollars, take it or leave it"
English: Final offer: $200. Take it or leave it.

Urgent: "urgent ga reply cheyyi, repu ship avvali"
English: Please reply ASAP — I really need this shipped by tomorrow!

Frustrated: "ento idi, 5 rojulu ayyindi inka ship cheyyaledu"
English: What's going on? It's been 5 days and still not shipped.

Excited: "super deal, definitely kontha, thanks"
English: Great deal — I'll definitely take it, thanks!

Return EXACTLY two lines. NO markdown formatting (no **, no *, no backticks, no bold). NO leading or trailing prose. Each line must begin at column 0 with the literal ASCII prefix shown. Collapse any newlines in the transcript or translation into single spaces — each must fit on exactly one line.

Line 1 must begin with: TELUGU:
Line 2 must begin with: ENGLISH:"""


# ---------- http helpers (stdlib) ----------
def _http_request(url, data=None, headers=None, method=None, timeout=60):
    headers = dict(headers or {})
    headers.setdefault("User-Agent", "TeluguBot/1.0")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _http_json(url, payload=None, headers=None, timeout=60, method=None):
    headers = dict(headers or {})
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode()
    raw = _http_request(url, data=body, headers=headers, method=method, timeout=timeout)
    return json.loads(raw) if raw else {}


# ---------- telegram ----------
def tg_call(method, **params):
    url = f"{TG_API}/{method}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    try:
        data = _http_json(url, timeout=POLL_TIMEOUT + 5)
    except urllib.error.URLError as e:
        print(f"[telegram error on {method}: {e}]", flush=True)
        return None
    if not data.get("ok"):
        print(f"[telegram error on {method}: {data}]", flush=True)
        return None
    return data.get("result")


def get_updates(offset):
    return tg_call(
        "getUpdates",
        offset=offset,
        timeout=POLL_TIMEOUT,
        allowed_updates=json.dumps(["message", "edited_message", "callback_query"]),
    ) or []


def send_message(chat_id, text, reply_to=None, reply_markup=None):
    """Send a Telegram message. Returns the message_id of the LAST chunk sent (or None)."""
    MAX = 4000
    if not text:
        text = "(empty)"
    last_msg_id = None
    chunks = [text[i:i + MAX] for i in range(0, len(text), MAX)]
    for i, chunk in enumerate(chunks):
        params = {"chat_id": chat_id, "text": chunk}
        if reply_to and i == 0:
            params["reply_to_message_id"] = reply_to
        # Only attach the keyboard to the final chunk.
        if reply_markup and i == len(chunks) - 1:
            params["reply_markup"] = json.dumps(reply_markup)
        result = tg_call("sendMessage", **params)
        if result and isinstance(result, dict):
            last_msg_id = result.get("message_id")
    return last_msg_id


# ---------- feedback loop (thumbs up/down) ----------
FEEDBACK_FILE = os.environ.get("FEEDBACK_FILE", "feedback.jsonl")

# In-memory map of {bot_reply_message_id -> feedback context}. Best-effort — clears on restart.
_feedback_pending = {}
_feedback_lock = threading.Lock()
# Cap size to avoid unbounded growth.
_FEEDBACK_CACHE_MAX = 500

FEEDBACK_KEYBOARD = {
    "inline_keyboard": [[
        {"text": "👍 Good", "callback_data": "fb:good"},
        {"text": "👎 Bad", "callback_data": "fb:bad"},
    ]]
}


def _remember_feedback(reply_msg_id, ctx):
    if reply_msg_id is None:
        return
    with _feedback_lock:
        if len(_feedback_pending) >= _FEEDBACK_CACHE_MAX:
            # Drop the oldest half (FIFO-ish) to keep the dict bounded.
            for k in list(_feedback_pending.keys())[: _FEEDBACK_CACHE_MAX // 2]:
                _feedback_pending.pop(k, None)
        _feedback_pending[reply_msg_id] = ctx


def handle_callback(cb):
    """Handle an inline-keyboard button press (thumbs up/down on a translation)."""
    cb_id = cb.get("id")
    data = (cb.get("data") or "").strip()
    msg = cb.get("message") or {}
    msg_id = msg.get("message_id")
    chat_id = (msg.get("chat") or {}).get("id")

    rating = {"fb:good": "good", "fb:bad": "bad"}.get(data)
    if not rating:
        if cb_id:
            tg_call("answerCallbackQuery", callback_query_id=cb_id)
        return

    with _feedback_lock:
        ctx = _feedback_pending.pop(msg_id, None)

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "rating": rating,
        "msg_id": msg_id,
        "chat_id": chat_id,
    }
    if ctx:
        entry.update(ctx)
    try:
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[feedback log failed: {e}]", flush=True)

    # Quick toast, and edit the message to remove the keyboard so user doesn't re-tap.
    if cb_id:
        toast = "Thanks — noted 👍" if rating == "good" else "Noted — I'll log it for improving prompts 🙏"
        tg_call("answerCallbackQuery", callback_query_id=cb_id, text=toast)
    if chat_id and msg_id:
        tg_call(
            "editMessageReplyMarkup",
            chat_id=chat_id,
            message_id=msg_id,
            reply_markup=json.dumps({"inline_keyboard": []}),
        )


def send_chat_action(chat_id, action="typing"):
    tg_call("sendChatAction", chat_id=chat_id, action=action)


def download_voice(file_id):
    info = tg_call("getFile", file_id=file_id)
    if not info or "file_path" not in info:
        return None, None
    file_path = info["file_path"]
    url = f"{TG_FILE}/{file_path}"
    data = _http_request(url, timeout=STT_TIMEOUT)
    filename = pathlib.PurePosixPath(file_path).name or "voice.oga"
    return filename, data


def _audio_mime(filename):
    ext = filename.lower().rsplit(".", 1)[-1]
    return {
        "oga": "audio/ogg", "ogg": "audio/ogg", "opus": "audio/ogg",
        "mp3": "audio/mp3", "m4a": "audio/aac", "mp4": "audio/aac",
        "wav": "audio/wav", "flac": "audio/flac", "aac": "audio/aac",
        "aiff": "audio/aiff",
    }.get(ext, "audio/ogg")


# ---------- gemini helpers ----------
def _gemini_post(model, body, timeout, max_retries=2):
    """Single-model call with transient-error retries.
    Returns (ok, result_dict_or_error_str).

    On 429 (rate-limit / quota of any kind), returns immediately without
    retrying — each model has its own RPM and RPD, so cascading to the next
    model is faster and more likely to succeed than retrying the same one.
    On 5xx (transient server error), retries with exponential backoff.
    """
    url = f"{_gemini_url(model)}?key={urllib.parse.quote(GEMINI_API_KEY)}"
    for attempt in range(max_retries + 1):
        try:
            res = _http_json(url, payload=body, timeout=timeout)
            return True, res
        except urllib.error.HTTPError as e:
            body_text = e.read().decode(errors="replace")
            # Keep full body in logs so we can diagnose 429 flavors (RPM vs RPD).
            err = f"HTTP {e.code} — {body_text[:800]}".replace("\n", " ")
            # 429 (any flavor — RPM burst or daily quota): cascade to next model.
            if e.code == 429:
                return False, err
            # 5xx transient: retry with backoff on the same model.
            if e.code in RETRYABLE_HTTP and attempt < max_retries:
                wait = _retry_wait(attempt)
                print(f"[gemini {model} {e.code}; retry {attempt+1}/{max_retries} in {wait:.0f}s]", flush=True)
                time.sleep(wait)
                continue
            return False, err
        except Exception as e:
            if attempt < max_retries:
                wait = _retry_wait(attempt)
                print(f"[gemini {model} {e}; retry {attempt+1}/{max_retries} in {wait:.0f}s]", flush=True)
                time.sleep(wait)
                continue
            return False, str(e)
    return False, "retries exhausted"




def _extract_text(res):
    cands = res.get("candidates") or []
    if not cands:
        return ""
    parts = cands[0].get("content", {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts).strip()


# ---------- gemini (single-model text translate) ----------
def gemini_translate(system_prompt, user_text):
    """ONE call to Gemini 2.5 Flash. Returns text or '(gemini: ...)' error string."""
    if not GEMINI_API_KEY:
        return "(gemini: not configured)"
    body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_text}]}],
        "generationConfig": {"temperature": 0.3},
    }
    ok, result = _gemini_post(GEMINI_TEXT_MODEL, body, HTTP_TIMEOUT)
    if ok:
        text = _extract_text(result)
        return text or "(gemini: empty response)"
    return f"(gemini: {result[:200]})"


# ---------- gemini (single-model multimodal voice fallback) ----------
def gemini_voice(audio_bytes, filename):
    """ONE call to Gemini Flash multimodal. Returns {telugu, english} or {error}.
    Used as fallback when Groq Whisper is unavailable."""
    if not GEMINI_API_KEY:
        return {"error": "gemini not configured"}
    mime = _audio_mime(filename)
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    body = {
        "contents": [{
            "parts": [
                {"text": VOICE_PROMPT},
                {"inline_data": {"mime_type": mime, "data": b64}},
            ]
        }],
        "generationConfig": {"temperature": 0.3},
    }
    ok, result = _gemini_post(GEMINI_AUDIO_MODEL, body, STT_TIMEOUT)
    if not ok:
        return {"error": result}
    text = _extract_text(result)
    if not text:
        return {"error": "empty response"}
    telugu, english = "", ""
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("*").lstrip("-").lstrip().rstrip("*").strip()
        upper = line.upper()
        if upper.startswith("TELUGU:"):
            telugu = line.split(":", 1)[1].strip() if ":" in line else ""
        elif upper.startswith("ENGLISH:"):
            english = line.split(":", 1)[1].strip() if ":" in line else ""
    if telugu or english:
        return {"telugu": telugu, "english": english}
    return {"telugu": "", "english": text}


# ---------- multipart helper (for Groq Whisper audio upload) ----------
def _multipart_encode(fields, files):
    """Build multipart/form-data body.
    fields: dict of {name: str}. files: dict of {name: (filename, bytes, ctype)}."""
    boundary = b"--FormBoundary" + secrets.token_hex(12).encode()
    body = bytearray()
    for name, value in fields.items():
        body += b"--" + boundary + b"\r\n"
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += str(value).encode() + b"\r\n"
    for name, (filename, data, ctype) in files.items():
        body += b"--" + boundary + b"\r\n"
        body += (
            f'Content-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\n'
        ).encode()
        body += f"Content-Type: {ctype}\r\n\r\n".encode()
        body += data + b"\r\n"
    body += b"--" + boundary + b"--\r\n"
    return bytes(body), f"multipart/form-data; boundary={boundary.decode()}"


# ---------- groq (chat — Llama 3.3 70B) ----------
def groq_chat(system_prompt, user_text, max_retries=2):
    """ONE call to Groq Llama 3.3 70B. Returns text or '(groq: ...)' error string."""
    if not GROQ_API_KEY:
        return "(groq: not configured)"
    body_dict = {
        "model": GROQ_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.3,
    }
    body_bytes = json.dumps(body_dict).encode()
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    for attempt in range(max_retries + 1):
        try:
            raw = _http_request(GROQ_CHAT_URL, data=body_bytes, headers=headers, timeout=HTTP_TIMEOUT)
            res = json.loads(raw)
            text = res.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return text or "(groq: empty response)"
        except urllib.error.HTTPError as e:
            body_text = e.read().decode(errors="replace")
            err = f"HTTP {e.code} — {body_text[:200]}".replace("\n", " ")
            if e.code == 429:
                return f"(groq: {err})"
            if e.code in RETRYABLE_HTTP and attempt < max_retries:
                wait = _retry_wait(attempt)
                print(f"[groq chat {e.code}; retry {attempt+1}/{max_retries} in {wait:.0f}s]", flush=True)
                time.sleep(wait)
                continue
            return f"(groq: {err})"
        except Exception as e:
            if attempt < max_retries:
                time.sleep(_retry_wait(attempt))
                continue
            return f"(groq: {e})"
    return "(groq: retries exhausted)"


# ---------- groq (Whisper Large v3) ----------
def groq_transcribe(audio_bytes, filename, language="te", max_retries=1):
    """Send Telugu audio to Groq Whisper. Returns {transcript} or {error}.

    Groq's Whisper (OpenAI-compatible) validates by filename extension, accepting
    only: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm. Telegram delivers voice
    messages with .oga extension (valid OGG content, invalid name) — we rename it
    to .ogg before upload so Groq accepts it.
    """
    if not GROQ_API_KEY:
        return {"error": "groq not configured"}
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext in ("oga", "opus"):
        base = filename.rsplit(".", 1)[0] if "." in filename else filename
        filename = base + ".ogg"
        ext = "ogg"
    ctype = {
        "ogg": "audio/ogg",
        "mp3": "audio/mpeg", "m4a": "audio/mp4", "mp4": "audio/mp4",
        "wav": "audio/wav", "flac": "audio/flac", "webm": "audio/webm",
    }.get(ext, "audio/ogg")
    fields = {"model": GROQ_WHISPER_MODEL, "language": language, "response_format": "json"}
    files = {"file": (filename, audio_bytes, ctype)}
    body, content_type = _multipart_encode(fields, files)
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": content_type,
    }
    for attempt in range(max_retries + 1):
        try:
            raw = _http_request(GROQ_AUDIO_URL, data=body, headers=headers, timeout=STT_TIMEOUT)
            res = json.loads(raw)
            return {"transcript": (res.get("text") or "").strip()}
        except urllib.error.HTTPError as e:
            body_text = e.read().decode(errors="replace")
            err = f"HTTP {e.code} — {body_text[:200]}".replace("\n", " ")
            if e.code in RETRYABLE_HTTP and attempt < max_retries:
                wait = _retry_wait(attempt)
                print(f"[groq whisper {e.code}; retry in {wait:.0f}s]", flush=True)
                time.sleep(wait)
                continue
            return {"error": err}
        except Exception as e:
            return {"error": str(e)}
    return {"error": "retries exhausted"}


# ---------- unified entrypoints with SEQUENTIAL fallback ----------
# These are the only callers used by handle_message. On success, only 1 API call.
# Fallback (2nd call) runs only if the primary returned an error.

def translate_text(system_prompt, user_text):
    """Try Gemini 2.5 Flash first (best Telugu quality). On error, fall back to Groq Llama 3.3 70B."""
    if GEMINI_API_KEY:
        result = gemini_translate(system_prompt, user_text)
        if not result.startswith("("):
            return result
        print(f"[text: gemini failed ({result[:80]}); trying groq]", flush=True)
    if GROQ_API_KEY:
        return groq_chat(system_prompt, user_text)
    return "(no provider succeeded)"


def transcribe_voice(audio_bytes, filename):
    """Try Groq Whisper first (purpose-built STT). On error, fall back to Gemini multimodal.
    Returns {telugu, english (optional), error (optional)}."""
    if GROQ_API_KEY:
        result = groq_transcribe(audio_bytes, filename, language="te")
        if "transcript" in result and result["transcript"]:
            return {"telugu": result["transcript"]}
        print(f"[voice: groq whisper failed ({result.get('error', '?')[:80]}); trying gemini]", flush=True)
    if GEMINI_API_KEY:
        result = gemini_voice(audio_bytes, filename)
        if "error" not in result:
            return result
        return {"error": result["error"]}
    return {"error": "no provider succeeded"}


# ---------- language detection ----------
# Distinctive Telugu function words that appear in Tinglish (Telugu typed in
# Roman letters). If any of these tokens appears, treat the message as Telugu.
# Kept to words unlikely to collide with common English / brand names.
_TINGLISH_MARKERS = frozenset({
    # pronouns
    "naaku", "naku", "nuvu", "nuvvu", "nenu", "memu", "meeru", "miru",
    "vaadu", "vaaru", "vallu",
    # be-verb forms
    "undi", "unnadi", "unnay", "unnayi", "unnam", "unnava", "unnavu", "unnav",
    "untundi", "untadu", "untaru", "untayi", "untundhi", "undaala",
    # action verbs
    "ayyindi", "ayyinda", "ayyava", "ayyavu", "ayinda", "ayyaya", "ayite",
    "vacchindi", "vacchindhi", "vastunna", "vasthava", "vacchav", "vastanu",
    "cheppu", "cheppandi", "cheppava", "cheppadu", "chepthunnanu",
    "chepputhunnanu",
    "chestava", "chestaru", "chestara", "chestavu", "chestaanu",
    "chesthunnanu", "chesindhi", "chesthunnav", "cheste", "chesinna",
    "istaru", "istharu", "istundi", "ichestha", "ichhesta", "ichindi",
    # question words
    "enti", "ento", "emaindi", "ekkada", "ekkadiki", "epudu", "entha",
    "evaru", "enduku", "emaina",
    # distinctive common words (removed ambiguous "kadu", "leka" — common Indian surnames)
    "telusa", "teliyadu", "avunu", "kuda", "lekapothe",
    "taggincha", "tagginchagalava", "takkuva", "ekkuva",
    "kavali", "leru", "ledhu", "ledu",
    "epatiki", "appude", "appati", "appudu", "kanipisthundi",
    "aithe", "aipoindi", "velthunna", "chesthundi", "idhi",
    # modifiers / particles
    "matram", "matrame", "reppu", "repu", "ninna", "nethi", "rojulu",
})


def is_telugu(text):
    """Return True if text is Telugu — Telugu script OR Tinglish (Telugu typed in Roman letters)."""
    # Telugu Unicode script
    if any("ఀ" <= c <= "౿" for c in text):
        return True
    # Tinglish: at least one distinctive Telugu marker word
    words = {
        w.strip(".,!?;:()[]{}'\"-").lower()
        for w in text.split()
    }
    return bool(words & _TINGLISH_MARKERS)


# ---------- handler ----------
WELCOME = (
    "👋 Translator bot ready.\n\n"
    "• Type English → everyday Telugu to copy/paste.\n"
    "• Type Telugu  → polished English.\n"
    "• Send a Telugu voice note → Telugu transcript + English.\n\n"
    "Send me anything and I'll translate."
)


def handle_message(msg):
    chat_id = msg["chat"]["id"]
    msg_id = msg.get("message_id")

    text = (msg.get("text") or "").strip()
    if text in ("/start", "/help"):
        send_message(chat_id, WELCOME)
        return

    voice = msg.get("voice") or msg.get("audio")
    if voice:
        reserved = try_reserve("voice")
        if reserved is None:
            send_message(
                chat_id,
                f"🛑 Daily voice limit reached ({VOICE_DAILY_LIMIT}/day).\n"
                f"Resets at {_next_reset_ist()}.\n"
                f"Text translation still works.",
                reply_to=msg_id,
            )
            return
        send_chat_action(chat_id, "typing")
        filename, audio = download_voice(voice["file_id"])
        if not audio:
            release("voice")
            send_message(chat_id, "⚠️ Couldn't download the voice file.", reply_to=msg_id)
            return

        # Step 1: transcribe (Groq Whisper → Gemini multimodal fallback)
        trans = transcribe_voice(audio, filename)
        if "error" in trans:
            release("voice")
            send_message(
                chat_id,
                "⚠️ Voice transcription failed on all providers. Please try again in a minute.",
                reply_to=msg_id,
            )
            return

        te = trans.get("telugu", "").strip()
        en = trans.get("english", "").strip()  # may already be set by Gemini multimodal fallback

        # Step 2: if we have Telugu but no English yet, polish via text path
        if te and not en:
            polished = translate_text(TE_TO_EN_SYSTEM, te)
            en = "" if polished.startswith("(") else polished

        reply = ""
        if te:
            reply += f"🎙 Telugu (heard):\n{te}\n\n"
        reply += f"💬 English (copy-paste ready):\n{en or '(translation failed — try again)'}"
        reply += f"\n\n{_status_line()}"
        reply_msg_id = send_message(chat_id, reply, reply_to=msg_id, reply_markup=FEEDBACK_KEYBOARD)
        _remember_feedback(reply_msg_id, {
            "direction": "voice_te2en",
            "input_audio_filename": filename,
            "telugu_transcript": te,
            "english": en,
        })
        return

    if not text:
        send_message(chat_id, "Send me English text, Telugu text, or a Telugu voice note.")
        return

    reserved = try_reserve("text")
    if reserved is None:
        send_message(
            chat_id,
            f"🛑 Daily text limit reached ({TEXT_DAILY_LIMIT}/day).\n"
            f"Resets at {_next_reset_ist()}.",
            reply_to=msg_id,
        )
        return

    send_chat_action(chat_id, "typing")
    direction = "te2en" if is_telugu(text) else "en2te"
    system_prompt = TE_TO_EN_SYSTEM if direction == "te2en" else EN_TO_TE_SYSTEM
    out = translate_text(system_prompt, text)

    # Error strings always start with "(" (e.g., "(gemini: ...)", "(groq: ...)", "(no provider...)").
    # Translations never start with "(" because prompts forbid prefixes.
    if out.startswith("("):
        release("text")
        send_message(
            chat_id,
            "⚠️ Translation failed on all providers. Please try again in a minute.",
            reply_to=msg_id,
        )
        return

    translated = out
    out += f"\n\n{_status_line()}"
    reply_msg_id = send_message(chat_id, out, reply_to=msg_id, reply_markup=FEEDBACK_KEYBOARD)
    _remember_feedback(reply_msg_id, {
        "direction": direction,
        "input": text,
        "output": translated,
    })


# ---------- health endpoint (UptimeRobot + host keepalive) ----------
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args, **kwargs):
        return


def start_health_server(port):
    srv = HTTPServer(("0.0.0.0", port), HealthHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Health server listening on :{port}", flush=True)


# ---------- main ----------
def main():
    start_health_server(PORT)

    me = tg_call("getMe")
    if not me:
        print("ERROR: couldn't reach Telegram. Check TELEGRAM_BOT_TOKEN.", flush=True)
        sys.exit(1)
    print(f"Bot online as @{me.get('username')} ({me.get('first_name')})", flush=True)

    offset = 0
    while True:
        try:
            updates = get_updates(offset)
        except KeyboardInterrupt:
            print("\nBye.", flush=True)
            return
        except Exception as e:
            print(f"[poll error: {e}; sleeping 5s]", flush=True)
            time.sleep(5)
            continue

        for upd in updates:
            offset = upd["update_id"] + 1
            if "callback_query" in upd:
                try:
                    handle_callback(upd["callback_query"])
                except Exception as e:
                    print(f"[callback error: {e}]", flush=True)
                continue
            msg = upd.get("message") or upd.get("edited_message")
            if not msg:
                continue
            try:
                handle_message(msg)
            except Exception as e:
                print(f"[handler error: {e}]", flush=True)
                chat_id = msg.get("chat", {}).get("id")
                if chat_id:
                    send_message(chat_id, "⚠️ Internal error. Please try again.")


if __name__ == "__main__":
    main()
