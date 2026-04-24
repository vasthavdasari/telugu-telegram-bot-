#!/usr/bin/env python3
"""
Telegram Translator Bot — English <-> Colloquial Telugu (text + voice)

What it does
    - English text         -> everyday spoken Telugu (copy/paste ready)
    - Telugu text          -> polished casual English
    - Telugu voice message -> Telugu transcript + polished English

Services used (all FREE, no credit card, Google-only stack)
    - Telegram Bot API       unlimited, personal use
    - Google Gemini 2.5      1000 req/day Flash-Lite (text) / 250 req/day Flash (audio)
                             handles both translation AND voice transcription

Required environment variables
    TELEGRAM_BOT_TOKEN   from @BotFather
    GEMINI_API_KEY       from aistudio.google.com/apikey
    PORT                 optional; defaults to 8080 (health endpoint)

Stdlib only. No pip install needed.
"""

import base64
import json
import os
import pathlib
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
GEMINI_API_KEY = _require_env("GEMINI_API_KEY")
PORT = int(os.environ.get("PORT", "8080"))

# Flash-Lite for text (higher daily quota, 1000 RPD).
# Flash for audio (better multimodal quality, 250 RPD).
GEMINI_TEXT_MODEL = "gemini-2.5-flash-lite"
GEMINI_AUDIO_MODEL = "gemini-2.5-flash"

TG_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
TG_FILE = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}"


def _gemini_url(model):
    return (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:generateContent"
    )


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
EN_TO_TE_SYSTEM = """You translate English to Telugu.

RULES:
- Output ONLY the Telugu text. No prose, no explanation, no romanization,
  no quotation marks, no "Translation:" prefix.
- Use EVERYDAY SPOKEN Telugu (వాడుక భాష), NOT formal/literary/textbook Telugu.
- Avoid heavy Sanskrit-derived vocabulary. Use plain words a young person
  would say to a friend.
- Code-mixing with English is fine and natural — keep common English words
  like "meeting", "okay", "sorry", "plan", "phone" etc. if they fit the tone.
- Keep length and register similar to the English input (casual stays casual,
  urgent stays urgent).
- If the English is a single word or greeting, translate naturally — e.g.,
  "hey" -> "హే" or "ఏంటి", not formal "నమస్కారం".
- Never translate proper nouns, brand names, or URLs.

Examples:
English: "hey, running late by 20 mins, meet you at the cafe"
Telugu:  హే, 20 mins late అవుతున్నా, cafe దగ్గర కలుద్దాం

English: "did you eat?"
Telugu:  తిన్నావా?

English: "can you send the pdf when you're free"
Telugu:  free అయినప్పుడు pdf పంపగలవా"""

TE_TO_EN_SYSTEM = """You translate Telugu to English.

RULES:
- Output ONLY the English text. No prose, no explanation, no prefix.
- Translate into natural everyday casual English — like a WhatsApp message,
  not formal. Polish it so it reads as a clean chat message ready to paste.
- Fix obvious speech-to-text errors based on context (e.g., if a word looks
  garbled but the surrounding context makes the intent clear, use the right word).
- Preserve the tone and length of the original.
- If English words are already code-mixed into the Telugu, keep those as-is.
- Never translate proper nouns, brand names, or URLs."""

VOICE_PROMPT = """The audio you received is a voice message spoken in Telugu.

Do TWO things:
1. Transcribe it verbatim in Telugu script.
2. Translate the transcription into a natural, polished, casual English chat
   message ready to paste into WhatsApp.
   - Fix obvious speech errors from context.
   - Keep it casual, not formal.
   - Preserve any English words already mixed into the Telugu.
   - Never translate proper nouns, brand names, or URLs.

Return your response in THIS EXACT format, with these two lines and nothing else:
TELUGU: <the Telugu transcript>
ENGLISH: <the polished English translation>"""


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
    return tg_call("getUpdates", offset=offset, timeout=POLL_TIMEOUT) or []


def send_message(chat_id, text, reply_to=None):
    MAX = 4000
    if not text:
        text = "(empty)"
    for i in range(0, len(text), MAX):
        chunk = text[i:i + MAX]
        params = {"chat_id": chat_id, "text": chunk}
        if reply_to and i == 0:
            params["reply_to_message_id"] = reply_to
        tg_call("sendMessage", **params)


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


# ---------- gemini (text translate) ----------
def gemini_translate(system_prompt, user_text, max_retries=5):
    body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_text}]}],
        "generationConfig": {"temperature": 0.3},
    }
    url = f"{_gemini_url(GEMINI_TEXT_MODEL)}?key={urllib.parse.quote(GEMINI_API_KEY)}"
    for attempt in range(max_retries + 1):
        try:
            res = _http_json(url, payload=body, timeout=HTTP_TIMEOUT)
            cands = res.get("candidates") or []
            if not cands:
                return f"(translate error: no candidates — {res.get('promptFeedback', {})})"
            parts = cands[0].get("content", {}).get("parts") or []
            out = "".join(p.get("text", "") for p in parts).strip()
            return out or "(translate error: empty response)"
        except urllib.error.HTTPError as e:
            body_text = e.read().decode(errors="replace")
            if e.code in RETRYABLE_HTTP and attempt < max_retries:
                wait = _retry_wait(attempt)
                print(f"[gemini text {e.code}; retry {attempt+1}/{max_retries} in {wait:.0f}s]", flush=True)
                time.sleep(wait)
                continue
            return f"(translate error: HTTP {e.code} — {body_text[:200]})"
        except Exception as e:
            if attempt < max_retries:
                wait = _retry_wait(attempt)
                print(f"[gemini text {e}; retry {attempt+1}/{max_retries} in {wait:.0f}s]", flush=True)
                time.sleep(wait)
                continue
            return f"(translate error: {e})"
    return "(translate error: retries exhausted)"


# ---------- gemini (audio transcribe + translate, single call) ----------
def gemini_voice(audio_bytes, filename, max_retries=5):
    """Send Telugu audio to Gemini. Returns {telugu, english} or {error}."""
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
    url = f"{_gemini_url(GEMINI_AUDIO_MODEL)}?key={urllib.parse.quote(GEMINI_API_KEY)}"
    for attempt in range(max_retries + 1):
        try:
            res = _http_json(url, payload=body, timeout=STT_TIMEOUT)
            cands = res.get("candidates") or []
            if not cands:
                return {"error": f"no candidates — {res.get('promptFeedback', {})}"}
            parts = cands[0].get("content", {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts).strip()
            if not text:
                return {"error": "empty response"}
            telugu, english = "", ""
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("TELUGU:"):
                    telugu = line[len("TELUGU:"):].strip()
                elif line.startswith("ENGLISH:"):
                    english = line[len("ENGLISH:"):].strip()
            if not (telugu or english):
                return {"telugu": "", "english": text}
            return {"telugu": telugu, "english": english}
        except urllib.error.HTTPError as e:
            body_text = e.read().decode(errors="replace")
            if e.code in RETRYABLE_HTTP and attempt < max_retries:
                wait = _retry_wait(attempt)
                print(f"[gemini audio {e.code}; retry {attempt+1}/{max_retries} in {wait:.0f}s]", flush=True)
                time.sleep(wait)
                continue
            return {"error": f"HTTP {e.code} — {body_text[:200]}"}
        except Exception as e:
            if attempt < max_retries:
                wait = _retry_wait(attempt)
                print(f"[gemini audio {e}; retry {attempt+1}/{max_retries} in {wait:.0f}s]", flush=True)
                time.sleep(wait)
                continue
            return {"error": str(e)}
    return {"error": "retries exhausted"}


# ---------- language detection ----------
def is_telugu(text):
    return any("ఀ" <= c <= "౿" for c in text)


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
        result = gemini_voice(audio, filename)
        if "error" in result:
            release("voice")
            send_message(chat_id, f"⚠️ Voice processing failed: {result['error']}", reply_to=msg_id)
            return
        te = result.get("telugu", "").strip()
        en = result.get("english", "").strip()
        reply = ""
        if te:
            reply += f"🎙 Telugu (heard):\n{te}\n\n"
        reply += f"💬 English (copy-paste ready):\n{en or '(no translation)'}"
        reply += f"\n\n{_status_line()}"
        send_message(chat_id, reply, reply_to=msg_id)
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
    if is_telugu(text):
        out = gemini_translate(TE_TO_EN_SYSTEM, text)
    else:
        out = gemini_translate(EN_TO_TE_SYSTEM, text)
    if out.startswith("(translate error"):
        release("text")
    else:
        out += f"\n\n{_status_line()}"
    send_message(chat_id, out, reply_to=msg_id)


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
            msg = upd.get("message") or upd.get("edited_message")
            if not msg:
                continue
            try:
                handle_message(msg)
            except Exception as e:
                print(f"[handler error: {e}]", flush=True)
                chat_id = msg.get("chat", {}).get("id")
                if chat_id:
                    send_message(chat_id, f"⚠️ Internal error: {e}")


if __name__ == "__main__":
    main()
