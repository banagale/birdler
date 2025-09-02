#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import re
from pathlib import Path
from typing import Optional
import subprocess
from threading import Thread, Event, Lock
from collections import deque
from difflib import SequenceMatcher
import random
import shutil as _shutil
import subprocess as _sp


def find_latest_session_log() -> Optional[Path]:
    # Honor explicit env var if set; return the path even if it doesn't
    # exist yet so the follower can wait for creation.
    env_path = os.environ.get("CODEX_TUI_SESSION_LOG_PATH")
    if env_path:
        p = Path(os.path.expanduser(env_path))
        return p
    # Default directory under ~/.codex/log
    base = Path(os.path.expanduser("~/.codex/log"))
    if not base.is_dir():
        return None
    candidates = sorted(base.glob("session-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tail Codex TUI session JSONL and print assistant messages (optionally speak via Birdler)")
    p.add_argument("--file", type=Path, help="Path to Codex session JSONL (defaults to latest in ~/.codex/log)")
    p.add_argument("--from-start", action="store_true", help="Read from start instead of tailing new lines only")
    p.add_argument("--sleep", type=float, default=0.2, help="Polling interval seconds for new lines")
    p.add_argument("--print", dest="do_print", action="store_true", help="Print assistant messages to stdout (default)")
    p.add_argument("--no-print", dest="do_print", action="store_false")
    p.set_defaults(do_print=True)
    # Simple heuristic filter: only lines that look like prompts/questions
    p.add_argument("--speakable-only", action="store_true", help="Only forward lines with a '?' or invitational phrases")
    # Birdler invocation options
    p.add_argument("--speak", action="store_true", help="Invoke birdler.py to synthesize spoken audio")
    p.add_argument("--play", action="store_true", help="Play the synthesized WAV after --speak (afplay/ffplay/aplay)")
    p.add_argument("--fast", action="store_true", help="Use persistent server for low-latency TTS and pass speed-friendly defaults")
    # De-duplication of near-identical lines
    p.add_argument("--dedupe-window", type=int, default=5, help="How many recent lines to compare for near-duplicates")
    p.add_argument(
        "--dedupe-similarity",
        type=float,
        default=0.92,
        help="Similarity ratio (0..1) above which a line is considered a duplicate",
    )
    p.add_argument("--no-dedupe", action="store_true", help="Disable near-duplicate suppression")
    p.add_argument("--voice", type=str, help="Managed voice name to use with birdler.py")
    p.add_argument("--output-dir", type=Path, default=Path("generated-audio"), help="Output dir for WAVs when speaking")
    p.add_argument("--atempo", type=float, help="Optional --atempo for birdler playback speed")
    p.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], help="Preferred device for TTS (fast mode only)")
    # Skip code blocks entirely and speak a short notice instead
    p.add_argument("--skip-codeblocks", action="store_true", help="If a message contains a fenced/inline/indented code block, replace with a short spoken notice")
    # Default behavior: strip code blocks and speak surrounding prose
    p.add_argument("--strip-codeblocks", dest="strip_codeblocks", action="store_true", default=True,
                   help="Strip code blocks and speak remaining prose (default)")
    p.add_argument("--no-strip-codeblocks", dest="strip_codeblocks", action="store_false",
                   help="Do not strip code blocks (only use with --skip-codeblocks or for raw text)")
    p.add_argument("--no-phrase-cache", action="store_true", help="Disable caching/reusing short notice audio")
    # Optional status prints
    p.add_argument("--verbose-status", action="store_true", help="Print queue and processing status lines")
    p.add_argument("--status", dest="status", action="store_true", default=True, help="Show a single-line live status ticker (default)")
    p.add_argument("--no-status", dest="status", action="store_false", help="Disable the live status ticker")
    p.add_argument("--status-interval", type=float, default=0.5, help="Status ticker refresh interval in seconds")
    # Speaking length controls
    p.add_argument("--speak-max-chars", type=int, default=0, help="Max characters to send to TTS (0=disabled)")
    p.add_argument("--speak-first-sentences", type=int, default=0, help="Speak only the first N sentences (0=disabled)")
    p.add_argument("--min-speak-chars", type=int, default=12, help="Minimum length to speak; if too short, try to include more sentences or skip")
    p.add_argument("--max-backlog", type=int, default=4, help="Max queued items to keep for speech; older ones are dropped")
    p.add_argument("--cooldown", type=float, default=0.0, help="Optional pause (seconds) after each spoken item")
    p.add_argument("--speak-headlines", action="store_true", help="If a message is long, speak only key section headlines (e.g., 'What changed', 'How to use')")
    return p.parse_args()


ALLOW_RE = re.compile(r"(?i)\b(want me to|shall i|should i|do you want me|would you like me)\b")


def is_speakable(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if "?" in t:
        return True
    return bool(ALLOW_RE.search(t))


def extract_assistant_text(obj: dict) -> Optional[str]:
    # Expect: { ts, dir, kind, payload: { id, msg: { AgentMessage: { message: "..." } } } }
    if not isinstance(obj, dict):
        return None
    if obj.get("kind") != "codex_event":
        return None
    payload = obj.get("payload") or {}
    msg = payload.get("msg") if isinstance(payload, dict) else None
    if isinstance(msg, dict):
        # Case 1: internally tagged enum (serde) => {"type":"agent_message","message":"..."}
        msg_type = msg.get("type")
        if isinstance(msg_type, str) and msg_type.lower() == "agent_message":
            m = msg.get("message")
            if isinstance(m, str):
                return m
        # Case 2: externally tagged variant (rare) => {"AgentMessage":{"message":"..."}}
        if "AgentMessage" in msg and isinstance(msg["AgentMessage"], dict):
            m = msg["AgentMessage"].get("message")
            if isinstance(m, str):
                return m
        # Case 3: nested event naming => {"AgentMessageEvent":{"message":"..."}}
        if "AgentMessageEvent" in msg and isinstance(msg["AgentMessageEvent"], dict):
            m = msg["AgentMessageEvent"].get("message")
            if isinstance(m, str):
                return m
    return None


def follow(path: Path, from_start: bool, interval: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # wait for file to appear
        while not path.exists():
            time.sleep(interval)
    with path.open("r", encoding="utf-8") as f:
        if not from_start:
            f.seek(0, os.SEEK_END)
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                time.sleep(interval)
                # Handle truncation/rotation
                try:
                    if path.stat().st_size < pos:
                        f.seek(0, os.SEEK_SET)
                except FileNotFoundError:
                    # File rotated; reopen latest
                    break
                continue
            yield line


def speak_with_birdler(text: str, voice: Optional[str], out_dir: Path, atempo: Optional[float]) -> tuple[int, Optional[Path]]:
    cmd = [sys.executable, "birdler.py", "--text", text, "--output-dir", str(out_dir)]
    if voice:
        cmd += ["--voice", voice]
    if atempo is not None:
        cmd += ["--atempo", str(atempo)]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        out_path = None
        # Parse a line like: "Saved output to: /path/file.wav (NNN bytes)"
        for stream in (proc.stdout or "", proc.stderr or ""):
            for line in stream.splitlines():
                if "Saved output to:" in line:
                    # Extract the path between the colon and the first parenthesis if present
                    try:
                        s = line.split("Saved output to:", 1)[1].strip()
                        s = s.split(" (", 1)[0].strip()
                        out_path = Path(s)
                        break
                    except Exception:
                        pass
            if out_path:
                break
        return proc.returncode, out_path
    except Exception:
        return 1, None


def play_wav(path: Path) -> bool:
    import shutil
    players = [
        ("afplay", ["afplay", str(path)]),
        ("ffplay", ["ffplay", "-v", "quiet", "-nodisp", "-autoexit", str(path)]),
        ("aplay", ["aplay", str(path)]),
    ]
    for name, cmd in players:
        if shutil.which(name):
            try:
                r = subprocess.run(cmd, check=False)
                return r.returncode == 0
            except Exception:
                return False
    print("[warn] No audio player found (looked for afplay, ffplay, aplay)", file=sys.stderr)
    return False


def _canon(s: str) -> str:
    # Lowercase, collapse whitespace, trim punctuation at ends
    t = (s or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"^[\s\-–—:,.!?]+|[\s\-–—:,.!?]+$", "", t)
    return t


def is_near_duplicate(text: str, recent: deque[str], thresh: float) -> bool:
    a = _canon(text)
    for b in recent:
        if not b:
            continue
        r = SequenceMatcher(None, a, b).ratio()
        if r >= thresh:
            return True
    return False


CODEBLOCK_FENCED_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
CODEBLOCK_INDENT_RE = re.compile(r"(?:^|\n)(?: {4}|\t).+", re.MULTILINE)
CODEBLOCK_INLINE_RE = re.compile(r"`[^`]+`")

# Short phrases to speak instead of code blocks
PHRASES = [
    "There is a code block here; skipping.",
    "Skipping a code block here.",
    "I'll skip reading this code block.",
]


def has_codeblock(text: str) -> bool:
    t = text or ""
    if CODEBLOCK_FENCED_RE.search(t):
        return True
    if CODEBLOCK_INDENT_RE.search(t):
        return True
    # consider inline code a code block if there are 2+ occurrences or if long
    inlines = CODEBLOCK_INLINE_RE.findall(t)
    if len(inlines) >= 2:
        return True
    if inlines and sum(len(x) for x in inlines) >= 40:
        return True
    return False


def choose_phrase() -> str:
    return random.choice(PHRASES)


def phrase_cache_path(base_dir: Path, voice: Optional[str], atempo: Optional[float], phrase: str) -> Path:
    v = voice or "default"
    t = f"atempo-{atempo}" if atempo is not None else "atempo-1.0"
    slug = re.sub(r"[^a-z0-9\-]+", "-", phrase.lower()).strip("-")[:40] or "phrase"
    return base_dir / "_phrases" / v / t / f"{slug}.wav"


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"([.!?])", text)
    if len(parts) == 1:
        return [text.strip()] if text.strip() else []
    out = []
    for i in range(0, len(parts) - 1, 2):
        s = (parts[i] + parts[i + 1]).strip()
        if s:
            out.append(s)
    if len(parts) % 2 == 1 and parts[-1].strip():
        out.append(parts[-1].strip())
    return out


def limit_for_speech(text: str, max_chars: int, first_sents: int) -> str:
    t = text.strip()
    if not t:
        return t
    chosen = t
    if first_sents and first_sents > 0:
        sents = _split_sentences(t)
        chosen = " ".join(sents[:first_sents]) if sents else t
    if max_chars and max_chars > 0 and len(chosen) > max_chars:
        cut = chosen.rfind(" ", 0, max_chars)
        cut = cut if cut != -1 else max_chars
        chosen = chosen[:cut].rstrip() + "…"
    return chosen


def strip_codeblocks(text: str) -> str:
    # Remove fenced, inline (backticked), and indented code segments; collapse whitespace
    t = CODEBLOCK_FENCED_RE.sub(" ", text or "")
    t = CODEBLOCK_INLINE_RE.sub(" ", t)
    # Remove indented code lines
    t = re.sub(r"(?:^|\n)(?: {4}|\t).+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


HEADLINE_RE = re.compile(r"\b(what changed|how to use|usage|notes|summary|next steps|recommend(?:ed)? run|quick start)\b", re.I)


def extract_headlines(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    hits: list[str] = []
    for ln in lines:
        m = HEADLINE_RE.search(ln)
        if m:
            s = m.group(1).lower()
            # normalize capitalization
            s = " ".join(w.capitalize() for w in s.split())
            if s not in hits:
                hits.append(s)
    if hits:
        return ". ".join(hits) + "."
    return ""


def main() -> int:
    args = parse_args()
    log_path: Optional[Path] = args.file
    if log_path is None:
        log_path = find_latest_session_log()
    if not log_path:
        print("Error: could not find a Codex session log. Set CODEX_TUI_RECORD_SESSION=1 and start Codex.", file=sys.stderr)
        return 2
    # Sanitize accidental newlines in path input (common when a shell wraps)
    raw = str(log_path)
    if "\n" in raw or "\r" in raw:
        cleaned = raw.replace("\n", "").replace("\r", "")
        if cleaned != raw:
            print(f"[warn] Provided path contained newlines; using: {cleaned}", file=sys.stderr)
        log_path = Path(cleaned)
    # If a directory was provided, choose a file within it.
    if log_path.is_dir():
        dir_path = log_path
        # Prefer fixed file name used by c.sh
        pinned = dir_path / "codex-session.jsonl"
        if pinned.exists() or os.environ.get("CODEX_TUI_SESSION_LOG_PATH"):
            log_path = pinned
        else:
            # Fallback to latest rotating session-*.jsonl
            candidates = sorted(dir_path.glob("session-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            log_path = candidates[0] if candidates else pinned
    print(f"[tail] {log_path}")

    recent = deque(maxlen=max(1, args.dedupe_window))

    # Background speech worker with backlog trimming
    speak_lock: Optional[Lock] = Lock() if args.speak else None
    speak_q: Optional[deque[str]] = deque() if args.speak else None
    stop_evt: Optional[Event] = Event() if args.speak else None
    # Status shared state
    status_lock: Lock = Lock()
    status = {
        "queue": 0,
        "last": "idle",
        "result": "",
    }

    # Server process and simple request/response helpers (fast mode)
    server_proc: Optional[_sp.Popen] = None
    def server_start() -> None:
        nonlocal server_proc
        if server_proc is not None:
            return
        cmd = [sys.executable, "tools/birdler_server.py"]
        if args.device:
            cmd += ["--device", args.device]
        cmd += ["--voice-dir", str(Path("voices"))]
        cmd += ["--no-crossfade"]
        server_proc = _sp.Popen(cmd, stdin=_sp.PIPE, stdout=_sp.PIPE, stderr=_sp.DEVNULL, text=True, bufsize=1)

    def server_say(text: str) -> Optional[Path]:
        assert server_proc and server_proc.stdin and server_proc.stdout
        req = {"kind": "speak", "text": text, "voice": args.voice, "out_dir": str(args.output_dir), "no_crossfade": True}
        try:
            server_proc.stdin.write(json.dumps(req) + "\n")
            server_proc.stdin.flush()
            line = server_proc.stdout.readline()
            if not line:
                return None
            res = json.loads(line)
            if res.get("ok") and res.get("path"):
                return Path(res["path"])  # type: ignore[arg-type]
        except Exception:
            return None
        return None

    def _speech_worker():
        assert speak_q is not None and speak_lock is not None and stop_evt is not None
        while not stop_evt.is_set():
            item = None
            with speak_lock:
                # Trim backlog to keep only the most recent N
                max_bl = max(1, args.max_backlog)
                if len(speak_q) > max_bl:
                    drop = len(speak_q) - max_bl
                    for _ in range(drop):
                        speak_q.popleft()
                if speak_q:
                    item = speak_q.popleft()
            if item is None:
                time.sleep(max(0.05, min(1.0, args.sleep)))
                continue
            try:
                # If this is a short phrase and caching is enabled, try cache first
                if (item in PHRASES) and (not args.no_phrase_cache):
                    cache_path = phrase_cache_path(args.output_dir, args.voice, args.atempo, item)
                    if cache_path.exists():
                        if args.verbose_status:
                            print(f"[speech] play cached phrase: {cache_path}")
                        with status_lock:
                            status["last"] = "play-cached"
                        if args.play:
                            play_wav(cache_path)
                        if args.cooldown > 0:
                            time.sleep(args.cooldown)
                        continue
                    # Not cached yet → synthesize once and cache
                    with status_lock:
                        status["last"] = "synth-phrase"
                    if args.fast:
                        if server_proc is None:
                            server_start()
                        wav_path = server_say(item)
                        rc = 0 if wav_path else 1
                    else:
                        rc, wav_path = speak_with_birdler(item, args.voice, cache_path.parent, args.atempo)
                    if rc == 0 and wav_path and wav_path.exists():
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            _shutil.move(str(wav_path), str(cache_path))
                        except Exception:
                            cache_path = wav_path
                        if args.play:
                            play_wav(cache_path)
                        if args.cooldown > 0:
                            time.sleep(args.cooldown)
                        continue
                # Regular path: synthesize and optionally play
                if args.verbose_status:
                    print("[speech] synth")
                with status_lock:
                    status["last"] = "synth(server)" if args.fast else "synth"
                if args.fast:
                    if server_proc is None:
                        server_start()
                    wav_path = server_say(item)
                    rc = 0 if wav_path else 1
                else:
                    rc, wav_path = speak_with_birdler(item, args.voice, args.output_dir, args.atempo)
                if rc == 0 and args.play and wav_path and wav_path.exists():
                    play_wav(wav_path)
                    with status_lock:
                        status["last"] = "play"
                else:
                    with status_lock:
                        status["last"] = f"rc={rc}"
                if args.cooldown > 0:
                    time.sleep(args.cooldown)
            except Exception:
                # Keep worker alive
                pass

    worker: Optional[Thread] = None
    if args.speak:
        worker = Thread(target=_speech_worker, name="speech-worker", daemon=True)
        worker.start()

    # Status ticker thread
    def _status_ticker():
        while True:
            if stop_evt is not None and stop_evt.is_set():
                break
            if not args.status:
                time.sleep(args.status_interval)
                continue
            q = 0
            last = "idle"
            with status_lock:
                q = len(speak_q) if speak_q is not None else 0
                status["queue"] = q
                last = status.get("last", "idle")
            msg = f"[status] q={q} last={last}        "
            # Write to stderr to avoid mixing with printed assistant text
            try:
                sys.stderr.write("\r" + msg)
                sys.stderr.flush()
            except Exception:
                pass
            time.sleep(max(0.1, args.status_interval))

    ticker: Optional[Thread] = None
    if args.speak or args.status:
        ticker = Thread(target=_status_ticker, name="status-ticker", daemon=True)
        ticker.start()

    try:
        for raw in follow(log_path, args.from_start, args.sleep):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = extract_assistant_text(obj)
            if not text:
                continue
            if not args.no_dedupe and is_near_duplicate(text, recent, max(0.0, min(1.0, args.dedupe_similarity))):
                continue
            if args.speakable_only and not is_speakable(text):
                continue
            if args.do_print:
                print(text, flush=True)
            if args.speak and speak_q is not None and speak_lock is not None:
                # Apply speaking limits to keep latency reasonable
                # Optionally replace code blocks with a short phrase
                processed_text = text
                if args.skip_codeblocks and has_codeblock(processed_text):
                    tts_text = choose_phrase()
                else:
                    if args.strip_codeblocks and has_codeblock(processed_text):
                        processed_text = strip_codeblocks(processed_text)
                    # headline-only option for long text
                    tts_text = ""
                    if args.speak_headlines:
                        tts_text = extract_headlines(processed_text)
                    # Fallback to first sentences if no headlines were found
                    if not tts_text:
                        tts_text = limit_for_speech(processed_text, args.speak_max_chars, args.speak_first_sentences)
                    # If too short, try including more sentences up to 4
                    if len(tts_text) < max(0, args.min_speak_chars):
                        for n in (2, 3, 4):
                            tts_text = limit_for_speech(processed_text, args.speak_max_chars, n)
                            if len(tts_text) >= max(0, args.min_speak_chars):
                                break
                if not tts_text:
                    continue
                with speak_lock:
                    speak_q.append(tts_text)
                    if args.verbose_status:
                        print(f"[speech] queued (q={len(speak_q)})")
            # Update recent after accepting the text
            if not args.no_dedupe:
                recent.append(_canon(text))
    except KeyboardInterrupt:
        pass
    finally:
        if stop_evt is not None:
            stop_evt.set()
        if worker is not None:
            worker.join(timeout=1.5)
        # Stop server
        if server_proc is not None:
            try:
                server_proc.kill()
            except Exception:
                pass
        if ticker is not None:
            try:
                sys.stderr.write("\n")
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
