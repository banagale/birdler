#!/usr/bin/env python3
"""
Tail an NDJSON stream and print events in real time.

Usage:
  poetry run python tools/codex_tail.py
  poetry run python tools/codex_tail.py --file dev/codex-mvp/events.ndjson --speakable-only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tail chat NDJSON and print events")
    p.add_argument("--file", type=Path, default=Path("dev/codex-mvp/events.ndjson"), help="NDJSON file to tail")
    p.add_argument("--speakable-only", action="store_true", help="Only print assistant prompts/questions")
    p.add_argument("--poll", type=float, default=0.2, help="Polling interval in seconds")
    return p.parse_args()


ALLOW_RE = re.compile(r"(?i)\b(want me to|shall i|should i|do you want me|would you like me)\b")


def is_speakable(ev: dict) -> bool:
    if ev.get("role") != "assistant":
        return False
    text = (ev.get("text") or "").strip()
    if not text:
        return False
    if "?" in text:
        return True
    return bool(ALLOW_RE.search(text))


def follow(path: Path, poll: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # Create empty file so tailer can start immediately
        path.touch()
    with path.open("r", encoding="utf-8") as f:
        # Seek to end to only read new lines
        f.seek(0, os.SEEK_END)
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                time.sleep(poll)
                # Handle truncation
                if path.stat().st_size < pos:
                    f.seek(0, os.SEEK_SET)
                continue
            yield line


def main() -> None:
    args = parse_args()
    try:
        for line in follow(args.file, args.poll):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                print(line)
                continue
            if args.speakable-only and not is_speakable(ev):
                continue
            ts = time.strftime("%H:%M:%S", time.localtime(ev.get("ts", time.time())))
            src = ev.get("source", "?")
            role = ev.get("role", "?")
            text = (ev.get("text") or "").strip().replace("\n", " ")
            print(f"[{ts}] {src}/{role}: {text}")
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

