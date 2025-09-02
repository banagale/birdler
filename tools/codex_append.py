#!/usr/bin/env python3
"""
Append a Codex-style chat event (NDJSON) for MVP streaming tests.

Usage:
  poetry run python tools/codex_append.py --text "Hello world"
  poetry run python tools/codex_append.py --file dev/codex-mvp/events.ndjson --role assistant --text "Run tests?"
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Append a chat event to an NDJSON file")
    p.add_argument("--file", type=Path, default=Path("dev/codex-mvp/events.ndjson"), help="Output NDJSON path")
    p.add_argument("--source", type=str, default="codex", help="Source label (codex|claude)")
    p.add_argument("--session-id", type=str, default=os.environ.get("CODEX_SESSION_ID", "mvp-session"))
    p.add_argument("--role", type=str, default="assistant", help="Role (assistant|user)")
    p.add_argument("--text", type=str, required=True, help="Message text to record")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.file.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "source": args.source,
        "session_id": args.session_id,
        "message_id": str(uuid.uuid4()),
        "ts": time.time(),
        "role": args.role,
        "text": args.text,
        "is_stream_final": True,
        "meta": {},
    }
    with args.file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

