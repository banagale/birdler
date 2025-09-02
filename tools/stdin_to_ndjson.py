#!/usr/bin/env python3
"""
Read stdin line-by-line and emit NDJSON events to a file.

MVP usage examples:

  # Linux (line-buffered):
  stdbuf -oL -eL codex run ... 2>&1 | \
    poetry run python tools/stdin_to_ndjson.py --source codex --out dev/codex-mvp/events.ndjson --role assistant

  # macOS (PTY to avoid buffering):
  script -q /dev/stdout sh -c 'claude chat ... 2>&1' | \
    poetry run python tools/stdin_to_ndjson.py --source claude --out dev/claude-mvp/events.ndjson --role assistant

By default, every non-empty line becomes one event. Optionally, restrict to lines
that begin with a specific prefix (e.g., "Assistant:") using --match-prefix.
ANSI escape sequences can be stripped via --strip-ansi.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[ -/]*[@-~]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert stdin lines to NDJSON events")
    p.add_argument("--out", type=Path, required=True, help="Output NDJSON path")
    p.add_argument("--source", type=str, default="codex", help="Source label (codex|claude)")
    p.add_argument("--session-id", type=str, default=os.environ.get("SESSION_ID", "mvp-session"))
    p.add_argument("--role", type=str, default="assistant", help="Role to tag events with")
    p.add_argument("--match-prefix", type=str, help="Only emit lines starting with this prefix (case-insensitive)")
    p.add_argument("--strip-ansi", action="store_true", help="Strip ANSI escape codes from input lines")
    p.add_argument("--coalesce-blank", action="store_true", help="Treat blank line as message boundary (flush buffer)")
    return p.parse_args()


def clean_line(s: str, strip_ansi: bool) -> str:
    if strip_ansi:
        s = ANSI_RE.sub("", s)
    return s.rstrip("\n\r")


def emit(out_f, source: str, session_id: str, role: str, text: str) -> None:
    if not text.strip():
        return
    ev = {
        "source": source,
        "session_id": session_id,
        "message_id": str(uuid.uuid4()),
        "ts": time.time(),
        "role": role,
        "text": text.strip(),
        "is_stream_final": True,
        "meta": {},
    }
    out_f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    out_f.flush()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prefix = (args.match_prefix or "").lower()

    # Coalescing mode: accumulate until blank line, then flush
    buf: list[str] = []

    with args.out.open("a", encoding="utf-8") as out_f:
        try:
            for raw in sys.stdin:
                line = clean_line(raw, args.strip_ansi)
                if args.coalesce-blank:
                    if not line.strip():
                        if buf:
                            emit(out_f, args.source, args.session_id, args.role, " ".join(buf))
                            buf.clear()
                        continue
                    if prefix and not line.lower().startswith(prefix):
                        continue
                    buf.append(line)
                else:
                    if not line.strip():
                        continue
                    if prefix and not line.lower().startswith(prefix):
                        continue
                    emit(out_f, args.source, args.session_id, args.role, line)
            # EOF: flush buffer if coalescing
            if args.coalesce-blank and buf:
                emit(out_f, args.source, args.session_id, args.role, " ".join(buf))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

