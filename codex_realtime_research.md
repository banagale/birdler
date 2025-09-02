# Codex/Claude CLI Realtime Conversation Exfiltration (Offline Research Plan)

This report outlines a thorough, network‑free methodology to discover and enable realtime access to assistant conversation text from Codex CLI and Claude CLI, suitable for driving speech via Birdler. It avoids web lookups and relies solely on local instrumentation, observation, and wrappers.

## Objectives
- Capture assistant messages in realtime with sub‑second latency.
- Prefer zero changes to upstream CLIs; no token/context impact.
- Work reliably on macOS and Linux; no Windows in MVP.
- Maintain a provider‑agnostic, NDJSON event stream compatible with our bridge.

## Constraints
- No external network research or docs; rely on local inspection.
- Do not modify the CLIs unless strictly necessary for robustness.

## Strategy Overview
- Primary: wrap the CLI process with a PTY/line‑buffering pipeline and convert stdout/stderr to NDJSON.
- Secondary: instrument running processes to discover any native logs and tail them.
- Optional: propose a tiny “native sink” patch for Codex (if we control that code) for long‑term robustness.

ASCII overview

```
[CLI (codex|claude)] -> [PTY/line-buffer] -> [stdin_to_ndjson] -> events.ndjson -> (tail/play)
                                     \
                                      -> stdout.log (human readable)
```

## Unified Event Schema
- One line per event (NDJSON):
  - `source`: `codex` | `claude`
  - `session_id`: stable per run or conversation
  - `message_id`: unique ID per message
  - `ts`: epoch seconds
  - `role`: `assistant` | `user`
  - `text`: message text (final)
  - `is_stream_final`: `true` for final messages (MVP)
  - `meta`: freeform object

Our `tools/stdin_to_ndjson.py` emits this schema.

## Realtime Capture Without Upstream Changes

### Linux (preferred: line‑buffer)
- Use `stdbuf` or `unbuffer` to force line buffering and stream all output:
- Codex
  - `stdbuf -oL -eL codex <args> 2>&1 | tee -a dev/codex-mvp/stdout.log | poetry run python tools/stdin_to_ndjson.py --source codex --out dev/codex-mvp/events.ndjson --role assistant --strip-ansi`
- Claude
  - `stdbuf -oL -eL claude <args> 2>&1 | tee -a dev/claude-mvp/stdout.log | poetry run python tools/stdin_to_ndjson.py --source claude --out dev/claude-mvp/events.ndjson --role assistant --strip-ansi`

Notes
- Add `--match-prefix 'Assistant:'` if the CLI prefixes assistant lines.
- Use `--coalesce-blank` if messages are multi‑line separated by blanks.

### macOS (preferred: PTY)
- Some CLIs buffer when stdout is a pipe; force a PTY using `script`:
- Codex
  - `script -q /dev/stdout sh -c 'codex <args> 2>&1' | tee -a dev/codex-mvp/stdout.log | poetry run python tools/stdin_to_ndjson.py --source codex --out dev/codex-mvp/events.ndjson --role assistant --strip-ansi`
- Claude
  - `script -q /dev/stdout sh -c 'claude <args> 2>&1' | tee -a dev/claude-mvp/stdout.log | poetry run python tools/stdin_to_ndjson.py --source claude --out dev/claude-mvp/events.ndjson --role assistant --strip-ansi`

## Discovering Native Logs (Offline Instrumentation)
If CLIs already persist history, tailing those logs may be more robust. Use these OS‑level tools to find files the process opens:

### macOS
- Find PID: `pgrep -fl 'codex|claude'`
- List open files: `lsof -p <PID> | rg -i '\\.(json|log|ndjson)$'`
- Trace file open calls:
  - `fs_usage -w -f files -p <PID>` (noisy; filter on `open`/`write`)
  - `opensnoop -p <PID>` (DTrace; may require privileges)
- Watch likely dirs for writes:
  - `fs_usage` against `~/Library/Application Support`, `~/Library/Logs`, `~/.config`, `~/.local/share`, project working dir.

### Linux
- Find PID: `pgrep -fl 'codex|claude'`
- List open files: `lsof -p <PID> | rg -i '\\.(json|log|ndjson)$'`
- Trace file I/O:
  - `strace -f -e trace=file -p <PID>` (see paths with `openat`, `stat`, etc.)
  - `inotifywait -m -r ~/.config ~/.local/share ~/.cache $PWD` to see new/modified files

### What To Look For
- JSON/NDJSON files named `events`, `conversation`, `history`, `messages`, or `session`.
- Log files under hidden app dirs: `~/.codex`, `~/.claude`, `~/.config/*codex*`, `~/.local/share/*claude*`, or project‑local `.codex` folders.

### If Found
- Tail with truncate‑safe logic (our tailer already does this).
- Normalize records to our event schema (adapter if needed).

## Session Identification
- Derive `session_id` from:
  - CLI‑exposed session/run IDs (if printed).
  - Hash of start time + working directory.
  - Log path components (e.g., parent directory name of a session folder).

## Heuristic Role/Utterance Detection
- Prefer structured/JSON output if the CLI offers it (`--json`, `--quiet --json`, etc.).
- If not, use prefixes like `Assistant:` or distinct render blocks to gate.
- Keep heuristics configurable (regex allowlist), and apply conservative length limits.

## Native Sink (Optional, Codex Only)
- If we own Codex CLI code, add an env‑gated FileSink that emits NDJSON on message finalization.
- Guard with `CODEX_LOG_PATH` and ensure non‑blocking, append‑only writes.
- This is a small, maintainable upstream change; not required for MVP.

## Privacy & Safety
- Keep file access local; do not transmit transcripts.
- Redact secrets in logs if necessary (configurable filtering).
- Disable by default; opt in via wrapper command or env var.

## Validation Plan (Local)
1) Baseline: run CLI wrapped with PTY/line‑buffer and emit NDJSON. In a second terminal, run `poetry run python tools/codex_tail.py --speakable-only` and verify prompt‑like utterances appear.
2) Stress: long outputs; confirm coalescing or line‑by‑line still usable.
3) Rotation: truncate the NDJSON file while tailing; ensure tailer recovers.
4) If native logs were discovered, repeat with a tailer reading those logs directly.

## Integration With Birdler (Next)
- `tools/ai_speak_bridge.py` will tail NDJSON and invoke `birdler.py --text ...` using a configured voice.
- Playback via `ffplay`/`afplay`; disabled in headless environments.

## Quick Commands (Recap)
- MVP tailer: `poetry run python tools/codex_tail.py --speakable-only`
- Emit mock event: `poetry run python tools/codex_append.py --text "Want me to run tests?"`
- Wrap Codex (Linux): `stdbuf -oL -eL codex … | poetry run python tools/stdin_to_ndjson.py --source codex --out dev/codex-mvp/events.ndjson --role assistant --strip-ansi`
- Wrap Codex (macOS): `script -q /dev/stdout sh -c 'codex … 2>&1' | poetry run python tools/stdin_to_ndjson.py --source codex --out dev/codex-mvp/events.ndjson --role assistant --strip-ansi`

## Deliverables Present In Repo
- `tools/stdin_to_ndjson.py`: stdin → NDJSON converter with ANSI stripping and coalescing.
- `tools/codex_tail.py`: truncate‑safe tailer with “speakable” heuristics.
- `tools/codex_append.py`: simple writer for local demo/testing.
- `docs/technical-briefing-ai-audio-bridge.md`: full architecture and phased plan.

---
This plan is self‑contained and does not rely on external documentation. It uses empirical instrumentation to discover any native logs and otherwise wraps the CLIs to obtain a robust realtime text stream suitable for the Birdler speech bridge.

