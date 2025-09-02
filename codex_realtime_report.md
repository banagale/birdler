# Codex CLI Realtime Conversation Exfiltration (Local Repo Analysis)

This report documents how to access Codex conversation text in realtime by inspecting the Codex source at `/Users/rob/code/open-source/codex` (no web lookups). It covers what is already persisted today, exact file locations, and the safest insertion points for a pluggable NDJSON sink for assistant outputs.

## Findings (Concrete)

- TUI trace log (for debugging):
  - Path: `~/.codex/log/codex-tui.log`
  - Source: docs note in `docs/advanced.md`.
  - Usage: `tail -F ~/.codex/log/codex-tui.log`

- Global message history (JSONL):
  - Path: `~/.codex/history.jsonl`
  - Module: `codex-rs/core/src/message_history.rs`
  - Schema (one record per line):
    - `{ "session_id": "<uuid>", "ts": <unix_seconds>, "text": "<message>" }`
  - Accessors:
    - Append: `append_entry(text, session_id, config)`
    - Metadata: `history_metadata(config) -> (log_id, entry_count)`
    - Lookup by offset: `lookup(log_id, offset, config)`
  - Who writes to it today: the TUI only appends user text messages when submitted.
    - Call path: `tui/src/chatwidget.rs::submit_user_message()` → `Op::AddToHistory { text }` → `core/src/codex.rs` handles `Op::AddToHistory` → `message_history::append_entry`.
    - Implication: `history.jsonl` currently contains only user inputs, not assistant outputs.

- Built‑in session recorder (JSONL):
  - Env flags:
    - `CODEX_TUI_RECORD_SESSION=1` → enable
    - `CODEX_TUI_SESSION_LOG_PATH=/abs/path/to/file.jsonl` (optional)
  - Default path resolution: `codex_core::config::log_dir` → `~/.codex/log` then `session-<timestamp>.jsonl`.
  - Module: `codex-rs/tui/src/session_log.rs` (on by env, no code change required)
  - What it logs:
    - App events flowing to/from TUI, including `AppEvent::CodexEvent(Event)` where `Event` carries the assistant messages (e.g., `EventMsg::AgentMessage { message }`).
    - For UI inserts it logs counts only (to avoid heavy payloads), but Codex events themselves include the assistant text.
  - Practical outcome: this session JSONL stream is the best zero‑change source for realtime assistant text.

- Assistant message emission (finals and deltas):
  - Core emits final text as `EventMsg::AgentMessage(AgentMessageEvent { message })` when processing `ResponseItem::Message` content.
    - Code: `codex-rs/core/src/codex.rs` (see `handle_response_item` and finalization blocks ~1900+).
  - TUI receives via `chatwidget.rs::on_agent_message(message: String)` and renders through the streaming controller.
    - Code: `codex-rs/tui/src/chatwidget.rs` and `tui/src/streaming/controller.rs`.

## Zero‑Change Realtime Capture (Recommended)

Enable the session recorder and tail the emitted JSONL file.

- Export env and run Codex:
  - `export CODEX_TUI_RECORD_SESSION=1`
  - Optional: `export CODEX_TUI_SESSION_LOG_PATH="$HOME/.codex/log/codex-session.jsonl"`
  - Launch Codex normally.

- Tail in another terminal and filter for assistant messages:
  - `tail -F "$HOME/.codex/log/codex-session.jsonl" | jq -r 'select(.kind=="codex_event" and .payload.msg.AgentMessage) | .payload.msg.AgentMessage.message'`

- Feed into Birdler (example):
  - `tail -F "$HOME/.codex/log/codex-session.jsonl" | jq -r 'select(.kind=="codex_event" and .payload.msg.AgentMessage) | .payload.msg.AgentMessage.message' | while IFS= read -r line; do [ -n "$line" ] && poetry run python birdler.py --voice tuba --text "$line" --output-dir generated-audio; done`

Notes:
- The exact JSON shape under `.payload.msg` is a Rust enum; depending on serializer it appears as `{ "AgentMessage": { "message": "..." } }`. The jq filter above assumes that variant tagging.
- If you prefer not to set `CODEX_TUI_SESSION_LOG_PATH`, the recorder writes to `~/.codex/log/session-<timestamp>.jsonl`; identify it by `ls -t ~/.codex/log/session-*.jsonl | head -n1`.

## Minimal Sink (Optional, Very Low Risk)

If you want explicit, durable NDJSON for assistant finals (not just a session recorder dump), there are two surgical insertion points:

1) TUI streaming final hook
- Location: `codex-rs/tui/src/streaming/controller.rs::apply_final_answer()`
- Rationale: called exactly once per assistant final; has full `message` string.
- Change: behind an env/config gate, write a single NDJSON line per final:
  - `{ "source": "codex", "session_id": "<uuid>", "role": "assistant", "ts": <unix>, "text": "..." }`
- Pros: accurate, low duplication; unaffected by deltas; UI‑scoped.
- Cons: only active in TUI (not headless `codex exec`).

2) Core event emission hook
- Location: `codex-rs/core/src/codex.rs` where `EventMsg::AgentMessage` is sent.
- Rationale: centralizes across TUI and non‑interactive flows; you can gate on `disable_response_storage` or a new `sinks.events_path`.
- Pros: works for both TUI and `codex exec`.
- Cons: core change; must avoid blocking or impacting latency (use single `write(2)` append and spawn_blocking like `message_history`).

Recommended implementation traits
- Mirror `message_history.rs` patterns: owner‑only `0600` perms, `O_APPEND`, single `write(2)`, advisory file locks with retry.
- Config surface:
  - Env: `CODEX_EVENTS_PATH=/path/events.ndjson` (simple; off when unset)
  - Or config: `[sinks] events_path = "~/.codex/events.ndjson"`

## What Not To Use

- `~/.codex/history.jsonl` is user‑only today. It does not include assistant messages, by design. Use the session recorder or add a sink.
- `~/.codex/log/codex-tui.log` is for tracing; it does not carry structured message text reliably.

## Wrapper Fallback (No Code, No Env)

For environments where setting env is undesirable, you can wrap Codex and convert stdout/stderr into NDJSON (works best for non‑interactive `codex exec`):

- Linux (line‑buffered):
  - `stdbuf -oL -eL codex exec "…" 2>&1 | poetry run python tools/stdin_to_ndjson.py --source codex --out dev/codex-mvp/events.ndjson --role assistant --strip-ansi`
- macOS (PTY):
  - `script -q /dev/stdout sh -c 'codex exec "…" 2>&1' | poetry run python tools/stdin_to_ndjson.py --source codex --out dev/codex-mvp/events.ndjson --role assistant --strip-ansi`

Note: wrapping the interactive TUI is noisy because it renders a full terminal UI; prefer the session recorder for TUI runs.

## Implementation Map (Pointers)

- Session recorder (enable today): `codex-rs/tui/src/session_log.rs`
  - Env: `CODEX_TUI_RECORD_SESSION`, optional path `CODEX_TUI_SESSION_LOG_PATH`
  - Writes JSONL with event payloads; includes assistant text via `AppEvent::CodexEvent(Event)`

- Global message history (user text): `codex-rs/core/src/message_history.rs`
  - File: `~/.codex/history.jsonl`
  - `append_entry()` controlled by `[history] persistence = "save-all"|"none"` in `~/.codex/config.toml`

- Assistant final event emission (core): `codex-rs/core/src/codex.rs`
  - Emits `EventMsg::AgentMessage(AgentMessageEvent { message })`

- TUI final message handling: `codex-rs/tui/src/chatwidget.rs`
  - `on_agent_message(message)` calls `stream.apply_final_answer(message, sink)`

- Streaming controller (best TUI sink hook): `codex-rs/tui/src/streaming/controller.rs`
  - `apply_final_answer()` and `finalize()` commit one batch per final; easy to append NDJSON here.

## Suggested Next Steps

- Short term (no code):
  - Set `CODEX_TUI_RECORD_SESSION=1`, run Codex, and tail the emitted session `.jsonl` for assistant text. I can provide a small jq or Python filter tailored to your exact payload.

- Medium term (tiny code patch):
  - Add a gated NDJSON sink at `tui/src/streaming/controller.rs::apply_final_answer()`.
  - Or add a core‑level sink around `EventMsg::AgentMessage` emission for both TUI and `codex exec`.

- Long term:
  - Expose a formal sink config in `~/.codex/config.toml` under `[sinks]`, with transport choices (file NDJSON, FIFO, or UDS). Default off.

---
This plan uses only the local Codex repository and provides exact filepaths and insertion points you can use immediately for reliable realtime access to assistant conversation text, ideal for feeding into Birdler for TTS.
