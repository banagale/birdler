# Codex → Birdler Handoff Context

This file captures the current setup so a fresh Codex session can resume work immediately.

## What’s in this repo (new pieces)

- `c.sh`: Wrapper to launch Codex with session recording enabled and an optional resume flag.
  - Sets `CODEX_TUI_RECORD_SESSION=1`.
  - Sets `CODEX_TUI_SESSION_LOG_PATH="$HOME/.codex/log/codex-session.jsonl"` by default.
  - `-c` / `--continue`: resume the latest session from `~/.codex/sessions/rollout-*.jsonl` via `-c experimental_resume=<path>`.
  - Pauses for Enter before launching (so the path is readable). Skip with `-y`/`--no-pause`.

- `tools/codex_session_tail.py`: Tails Codex’s session JSONL and emits assistant finals; can synthesize and play audio.
  - Accepts a file or directory (picks `codex-session.jsonl` or latest `session-*.jsonl`).
  - Extracts final assistant messages from `payload.msg` (internally tagged enum).
  - Filters and pacing:
    - Near-duplicate suppression (`--dedupe-window`, `--dedupe-similarity`, `--no-dedupe`).
    - Speak length limits (`--speak-first-sentences`, `--speak-max-chars`, `--min-speak-chars`).
    - Backlog trimming (`--max-backlog`): keep only newest N items when behind.
    - Code handling (default: strip code):
      - `--strip-codeblocks` (default): strip fenced/inline/indented code, speak remaining prose.
      - `--skip-codeblocks`: replace with a short phrase (cached audio).
    - Headlines mode: `--speak-headlines` speaks section titles (e.g., “What changed”, “How to use”).
  - Audio:
    - `--speak`: synthesize with Birdler.
    - `--play`: auto-play via `afplay`/`ffplay`/`aplay` after synth.
    - `--cooldown`: pause after each spoken item.
  - Status: single-line ticker on stderr (`--status` on by default; `--no-status` to disable). Optional `--verbose-status` for per-event logs.
  - Fast path: `--fast` starts a persistent TTS server and sends requests for near‑instant synth. Optional `--device` (cpu|cuda|mps).

- `tools/birdler_server.py`: Persistent TTS server (stdin/stdout NDJSON)
  - Loads ChatterboxTTS once and caches prepared voice embeddings.
  - Request format: `{ "kind": "speak", "text": "...", "voice": "ripley", "out_dir": "generated-audio", "no_crossfade": true }`
  - Response: `{ "kind": "result", "ok": true, "path": "/abs/path.wav" }`

- `tools/audio_smoke.py`: Generates a short test tone and plays it via `afplay`/`ffplay`/`aplay`.

## How to resume (two terminals)

- Terminal A — launch Codex with recording:
  - `./c.sh`
  - Optional resume: `./c.sh -c`

- Terminal B — tail and speak with fast TTS:
  - `poetry run python tools/codex_session_tail.py --file ~/.codex/log/ --speak --play --fast --voice ripley --speak-headlines --speak-first-sentences 1 --speak-max-chars 200 --min-speak-chars 24 --max-backlog 3 --cooldown 0.2`
  - Notes:
    - Default strips code for speech; add `--skip-codeblocks` to speak a short cached phrase.
    - Status ticker shows `q=… last=synth(server)|play|…` on stderr.

## Voice bootstrap (one-time per voice)

- Ensure `voices/<voice>/samples/reference.wav` exists, or run Birdler once with `--audio-sample`:
  - `python birdler.py --voice ripley --audio-sample audio-samples/bigbird/bigbird_youtube_clean.wav --bootstrap-only --build-embedding`

## Known behaviors / rationale

- Session log path: we pin to `~/.codex/log/codex-session.jsonl` for stable tailing. The tailer also accepts a directory and finds the active file.
- Filtering for speech keeps pace: headlines + first sentence(s) + char limit reduce clip duration.
- Backlog trimming and dedupe prevent falling behind when messages arrive fast.
- Code handling avoids reading code aloud; default is to strip it and speak the surrounding prose.

## Quick commands

- Smoke test audio: `poetry run python tools/audio_smoke.py --play`
- Tail only (no speech): `poetry run python tools/codex_session_tail.py --file ~/.codex/log/`
- Tail + speak (fast): see Terminal B command above.

## Next steps (optional enhancements)

- Server: add a small local socket instead of stdio; batch requests if needed.
- Tailer: ETA hint based on recent synth durations; expand headline patterns; configurable phrase list.
- Birdler: explicit `--no-crossfade`/`--no-trim` when speed is prioritized (used by fast mode server).

