# AI Conversation → Speech Bridge (Technical Briefing)

This document proposes a new feature that reads AI coding session messages in near real time (from Codex CLI or Claude Code) and speaks selected assistant utterances using Birdler.

Goals:
- Attach to either Codex CLI or Claude Code conversation history in real time.
- Detect “speakable” assistant outputs (e.g., questions/prompts like “Want me to run the test suite now?”).
- Use `birdler.py` to synthesize and optionally play audio over the system’s speakers.
- Be reliable, low-latency, and minimally invasive to existing tools.

Assumptions & Constraints:
- Runtime: Python >= 3.12 (project standard).
- No heavy model inference in the watcher; Birdler handles TTS; the bridge performs I/O and light parsing.
- Claude Code stores JSON conversation history under `~/.claude/**` (exact shape varies by version).
- Codex CLI may not persist logs by default; we can add a lightweight emitter or rely on a wrapper.
- Cross‑platform target: macOS and Linux. Windows is out-of-scope for v1, but not precluded.
- Avoid blocking the terminal session; the bridge runs as a separate process/daemon.


## High-Level Architecture

Components:
- Source adapters: read messages from external tools (Claude Code, Codex CLI).
- Event bus: in-process queue to decouple ingestion from synthesis/playback.
- Intent filter: classifies “speakable” utterances and extracts a single sentence.
- TTS orchestrator: invokes Birdler with configured voice and options.
- Audio playback: plays the generated WAV (ffplay/afplay) or leaves file for user.
- State store: de-duplicates messages and remembers last processed offsets per session.

ASCII overview:

```
          +----------------+          +----------------+          +-----------------+
Claude -->| Claude Adapter |--events->|                 |   --->  | Intent Filter   |--speakable-->[Queue]
Code      +----------------+          |                 |  /       +-----------------+
                                       |    Event Bus    | /
Codex -->+----------------+            |   (async.Queue) |/        +-----------------+       +-----------------+
CLI       | Codex Adapter |--events--->|                 |----->   | TTS Orchestrator|-----> |  Birdler CLI    |
          +----------------+           +-----------------+          +-----------------+       +-----------------+
                                                                                 |                        |
                                                                                 v                        v
                                                                       (WAV path/output-dir)       (ffplay/afplay)
                                                                                 |                        |
                                                                                 +-----------playback-----+
```


## Data Model and Event Schema

Unified event schema we normalize to from each source:

```
Event {
  source: "claude" | "codex"
  session_id: str          # stable per conversation
  message_id: str          # unique id to de-dupe
  ts: float                # epoch seconds when finalized
  role: "assistant" | "user" | ...
  text: str                # full text of the assistant chunk or final message
  is_stream_final: bool    # true when message no longer changes
  meta: dict               # source-specific metadata
}
```

Speakable selection output:

```
SpeakTask {
  text: str                # succinct, one-sentence content to speak
  priority: int            # optional future use
  session_id: str
  source: "claude" | "codex"
  message_id: str
}
```


## Source Adapters

### Claude Code Adapter

Discovery:
- Look under `~/.claude/**` for per-session directories or a global store (patterns vary by release).
- Heuristics: pick the most recently modified `conversation.json`, `messages.json`, or `events.ndjson`.
- Provide an override via `CLAUDE_STORE_DIR` env var for reliability.

Ingestion:
- Prefer newline-delimited JSON (NDJSON) streams if available; else tail JSON array files with truncate-aware logic.
- Use `watchdog` (filesystem events) where available; else fallback to adaptive polling with backoff.
- Parse items and map to unified `Event` objects. Identify assistant role fields and “final” signals (e.g., `stop_reason`, `final: true`, or end-of-stream markers).

De-duplication:
- Use `message_id` or derive a stable hash from `(session_id, role, text[:N], ts_rounded)`.
- Maintain an in-memory LRU set per session and an on-disk `state.json` with last offsets.

Streaming handling:
- Buffer partial chunks; only emit when `is_stream_final=True`. As a fallback, emit on no change for X ms.

### Codex CLI Adapter

Two strategies; support both:

1) Cooperative logging (preferred):
   - Add an opt-in env var to Codex CLI (e.g., `CODEX_LOG_PATH=~/.codex/sessions/<id>/events.ndjson`).
   - Codex CLI appends NDJSON per assistant message with a stable `message_id`.
   - The bridge tails this file like the Claude adapter.

2) Wrapper mode (no changes to Codex CLI):
   - Run Codex CLI under a PTY wrapper that tees stdout to an NDJSON file (e.g., `script`/`ptyprocess`).
   - Apply a line-oriented parser that recognizes assistant speaker prefixes (e.g., “Assistant: …”).
   - This is less robust; include toggles and clear docs.

Either path normalizes to `Event`.


## Intent Filter (Speakability)

Purpose: reduce noise by only speaking messages likely to be actionable prompts or brief status updates.

Rules (v1 heuristics, no ML):
- Only role == "assistant".
- Extract the first sentence containing a question mark `?` OR matching invitational phrases:
  - Regex list: `(?i)\\b(want me to|shall I|should I|do you want me|would you like me)\\b`.
- Hard caps: 3–20 words; if longer, truncate to sentence boundary and append ellipsis optionally.
- Cooldown per session to avoid rapid-fire TTS: e.g., min 1.5s between enqueues.
- De-duplicate same text back-to-back (window of last 5 speak tasks per session).

Future (optional):
- Lightweight text ranking (e.g., keyphrase match for “tests”, “build”, “deploy”).
- Configurable keyword whitelist/blacklist.


## TTS Orchestrator and Playback

Birdler invocation examples:
- Direct text: `poetry run python birdler.py --voice <name> --text "Hello" --output-dir generated-audio`
- Existing voice prompt/bootstrap controlled by `--audio-sample` + `--voice` during initial setup.

Orchestrator behavior:
- Serialize tasks from the event queue to avoid overlapping playback.
- Generate output filename slugs based on message and session.
- Respect Birdler flags exposed today (e.g., `--atempo`, `--normalize`, `--seed`, `--workers`, validation flags as needed).

Playback options (macOS/Linux):
- Prefer `ffplay -nodisp -autoexit` when `ffmpeg` present.
- macOS fallback: `afplay <wav>`.
- Linux fallback: `paplay`/`aplay` (configurable).
- Or skip playback and leave WAV in `generated-audio/` with a short console notification.

ASCII synthesis/playback flow:

```
[SpeakTask] -> [TTS Orchestrator] -> birdler.py -> [WAV path]
                    |                              |
                    +--------> [Log]               v
                                     [Playback driver: ffplay/afplay]
```


## Process Lifetime & Concurrency

- Single process with `asyncio` loop.
- One adapter task per source (enabled via config): `claude_task`, `codex_task`.
- One consumer task: `speak_task`.
- Shared `asyncio.Queue` (bounded, e.g., size 10) to avoid unbounded memory.
- Cooperative cancellation and graceful shutdown on SIGINT/SIGTERM.

ASCII runtime:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ asyncio loop                                                                 │
│  ┌───────────────┐   put()      ┌───────────────┐   get()        ┌────────┐ │
│  │ Claude task   │ ───────────▶ │               │ ─────────────▶ │ Speak  │ │
│  ├───────────────┤              │   Queue       │                │ task   │ │
│  │ Codex task    │ ───────────▶ │   (bounded)   │                └────────┘ │
│  └───────────────┘              └───────────────┘                             │
└──────────────────────────────────────────────────────────────────────────────┘
```


## Configuration

Environment variables (via `.env`):
- `AI_SPEAK_SOURCES=claude,codex`           # which adapters to enable
- `CLAUDE_STORE_DIR=~/.claude`              # override Claude data root
- `CODEX_LOG_PATH=~/.codex/sessions`        # preferred NDJSON dir for Codex
- `BIRDLER_VOICE=<name>`                    # managed voice to use
- `BIRDLER_VOICE_DIR=voices`                # root for voice data
- `BIRDLER_OUT_DIR=generated-audio`         # where WAVs are written
- `PLAYBACK=ffplay|afplay|aplay|none`       # playback backend
- `SPEAK_MIN_INTERVAL_MS=1500`              # cooldown per session
- `SPEAK_MAX_WORDS=20`                      # hard cap per utterance
- `SPEAK_REGEX_ALLOW=...`                   # optional custom regex (CSV)

CLI (optional separate runner):
- `python tools/ai_speak_bridge.py --sources claude codex --playback ffplay --voice neo`


## Implementation Plan (Phased)

Phase 0 – Spike (1–2 days):
- Implement a standalone `tools/ai_speak_bridge.py` with only Claude adapter (polling) + ffplay.
- Heuristic intent filter, simple de-dup by last line hash.
- Manual testing against a known `~/.claude/.../conversation.json` file.

Phase 1 – Solidify Adapters (2–3 days):
- Add `watchdog` watcher with robust tail semantics (handle rotations/truncations).
- Normalize to `Event` schema; add persistent `state.json` with session offsets.
- Codex cooperative logging path; document env var integration.

Phase 2 – Orchestrator & Playback (1–2 days):
- Async queue, single consumer; Birdler subprocess wrapper with timeouts and backpressure.
- Playback driver with backends and detection of available tool (`ffplay`/`afplay`).

Phase 3 – Quality & Controls (2 days):
- Cooldowns, de-dupe window, configurable regex, max words, truncation at sentence boundary.
- Logs with structured fields; dry-run mode.

Phase 4 – Tests & Docs (1–2 days):
- Unit tests for: Claude/Codex parsers, intent filter, de-dupe, orchestrator sequencing.
- Integration test: synthetic NDJSON with appended events, verify Birdler invocation is called (mock `subprocess.run`).
- README updates for the tool and configuration.


## Key Modules & Sketches

Interfaces (Python sketches):

```python
# tools/ai_speak_bridge.py (entry)
class ConversationSource(Protocol):
    async def run(self, queue: asyncio.Queue[Event]) -> None: ...

@dataclass
class Event:
    source: str
    session_id: str
    message_id: str
    ts: float
    role: str
    text: str
    is_stream_final: bool
    meta: dict[str, Any]

def select_speakable(event: Event, *, max_words=20) -> SpeakTask | None: ...

async def tts_worker(queue: asyncio.Queue[SpeakTask], cfg: Cfg):
    while True:
        task = await queue.get()
        wav = run_birdler(task.text, cfg)  # subprocess
        play(wav, cfg)
```

Claude adapter sketch (polling or watchdog):

```python
class ClaudeAdapter(ConversationSource):
    def __init__(self, store_dir: Path, state: State): ...
    async def run(self, queue):
        while True:
            path = self._discover_latest()
            for ev in self._tail_new(path):
                if ev.role == "assistant" and ev.is_stream_final:
                    await queue.put(ev)
            await asyncio.sleep(self.poll_interval)
```

Birdler invocation:

```python
def run_birdler(text: str, cfg: Cfg) -> Path:
    out_dir = cfg.out_dir
    cmd = [
        sys.executable, "birdler.py", "--voice", cfg.voice,
        "--text", text, "--output-dir", str(out_dir),
    ]
    if cfg.atempo: cmd += ["--atempo", str(cfg.atempo)]
    subprocess.run(cmd, check=True)
    return find_latest_wav(out_dir)
```


## Error Handling & Resilience

- Missing Claude dir: warn once, retry with backoff; optionally run with only Codex source.
- File rotation/truncation: keep inode + offset; on truncation, reset offset to 0.
- Birdler failures: retry up to N times with small backoff; if repeated, drop task and log.
- Playback failures: skip playback and still keep WAV; include path in logs.


## Security & Privacy

- Only read local files under configured roots; do not transmit text externally (beyond TTS model running locally).
- Avoid logging full message content by default; log hashes or first 10 words.
- Clean temporary audio if `PLAYBACK=none` and `KEEP_WAV=false` are set.
- Honor `.env` for user-specific locations; do not commit secrets.


## Performance Targets

- Detection latency: < 300ms (watchdog) or < 1s (polling).
- Synthesis time: dominated by Birdler; bridge adds ~5–20ms overhead.
- Memory: < 50MB RSS typical.


## Testing Strategy (Pytest)

- Mock filesystem: use `tmp_path` to create fake Claude/Codex stores; append NDJSON lines; assert enqueues.
- Mock `subprocess.run` for Birdler and playback; assert correct args and sequencing.
- Parametrize speakable heuristics: question vs. non-question, caps, truncation.
- State persistence: simulate restart with a saved `state.json`; ensure no duplicates.


## Developer Workflow

Local run (example):

```
export AI_SPEAK_SOURCES=claude
export CLAUDE_STORE_DIR="$HOME/.claude"
export BIRDLER_VOICE=tuba
export PLAYBACK=ffplay
poetry run python tools/ai_speak_bridge.py
```

Troubleshooting:
- If you see duplicates, clear `~/.cache/ai-speak-bridge/state.json`.
- If no playback, verify `ffplay` or `afplay` is available; set `PLAYBACK=none` to disable.
- If Birdler is slow on first run, use `--warmup` in the orchestrator configuration to prime caches.


## Rollout & Future Enhancements

- Phase rollout behind `AI_SPEAK_SOURCES` feature flag; default off.
- Add a tiny Codex CLI hook to emit NDJSON (one-time addition, guarded by env var) for best reliability.
- Optional on-device VAD to gate speaking if mic is hot (avoid feedback).
- Optional summarizer: extract key interrogative even if message is long.
- Windows support: add `powershell -c (New-Object Media.SoundPlayer ...)` backend or `ffplay` on WSL.


## Complete Workflow Diagrams (Claude and Codex)

Claude Code end-to-end:

```
~/.claude/.../conversation.json -> [Claude Adapter]
     | (watch/truncate-safe tail, parse JSON/NDJSON)
     v
 [Event (assistant, final)] ---> [Intent Filter] -> [SpeakTask]
                                      |
                                      v
                               [Queue (bounded)]
                                      |
                                      v
                               [TTS Orchestrator]
                                      |
                                      v
        birdler.py --voice <v> --text "..." --output-dir <dir>
                                      |
                                      v
                             [WAV path] -> [Playback]
```

Codex CLI end-to-end (cooperative logging):

```
Codex CLI (env: CODEX_LOG_PATH=...) -> <events.ndjson>
                    |
                    v
             [Codex Adapter]
                    |
               normalize
                    v
                 [Event] ---> [Intent Filter] -> [SpeakTask] -> [Queue] -> [TTS] -> [Playback]
```

Codex CLI end-to-end (wrapper mode):

```
pty wrapper (tee stdout) -> <events.ndjson>
               |
               v
        [Line parser -> role: assistant] -> [Event] -> ... (same as above)
```


## Summary

This bridge cleanly decouples ingestion (Claude/Codex) from synthesis/playback via a small, testable async pipeline. It leverages Birdler’s existing CLI with minimal coupling, supports incremental rollout, and is robust to source variability by normalizing all inputs to a common event schema and applying conservative, configurable heuristics to decide what gets spoken.

