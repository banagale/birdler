# Birdler Voice‑Directory & Embedding Caching Refactor Roadmap

This document outlines a step‑by‑step plan to refactor Birdler so that it automatically
manages speaker directories containing raw samples and pre‑computed conditioning
embeddings for repeatable, efficient TTS runs.

---

## 1. Goals & Requirements

1. **Voice‑centric API** Instead of requiring `--audio-sample /path/to/file.wav`, users supply
   `--voice <name>` to register or select an existing speaker directory.
2. **Automatic bootstrapping** On first run for a new voice:
   - Create `voices/<voice>/samples/` and `voices/<voice>/embedding/`.
   - Copy (or symlink) the provided WAV into `samples/reference.wav`.
   - Compute and save the speaker embedding via `tts.get_audio_conditioning()`.
3. **Cached embeddings** On subsequent runs, Birdler loads `voices/<voice>/embedding/cond.pt`
   and skips re‑encoding the raw WAV.
4. **Backward compatibility** Still accept `--audio-sample` in legacy mode, but warn and
   suggest `--voice`.
5. **Minimal disruption** Keep chunking and crossfade logic intact; isolate voice management.

## 2. CLI Changes

| Old Flag                         | New Flag                    | Behavior                                                          |
|----------------------------------|-----------------------------|-------------------------------------------------------------------|
| `--audio-sample <wav>`           | supported (legacy)          | Still works; warns. May be combined with `--voice` to bootstrap.  |
| —                                | `--voice <name>`            | Primary way to select/register a speaker directory in `voices/`.  |
| —                                | `--voice-dir <path>`        | Optional override for the `voices/` root directory.               |

**Example usage:**
```bash
# First run for voice “ripley” with a downloaded WAV:
python birdler.py \
  --voice ripley \
  --audio-sample /downloads/ripley_ref.wav \
  --text-file text-scripts/story.txt \
  --output-dir generated-audio

# Subsequent runs reuse the cached embedding (no WAV needed):
python birdler.py \
  --voice ripley \
  --text-file text-scripts/another_story.txt \
  --output-dir generated-audio
```

## 3. Directory Layout

```
project-root/
├── birdler.py
├── voices/                    # Root for all registered speaker data
│   ├── ripley/
│   │   ├── samples/           # Raw WAV(s) for “ripley”
│   │   │   └── reference.wav
│   │   └── embedding/         # Precomputed conditioning tensor
│   │       └── cond.pt
│   └── [other voices…]/
├── generated-audio/
├── text-scripts/
└── …
```

## 4. Implementation Plan

### 4.1. Add `--voice` CLI option
- Extend `parse_args()` to support `--voice` alongside legacy `--audio-sample`.
- Allow combining `--voice` + `--audio-sample` on first run to bootstrap.
- Document `--voice-dir` override.

### 4.2. Centralize voice directory logic
- Implement `prepare_voice(args)` to:
  1. Resolve `voices_dir` (default `voices/`).
  2. Validate args; ensure `voice` or `audio-sample` is provided.
  3. Create `voices/<voice>/samples/` and `voices/<voice>/embedding/`.

### 4.3. Copy/raw WAV into samples/
- In `prepare_voice()`, if a new voice or empty samples dir, copy or symlink
  `args.audio_sample` → `voices/<voice>/samples/reference.wav`.

### 4.4. Compute or load speaker embedding
- Add `get_or_build_embedding(tts, voice_root)`:
  ```python
  emb_path = voice_root / 'embedding' / 'cond.pt'
  if emb_path.exists():
      return torch.load(emb_path)
  cond = tts.get_audio_conditioning(str(voice_root/'samples'/'reference.wav'))
  torch.save(cond, emb_path)
  return cond
  ```

### 4.5. Wire into `main()`
- Replace legacy `audio_prompt_path` logic:
  1. Call `prepare_voice(args)` to bootstrap dirs.
  2. Call `cond = get_or_build_embedding(tts, voice_root)`.
  3. In the chunk loop, pass `audio_prompt_cond=cond` to `tts.generate()`.

### 4.6. Deprecate `audio_prompt_path`
- Remove or warn about the old path-based prompt API; prefer `audio_prompt_cond`.

### 4.7. Update documentation
- Rewrite README usage examples for `--voice`.
- Document `voices/` structure and caching behavior.

### 4.8. Testing & Verification
- Unit tests for `prepare_voice()` and `get_or_build_embedding()`.
- Integration smoke‑test: first run vs. second run logs and file presence.
- Pre‑commit checks and a manual end‑to‑end run to confirm correctness.

## 5. Roadmap & Timeline

| Phase                        | Description                                             | Effort  |
|------------------------------|---------------------------------------------------------|---------|
| Phase 1: CLI & dirs          | Add `--voice`, build `prepare_voice()` scaffolding      | 1 day   |
| Phase 2: Embedding caching   | Implement `get_or_build_embedding()`, swap generate API | 1–2 days|
| Phase 3: Backcompat          | Deprecation warnings, legacy flag support               | ½ day   |
| Phase 4: Docs & tests        | Update README, add unit/integration tests               | 1 day   |
| Phase 5: Review & polish     | Cleanups, pre-commit, final review                      | ½ day   |

## 6. Summary of Key Interfaces

| Function                  | Responsibility                                               |
|---------------------------|--------------------------------------------------------------|
| `prepare_voice(args)`     | Validate args & create `voices/<voice>` subdirectories       |
| `get_or_build_embedding`  | Load or compute & save the speaker‑conditioning embedding    |
| Modified `main()`         | Wire voice preparation & cached embedding into generation    |

With this plan, Birdler will seamlessly bootstrap and cache voice embeddings,
minimizing startup overhead and providing a polished, maintainable workflow.
