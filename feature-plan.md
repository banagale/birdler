# Birdler ← Chatterbox‑TTS‑Extended: Feature Plan (Phased, Atomic, Tested)

This plan implements todos‑chatterboxtts‑extended.md in incremental phases. Each phase ships in small, atomic commits with new/updated tests; run `poetry run pytest -q` between phases. External tools (yt‑dlp, ffmpeg, RNNoise, auto‑editor) are stubbed in tests.

## Phase 1 — Chunking Upgrades
- Add chunking module: `split_sentences`, reuse greedy pack, add `split_long_sentence`, `enforce_min_chunk_length`, `get_chunks`.
- Wire `get_text_chunks` to new `get_chunks` (CLI: add `--min-chars`, keep `--max-chars`, `--hard-max-chars`).
- Tests: min‑len merge; over‑max long‑sentence split (no piece > hard_max).
- Commits: feat(chunking): min‑length merge + recursive splitting; refactor(cli): route through chunking.get_chunks.

## Phase 2 — Deterministic Seeding + Guardrails
- Add `seeding.py`: `derive_seed(base, chunk, cand, attempt)`, `generate_with_seed(...)` (per‑call generator or forked RNG); add `--seed`, `--deterministic`.
- Device guard: `set_determinism(device)` (disable TF32; deterministic algos when possible).
- Tests: seed grid uniqueness; repeatability with same seed.
- Commits: feat(seeding): deterministic seeds; feat(device): determinism helper + flag.

## Phase 3 — Parallel Generation Scaffold
- `gen_parallel.py`: thread pool, per‑task seeds, collect by chunk index to preserve order; CLI: `--workers`, `--n-candidates`, `--max-attempts`.
- Tests: ordering preserved under delays; distinct seeds across cand/attempts.
- Commit: feat(gen): parallel chunk generation with deterministic seeds.

## Phase 4 — Validation + Selection + Retries
- `validate.py`: load whisper (openai/faster), `transcribe_and_score`; `select.py`: `pick_candidate`, `assemble_waveforms`; `retry.py` loop.
- CLI: `--validate`, `--whisper-backend`, `--whisper-model`, `--validate-threshold`, `--prefer-longest-on-fail`.
- Tests: selection policy; retry converges on pass with derived seeds (whisper stubbed).
- Commits: feat(validate): validation + selection; feat(retry): retry failed chunks.

## Phase 5 — Post‑processing Chain (Optional)
- `denoise.py` (RNNoise CLI or pyrnnoise), `post_silence.py` (auto‑editor), `normalize.py` (FFmpeg EBU/peak) with clear missing‑tool guidance; CLI: `--denoise`, `--auto-editor`, `--normalize`, `--keep-original`.
- Tests: command construction and replace‑in‑place behavior (subprocess stubbed).
- Commits: feat(post): denoise + silence trim + normalize; chore(post): helpful errors.

## Phase 6 — Settings Artifacts + Naming
- `settings_io.py`: write `.settings.json` + `.settings.csv` (no raw text; include seeds, device, flags, output list).
- `naming.py`: `build_outname(voice, text, gen_idx, seed, ts)`; CLI: `--run-index`.
- Tests: artifacts present and scrubbed; filename encodes metadata.
- Commits: feat(io): settings artifacts; feat(naming): stable filenames.

## Phase 7 — Progress/Logging Polish
- `progress.py`: `[PROGRESS] N/M` hooks; ensure final assembly respects chunk order.
- Tests: stdout captures progress markers.
- Commit: feat(log): progress hooks and ordered assembly.

## Phase 8 — Optional VC Subcommand
- `vc_cmd.py`: simple VC wrapper; CLI: `--vc-input`, `--vc-target-ref`, `--vc-watermark`, `--vc-pitch-shift` (mutually exclusive with TTS).
- Tests: argument routing + save call (stub).
- Commit: feat(vc): optional VC subcommand.

## Cross‑Cutting
- README: new flags/flows/presets/troubleshooting; tool install hints.
- Error policy: deterministic, actionable messages; compat mode remains explicit.
- Always keep commits focused; add tests first (red), implement (green), refactor if needed.
