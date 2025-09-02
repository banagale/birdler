user
suggest what might be best to work on next in this project

codex
I’m going to quickly scan the repo structure and key files to see what’s implemented and what’s missing
before suggesting high-impact next steps.

>_
  ✓ 🔎 rg --files -n
  ✓ 📖 README.md
  ✓ 📖 birdler.py
  ✓ 📖 birdler.py
  ✓ 📖 birdler.py
  ✓ 📖 gen_parallel.py
  ✓ 📖 postprocess.py
  ✓ 📖 chunking.py
  ✓ 📖 seeding.py
  ✓ 📖 test_birdler.py
  ✓ 📖 test_args.py
  ✓ 📖 test_chunking_minlen.py
  ✓ 📖 test_integratioN_smoke.py
  ✓ 📖 test_integration_smoke.py
  ✓ 📖 test_parallel_retry_cli.py
  ✓ 📖 test_parallel_gen.py
  ✓ 📖 test_retry_loop.py
  ✓ 📖 retry.py
  ✓ 📖 test_selection_policy.py
  ✓ 📖 selection.py
  ✓ 📖 test_progress_messages.py
  ✓ 📖 progress.py
  ✓ 📖 test_settings_io.py
  ✓ 📖 settings_io.py
  ✓ 📖 test_naming.py
  ✓ 📖 test_text_splitting.py
  ✓ 📖 test_output_dir_prompt.py
  ✓ 📖 test_youtube_bootstrap.py
  ✓ 📖 test_youtube_missing_tool.py
  ✓ 📖 test_parallel_cli_integration.py
  ✓ 📖 test_compat_legacy_prompt.py
  ✓ 📖 test_post_denoise_stub.py
  ✓ 📖 test_normalize_cmdline.py
  ✓ 📖 test_atempo_cmdline.py
  ✓ 📖 test_import_and_cli.py
  ✓ 📖 validate.py
  ✓ 📖 test_voice_workflow.py
  ✓ 📖 test_parallel_retry_cli.py
  ✓ ⌨️ awk 'NR>=1 && NR<=240{print NR":"$0}' birdler.py | sed -n '1,240p'
  ✓ ⌨️ awk 'NR>=240 && NR<=720{print NR":"$0}' birdler.py
  ✓ ⌨️ awk 'NR>=720 && NR<=1200{print NR":"$0}' birdler.py
  ✓ 📖 birdler.py
  ✓ 📖 pyproject.toml
  ✓ 🔎 validate|retry|select_best|validate_candidates_map
  ✗ 🧪 pytest -q
    sandbox error: command was killed by a signal
  ✓ 📖 feature-plan.md
  ✓ 📖 todos-chatterboxtts-extended.md

codex
Here’s a concise, impact-first roadmap based on the current repo.

High-Impact

- Validation pipeline: Implement real Whisper/Faster-Whisper in validate.py, wire --validate to run
per-chunk ASR scoring, pick with selection.pick_candidate, and retry via retry.retry_failed_chunks.
Persist candidates to temp files if needed for ASR.
- Selection policy: Finish selection.select_best to honor --validate-threshold and
--prefer-longest-on-fail, and log chosen seeds/candidates.
- Voice UX: Add --list-voices, --show-voice <name>, and a “rebuild embedding” command; improve
_print_embedding_diagnostics with actionable tips.
- Robust error paths: Tighten subprocess error handling for ffmpeg, yt-dlp, auto-editor, denoise; emit
clear guidance and skip gracefully when tools are missing.
- Reproducibility: Add a quick test ensuring identical outputs for the same --seed across runs (already
scaffolded with seeding.py).

Quality & Polish

- Post-processing sequencing: Make denoise → trim → normalize explicit; validate output loudness and
add a small regression test around the chain.
- Chunking edges: Add tests for punctuation-heavy text, very long single tokens, and --min-chars
interactions to guard regressions.
- Progress/logging: Include per-chunk seed, candidate index, and duration in [PROGRESS]; add a summary
table at the end with chosen picks.

DX/Tooling

- CI: Add GitHub Actions for ruff and pytest -q (Python 3.12). Cache pip/poetry.
- Presets: Ship a few --preset expressive|neutral|narration that set cfg-weight, exaggeration,
temperature, repetition-penalty.
- Docs: Expand README with an end-to-end “new voice” flow (YouTube → curate ref → build embedding →
synthesize), plus validation examples.

If you want, I can start by wiring the validation + selection + retry path (most leverage) and add
minimal tests to lock behavior.
