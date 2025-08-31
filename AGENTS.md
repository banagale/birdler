# Repository Guidelines

## Project Structure & Module Organization
- `birdler.py`: Main CLI for TTS generation (voice cloning, chunking, IO).
- `tests/`: Pytest suite for args, chunking, and IO behaviors.
- `text-scripts/`: Example input texts for synthesis.
- `audio-samples/`: Example/placeholder voice prompts; keep files small and clean.
- `generated-audio/`: Output WAVs (git-ignored placeholder present).
- `README.md`: Usage, flags, and installation notes.

## Build, Test, and Development Commands
- Install (dev): `poetry install --with dev`
- Lint: `poetry run ruff check .` (autofix: add `--fix`)
- Format: `poetry run ruff format .`
- Test: `poetry run pytest -q`
- Run CLI: `python birdler.py --audio-sample audio-samples/bigbird/bigbird_youtube_clean.wav --output-dir generated-audio`
  - Optional: `poetry run python birdler.py ...`

## Coding Style & Naming Conventions
- Python 3.12; 4-space indent; LF endings.
- Ruff enforces: 120-col line length, double quotes, sorted imports (E,F,I).
- Naming: snake_case for functions/vars, UPPER_SNAKE for constants.
- Prefer `pathlib.Path` for paths and pure functions for helpers.

## Testing Guidelines
- Framework: Pytest (configured via `pyproject.toml`).
- Location/Names: place tests in `tests/` as `test_*.py`.
- Scope: cover CLI arg parsing, text chunking, and side-effect boundaries.
- Isolation: mock external tools (e.g., `yt-dlp`, `subprocess.run`, `which`) and filesystem with `tmp_path`.
- Run quickly; heavy deps (torch/torchaudio) are imported inside `main()`—unit tests should not require them.

## Commit & Pull Request Guidelines
- Commits: short, imperative subject lines (no Conventional Commits required).
  - Examples from history: “Improve output file handling”, “Pick chunk size dynamically”.
- PRs: focused scope, clear description, steps to reproduce, and test updates.
  - Link related issues; include sample commands and expected logs; update README if flags change.

## Security & Configuration Tips
- `.env` is loaded for local config; never commit secrets.
- For YouTube extraction, ensure `ffmpeg` is installed and `yt-dlp` (or `youtube-dl`) available.
- Avoid committing large audio assets; keep `generated-audio/` outputs local.
