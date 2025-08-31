# Birdler

Command-line voice cloning built on ChatterboxTTS, with managed voices, reproducible runs, parallel generation, and optional post‑processing.

## Quick Start

Install (Python 3.12+ recommended):
- Using Poetry: `poetry install --with dev`
- Or pip: `pip install chatterbox-tts torch torchaudio yt-dlp` (plus `ffmpeg` on PATH for media ops)

Add a new voice from a clean WAV (builds and caches embedding):
```bash
birdler --voice tuba   --audio-sample "/abs/path/to/tuba.wav"   --bootstrap-only --build-embedding
```

Synthesize a script (uses cached embedding):
```bash
birdler --voice tuba   --text-file text-scripts/style-charles-bukowski.txt   --output-dir generated-audio
```

Bootstrap from YouTube, then build embedding after you edit `voices/<name>/samples/reference.wav`:
```bash
birdler --voice ripley --youtube-url 'https://youtu.be/xxx' --bootstrap-only
birdler --voice ripley --bootstrap-only --build-embedding
```

Expressive preset:
```bash
birdler --voice tuba --text "Hello" --cfg-weight 0.3 --exaggeration 0.8 --temperature 0.7
```

Deterministic run (seeded) and parallelism:
```bash
birdler --voice tuba --text "Hello" --seed 123 --run-index 1   --workers 4 --n-candidates 2 --max-attempts 1
```

Optional post-processing:
```bash
birdler --voice tuba --text "Hi" --denoise --auto-editor --normalize ebu
```

Compat mode (no embedding; path-based prompting):
```bash
birdler --voice tuba --compat-legacy-prompt --text "Hello"
```

## Key Concepts & Flags

- Managed voices: `voices/<name>/{samples/reference.wav, embedding/cond.pt}`
- Building embeddings: `--bootstrap-only --build-embedding` (first time); compat mode available.
- Chunking: sentence-aware with soft/hard caps and minimum chunk length.
- Parallel gen: `--workers`, `--n-candidates`, `--max-attempts`; deterministic seeds via `--seed`.
- Validation & retry (experimental scaffold): `--validate`, `--validate-threshold`; retries missing chunks.
- Post‑processing (optional): `--denoise` (RNNoise CLI), `--auto-editor`, `--normalize ebu|peak`.
- Determinism: `--deterministic` enables deterministic algorithms; filenames include `_gen{N}` and optional `_seed{S}`.
- Settings artifacts: writes `.settings.json` and `.settings.csv` next to the WAV (no raw text).
- Progress: prints `[PROGRESS] N/M` and per-chunk previews.

Defaults (neutral): `--cfg-weight 0.5 --exaggeration 0.5 --temperature 0.8 --repetition-penalty 1.2`

## Tips for Reference Clips
- Use a clean, single‑speaker 5–15s clip (mono PCM WAV). Avoid music/SFX/reverb.
- For YouTube, extract best audio, isolate the speaker in an editor, export mono WAV, then `--build-embedding`.

## Acknowledgements
- ChatterboxTTS by Resemble AI powers synthesis and watermarks (Perth). Outputs include imperceptible watermarks.
- This project incorporates ideas inspired by the Chatterbox‑TTS‑Extended work (parallelization, determinism, validation hooks).

## Troubleshooting
- “Embedding not available”: run bootstrap with a clean WAV (`--build-embedding`) or use `--compat-legacy-prompt`.
- Missing tools: `yt-dlp`, `ffmpeg`, `auto-editor`, `denoise` are optional; install for related features.
- Device: prefers Apple `mps` on macOS, `cuda` on NVIDIA, else `cpu`.

## License

MIT License © Rob Banagale
