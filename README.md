# Birdler

# Birdler is a simple command-line voice cloning tool built on top of ChatterboxTTS.

It lets you generate expressive, characterful speech by providing a short reference
audio sample (the “voice prompt”), customizable generation parameters, and optional
text input (from a file or direct string).

## Features

- **Voice cloning** via a clean reference sample (your own data, character voices, etc.)
- Configurable **guidance scale**, **exaggeration**, **temperature**, and **repetition penalty**
- Automatic device selection (CPU, CUDA, or Apple MPS)
- Splits longer text into chunks, synthesizes each, then concatenates the final waveform
- Custom text input via `--text-file` or `--text` to read a script from a file or
  provide a direct text string

## Installation

Make sure you have Python 3.12+ and install dependencies:

```bash
pip install torch torchaudio chatterbox-tts
```

For YouTube audio extraction (optional feature), install `yt-dlp` (and ensure `ffmpeg` is on your PATH):

```bash
pip install yt-dlp    # or youtube-dl
```

## Usage (managed voices)

First run bootstraps a voice directory and caches a speaker embedding:

```bash
python birdler.py \
  --voice ripley \
  --audio-sample audio-samples/ripley/aliens-ripley-scene-clip-clean.wav \
  --text-file text-scripts/style-charles-bukowski.txt \
  --output-dir generated-audio
```

Subsequent runs can omit the sample and reuse the cached embedding:

```bash
python birdler.py --voice ripley --text "Another line" --output-dir generated-audio
```

### Synthesize from a text script file

By default, `text-scripts/style-charles-bukowski.txt` is provided as a placeholder for Bukowski's poem
Style (replace it with the full text if desired).

```bash
python birdler.py \
  --text-file text-scripts/style-charles-bukowski.txt \
  --output-dir generated-audio
```

### Synthesize from a direct text string

```bash
python birdler.py \
  --text "Your custom text here" \
  --output-dir generated-audio
```

### Extract audio from YouTube

```bash
python birdler.py \
  --youtube-url 'https://www.youtube.com/watch?v=rkBhLjwuq20' \
  --output-dir audio-samples
```

### Arguments

- `--voice`: Managed voice name stored under `voices/<name>` (required for TTS).
- `--voice-dir`: Root directory for managed voices (default: `voices`).
- `--audio-sample`: Reference WAV used only to bootstrap a new `--voice` on first run.
- `--youtube-url`: YouTube URL to extract audio from (requires yt-dlp or youtube-dl); if set, extracts audio and exits.
- `--output-dir`: Directory where the generated WAV or extracted audio will be saved.
- `--device`: Force a device (`cpu`, `cuda`, or `mps`); auto-detected if omitted.
- `--cfg-weight`: Guidance scale (higher = more faithful to prompt).
- `--exaggeration`: Expressiveness factor (higher = more dramatic).
- `--temperature`: Sampling temperature (lower = more deterministic).
- `--repetition-penalty`: Penalty to discourage stuttering.

- `--text-file`: Path to a text file to synthesize (mutually exclusive with --text).
- `--text`: Text string to synthesize directly (mutually exclusive with --text-file).

Generated audio will be written as `generated-audio/bigbird_exhausting_week.wav` by default.

## How it works

1. **Load** a pretrained ChatterboxTTS model onto your chosen device.
2. **Read** your reference sample and the text chunks (default or user-supplied script).
3. **Generate** each chunk in turn using the provided TTS parameters.
4. **Concatenate** the audio chunks and save the final waveform.

This script is a quick demo harness—you can adapt the chunks or hook into the
ChatterboxTTS API directly for more advanced workflows.

## License

MIT License © Rob Banagale
