#!/usr/bin/env bash
set -euo pipefail

# Birdler run script: generates the Style poem with the Ripley voice (slowed playback)
# Usage: ./run_ripley_style_slow.sh

VOICE="ripley"
TEXT_FILE="text-scripts/style-charles-bukowski.txt"
OUT_DIR="generated-audio"

# Optional: path to a clean reference wav for first-time bootstrap
REF_WAV="audio-samples/ripley/aliens-ripley-scene-clip-clean.wav"

echo "[info] Using voice=$VOICE text=$TEXT_FILE out=$OUT_DIR"

# Ensure embedding exists; if missing and REF_WAV exists, bootstrap and build
if [[ ! -f "voices/$VOICE/embedding/cond.pt" ]]; then
  if [[ -f "$REF_WAV" ]]; then
    echo "[info] Bootstrapping voice '$VOICE' from $REF_WAV and building embedding"
    poetry run python birdler.py \
      --voice "$VOICE" \
      --audio-sample "$REF_WAV" \
      --bootstrap-only --build-embedding --force-voice-ref
  else
    echo "[warn] voices/$VOICE/embedding/cond.pt not found and REF_WAV missing.\n" \
         "      Please provide a reference WAV and run bootstrap: \n" \
         "      poetry run python birdler.py --voice $VOICE --audio-sample /abs/path/to/ref.wav --bootstrap-only --build-embedding"
  fi
fi

echo "[run] Option C — expressive, deterministic, larger chunks, parallel, slowed (atempo=0.9)"
poetry run python birdler.py \
  --voice "$VOICE" \
  --text-file "$TEXT_FILE" \
  --output-dir "$OUT_DIR" \
  --cfg-weight 0.3 \
  --exaggeration 0.8 \
  --temperature 0.7 \
  --max-chars 600 \
  --hard-max-chars 800 \
  --min-chars 30 \
  --warmup \
  --seed 123 \
  --run-index 3 \
  --workers 4 \
  --n-candidates 2 \
  --max-attempts 1 \
  --atempo 0.9

echo "[done] Run completed. Check $OUT_DIR and adjacent .settings.json/.csv files."

