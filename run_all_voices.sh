#!/usr/bin/env bash
set -euo pipefail

# Generate a sample for every managed voice under voices/<name>/.
#
# Usage:
#   ./run_all_voices.sh
#
# Configuration (env vars override defaults):
#   TEXT_FILE   Path to a text file to synthesize (preferred)
#   TEXT        Fallback direct text string if TEXT_FILE is empty
#   OUT_DIR     Output directory (default: generated-audio)
#   CFG         Guidance scale (default: 0.5)
#   EXAG        Exaggeration (default: 0.5)
#   TEMP        Temperature (default: 0.8)
#   MIN_CHARS   Minimum chunk length (default: 20)
#   MAX_CHARS   Soft chunk length (default: 280)
#   HARD_MAX    Hard chunk cap (default: 360)
#   WORKERS     Parallel workers (default: 1)
#   N_CANDS     Candidates per chunk (default: 1)
#   MAX_ATTEMPTS Max validation retries (default: 1)
#   SEED        Base seed (optional)
#   RUN_INDEX   Run index for stable naming (default: 0)
#   ATEMPO      Playback speed multiplier (0.5–2.0, optional)
#   DENOISE     If set to 1, apply RNNoise
#   AUTO_EDITOR If set to 1, run auto-editor
#   NORMALIZE   If set, one of: ebu|peak
#   WARMUP      If set to 1, run warmup generate
#   DRY_RUN     If set to 1, only print commands

TEXT_FILE=${TEXT_FILE:-}
TEXT=${TEXT:-}
OUT_DIR=${OUT_DIR:-generated-audio}
CFG=${CFG:-0.5}
EXAG=${EXAG:-0.5}
TEMP=${TEMP:-0.8}
MIN_CHARS=${MIN_CHARS:-20}
MAX_CHARS=${MAX_CHARS:-280}
HARD_MAX=${HARD_MAX:-360}
WORKERS=${WORKERS:-1}
N_CANDS=${N_CANDS:-1}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-1}
SEED=${SEED:-}
RUN_INDEX=${RUN_INDEX:-0}
ATEMPO=${ATEMPO:-}
DENOISE=${DENOISE:-}
AUTO_EDITOR=${AUTO_EDITOR:-}
NORMALIZE=${NORMALIZE:-}
WARMUP=${WARMUP:-}
DRY_RUN=${DRY_RUN:-}

if [[ -z "$TEXT_FILE" && -z "$TEXT" ]]; then
  echo "[warn] Neither TEXT_FILE nor TEXT set. Using a short default string."
  TEXT="Hello from Birdler"
fi

echo "[info] Output dir: $OUT_DIR"
mkdir -p "$OUT_DIR"

shopt -s nullglob
VOICES=()
# Collect voice names from managed voices and audio-samples
# Managed voices
for d in voices/*; do
  [[ -d "$d" ]] || continue
  name=$(basename "$d")
  VOICES+=("$name")
done
# Audio-samples candidates
for d in audio-samples/*; do
  [[ -d "$d" ]] || continue
  name=$(basename "$d")
  # Add if not already present
  found=0
  for v in "${VOICES[@]}"; do
    if [[ "$v" == "$name" ]]; then found=1; break; fi
  done
  if [[ $found -eq 0 ]]; then VOICES+=("$name"); fi
done
# Filter to those that either already have a managed reference or have a sample WAV to bootstrap
FILTERED=()
for v in "${VOICES[@]}"; do
  if [[ -f "voices/$v/samples/reference.wav" ]]; then
    FILTERED+=("$v")
  else
    # Check for bootstrappable sample
    wv=(audio-samples/"$v"/*.wav)
    if [[ -f "${wv[0]:-}" ]]; then
      FILTERED+=("$v")
    fi
  fi

done
VOICES=("${FILTERED[@]}")


if [[ ${#VOICES[@]} -eq 0 ]]; then
  echo "[warn] No voices found under voices/<name>/samples/reference.wav"
  exit 0
fi

echo "[info] Voices: ${VOICES[*]}"

for v in "${VOICES[@]}"; do
  echo "[voice] $v"
  # Ensure embedding exists; build if missing (prefer a *clean*.wav under audio-samples)
  if [[ ! -f "voices/$v/embedding/cond.pt" ]]; then
    sample=""
    clean=(audio-samples/"$v"/*clean*.wav)
    if [[ -f "${clean[0]:-}" ]]; then
      sample="${clean[0]}"
    else
      any=(audio-samples/"$v"/*.wav)
      if [[ -f "${any[0]:-}" ]]; then
        sample="${any[0]}"
      fi
    fi
    if [[ -n "$sample" ]]; then
      cmd=(poetry run python birdler.py --voice "$v" --audio-sample "$sample" --bootstrap-only --build-embedding --force-voice-ref)
      echo "[run] ${cmd[*]}"
      if [[ "$DRY_RUN" != "1" ]]; then
        "${cmd[@]}"
      fi
    else
      echo "[warn] No sample WAV found under audio-samples/$v; skipping bootstrap"
    fi
  fi

  cmd=(poetry run python birdler.py --voice "$v" --output-dir "$OUT_DIR" \
       --cfg-weight "$CFG" --exaggeration "$EXAG" --temperature "$TEMP" \
       --min-chars "$MIN_CHARS" --max-chars "$MAX_CHARS" --hard-max-chars "$HARD_MAX" \
       --workers "$WORKERS" --n-candidates "$N_CANDS" --max-attempts "$MAX_ATTEMPTS" \
       --run-index "$RUN_INDEX")

  if [[ -n "$SEED" ]]; then cmd+=(--seed "$SEED"); fi
  if [[ -n "$ATEMPO" ]]; then cmd+=(--atempo "$ATEMPO"); fi
  if [[ "$DENOISE" == "1" ]]; then cmd+=(--denoise); fi
  if [[ "$AUTO_EDITOR" == "1" ]]; then cmd+=(--auto-editor); fi
  if [[ -n "$NORMALIZE" ]]; then cmd+=(--normalize "$NORMALIZE"); fi
  if [[ "$WARMUP" == "1" ]]; then cmd+=(--warmup); fi

  if [[ -n "$TEXT_FILE" ]]; then
    cmd+=(--text-file "$TEXT_FILE")
  else
    cmd+=(--text "$TEXT")
  fi

  echo "[run] ${cmd[*]}"
  if [[ "$DRY_RUN" != "1" ]]; then
    "${cmd[@]}"
  fi

done

echo "[done] Completed for voices: ${VOICES[*]}"
