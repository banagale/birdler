#!/usr/bin/env bash
set -euo pipefail

# Generate polished samples (denoise + auto-editor + normalize) for every managed voice.
# Usage: ./run_all_voices_polish.sh

# Inherit all env controls from run_all_voices.sh, but set polish defaults:
TEXT_FILE=${TEXT_FILE:-text-scripts/style-charles-bukowski.txt}
OUT_DIR=${OUT_DIR:-generated-audio}
CFG=${CFG:-0.3}
EXAG=${EXAG:-0.8}
TEMP=${TEMP:-0.7}
MIN_CHARS=${MIN_CHARS:-30}
MAX_CHARS=${MAX_CHARS:-600}
HARD_MAX=${HARD_MAX:-800}
WORKERS=${WORKERS:-4}
N_CANDS=${N_CANDS:-2}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-1}
SEED=${SEED:-123}
RUN_INDEX=${RUN_INDEX:-4}
ATEMPO=${ATEMPO:-0.9}
DENOISE=${DENOISE:-1}
AUTO_EDITOR=${AUTO_EDITOR:-1}
NORMALIZE=${NORMALIZE:-ebu}
WARMUP=${WARMUP:-1}
DRY_RUN=${DRY_RUN:-}

export TEXT_FILE OUT_DIR CFG EXAG TEMP MIN_CHARS MAX_CHARS HARD_MAX WORKERS N_CANDS MAX_ATTEMPTS \
       SEED RUN_INDEX ATEMPO DENOISE AUTO_EDITOR NORMALIZE WARMUP DRY_RUN

exec "$(dirname "$0")/run_all_voices.sh"
