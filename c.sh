#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to enable Codex TUI session recording with a stable path.
# Also supports resuming the most recent Codex session with -c / --continue.
#
# Usage examples:
#   ./c.sh                         # launch Codex TUI (records to ~/.codex/log/codex-session.jsonl)
#   ./c.sh -c                      # launch Codex TUI and resume latest session
#   ./c.sh codex exec "…"            # run headless exec with the same env

continue_last=false
no_pause=false
pass_args=()
while (($#)); do
  case "$1" in
    -c|--continue)
      continue_last=true
      shift
      ;;
    -y|--no-pause)
      no_pause=true
      shift
      ;;
    --)
      shift; pass_args+=("$@"); break ;;
    *)
      pass_args+=("$1"); shift ;;
  esac
done

# Default session log path (can be overridden by env before calling this script)
: "${CODEX_TUI_RECORD_SESSION:=1}"
DEFAULT_LOG_PATH="$HOME/.codex/log/codex-session.jsonl"
if [[ -z "${CODEX_TUI_SESSION_LOG_PATH:-}" ]]; then
  export CODEX_TUI_SESSION_LOG_PATH="$DEFAULT_LOG_PATH"
fi
export CODEX_TUI_RECORD_SESSION

mkdir -p "$(dirname "$CODEX_TUI_SESSION_LOG_PATH")"
echo "[codex] Recording session events to: $CODEX_TUI_SESSION_LOG_PATH" >&2

# If no args provided, default to launching the TUI
if [[ ${#pass_args[@]} -eq 0 ]]; then
  pass_args=(codex)
fi

# If resume requested, locate latest rollout session and inject config override.
if $continue_last; then
  sessions_dir="$HOME/.codex/sessions"
  if compgen -G "$sessions_dir/rollout-*.jsonl" > /dev/null; then
    latest_session=$(ls -t "$sessions_dir"/rollout-*.jsonl | head -n1)
    echo "[codex] Resuming session: $latest_session" >&2
    # Insert config override: -c experimental_resume="<path>"
    pass_args=("${pass_args[@]}" -c "experimental_resume=$latest_session")
  else
    echo "[codex] No previous sessions found under $sessions_dir; starting fresh." >&2
  fi
fi

# Pause to let user read the info line(s) before TUI clears the screen,
# unless disabled via -y/--no-pause or when stdin is not a TTY.
if ! $no_pause && [ -t 0 ]; then
  read -r -p "[codex] Press Enter to launch…" _
fi

exec "${pass_args[@]}"
