#!/usr/bin/env bash
# Runs generate.py in a tmux session, so it keeps going when SSH disconnects, and logs its output to untracked/logs/.
#     bash vast/run.sh --config runs.yaml
#     bash vast/run.sh 1000 --prompts "PROMPT A" "PROMPT B"
# Arguments go straight to generate.py (python generate.py --help lists them).
# A --dry-run runs right here instead of in tmux.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
VENV_PY="$PROJECT_DIR/.venv/bin/python"
[ -x "$VENV_PY" ] || { echo "error: no .venv yet. Run: bash vast/setup.sh" >&2; exit 1; }

for arg in "$@"; do
    if [ "$arg" = --dry-run ]; then
        exec "$VENV_PY" generate.py "$@"
    fi
done

mkdir -p untracked/logs
log="untracked/logs/$(date +%Y-%m-%d_%H-%M-%S).log"

#First free session name: gen, gen2, gen3, ...
session=gen
n=1
while tmux has-session -t "=$session" 2>/dev/null; do
    n=$((n + 1))
    session="gen$n"
done

args=""
(($#)) && args="$(printf '%q ' "$@")"
#tee -i ignores Ctrl+C, so it keeps logging while generate.py saves the current run and stops
#The tmux server may predate this shell, so pass on its Hugging Face token explicitly
token=""
[ -n "${HF_TOKEN:-}" ] && token="HF_TOKEN=$(printf %q "$HF_TOKEN") "
command="cd $(printf %q "$PROJECT_DIR") && ${token}PYTHONUNBUFFERED=1 $(printf %q "$VENV_PY") generate.py ${args}2>&1 | tee -i -a $(printf %q "$log")"
command+='; echo; echo "Finished (exit code ${PIPESTATUS[0]}). Press Enter to close this window."; read -r'
TMUX='' tmux new-session -d -s "$session" bash -c "$command"

echo "Started in tmux session '$session'. Log: $PROJECT_DIR/$log"
if [ -n "${TMUX:-}" ]; then
    echo "Watch it:  tmux switch-client -t $session    (back: tmux switch-client -l)"
else
    echo "Watch it:  tmux attach -t $session    (leave it running: Ctrl+B, then D)"
fi
echo "Or follow the log:  tail -f $PROJECT_DIR/$log"
echo "Stop it, saving the current run: attach and press Ctrl+C"
