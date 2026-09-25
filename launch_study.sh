#!/usr/bin/env bash
# Run on the GPU server. The SSH client can disconnect immediately afterward.
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ -f ../env.sh ]]; then source ../env.sh; fi
if (( $# < 2 || $# > 3 )); then
  echo 'Usage: bash launch_study.sh PLAN.json GPU-UUID [GPU-UUID]' >&2
  exit 2
fi
plan=$1
shift
[[ -f "$plan" ]] || { echo "Missing plan: $plan" >&2; exit 2; }
[[ -x .venv/bin/python ]] || { echo 'Expected .venv/bin/python; install requirements first.' >&2; exit 2; }
command -v tmux >/dev/null
if tmux has-session -t '=optimizer-training' 2>/dev/null; then
  echo 'optimizer-training already exists. Inspect it with: tmux attach -t optimizer-training' >&2
  exit 2
fi
# Check now for a readable plan and idle GPUs. The queue checks again before each trial.
.venv/bin/python - "$plan" "$@" <<'PY'
import sys
from study import check_plan, ensure_free_gpu, load_json
check_plan(load_json(sys.argv[1]))
for gpu in sys.argv[2:]:
    ensure_free_gpu(gpu)
PY
mkdir -p results
rm -f results/queue.exit
printf -v job '%q ' .venv/bin/python -u study.py run --plan "$plan" --gpus "$@" --hours "${STUDY_HOURS:-8}"
job="set -o pipefail; $job 2>&1 | tee -i -a results/queue.log; code=\${PIPESTATUS[0]}; printf '%s\\n' \"\$code\" > results/queue.exit; exit \"\$code\""
tmux new-session -d -s optimizer-training -c "$PWD" bash -c "$job"
tmux set-option -t optimizer-training prefix C-a
tmux set-option -t optimizer-training mouse on
printf 'Started on this server. Attach: tmux attach -t optimizer-training\nDetach: Ctrl+A, then d. Stop and save: Ctrl+C inside the session.\nQueue log: %s/results/queue.log\n' "$PWD"
