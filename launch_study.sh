#!/usr/bin/env bash
# Run on the GPU server. The SSH client can disconnect immediately afterward.
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ -f ../env.sh ]]; then source ../env.sh; fi
retry_args=()
calibrate=false
if [[ ${1:-} == --calibrate ]]; then
  calibrate=true
  shift
fi
if [[ ${1:-} == --retry-failed ]]; then
  retry_args=(--retry-failed)
  shift
fi
if (( $# < 2 || $# > 3 )); then
  echo 'Usage: bash launch_study.sh [--retry-failed | --calibrate] PLAN-or-SPEC.json GPU-UUID [GPU-UUID]' >&2
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
.venv/bin/python - "$plan" "$calibrate" "$@" <<'PY'
import sys
from study import check_plan, ensure_free_gpu, load_json
if sys.argv[2] != 'true':
    check_plan(load_json(sys.argv[1]))
if len(set(sys.argv[3:])) != len(sys.argv[3:]):
    raise ValueError('Select distinct GPUs')
for gpu in sys.argv[3:]:
    ensure_free_gpu(gpu)
PY
mkdir -p results
queue_log=results/queue.log
queue_exit=results/queue.exit
if $calibrate; then
  (( ${#retry_args[@]} == 0 )) || { echo 'Calibration never retries failed trials automatically.' >&2; exit 2; }
  queue_log=results/calibration-queue.log
  queue_exit=results/calibration-queue.exit
  printf -v job '%q ' .venv/bin/python -u calibration.py --spec "$plan" --gpus "$@"
else
  printf -v job '%q ' .venv/bin/python -u study.py run --plan "$plan" --gpus "$@" --hours "${STUDY_HOURS:-8}" "${retry_args[@]}"
fi
rm -f "$queue_exit"
# A persistent tmux server keeps its original environment. Load scratch cache
# paths in the new pane as well as in this shell before importing Mamba.
job="set -o pipefail; if [[ -f ../env.sh ]]; then source ../env.sh || exit; fi; $job 2>&1 | tee -i -a $queue_log; code=\${PIPESTATUS[0]}; printf '%s\\n' \"\$code\" > $queue_exit; exit \"\$code\""
tmux new-session -d -s optimizer-training -c "$PWD" bash -c "$job"
tmux set-option -t optimizer-training prefix C-a
tmux set-option -t optimizer-training mouse on
printf 'Started on this server. Attach: tmux attach -t optimizer-training\nDetach: Ctrl+A, then d. Stop and save: Ctrl+C inside the session.\nQueue log: %s/%s\n' "$PWD" "$queue_log"
