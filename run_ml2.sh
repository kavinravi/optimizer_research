#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

gpu=${1:?Usage: bash run_ml2.sh GPU-UUID [smoke|benchmark]}
mode=${2:-smoke}
case "$gpu" in GPU-*) ;; *) echo 'Pass the UUID of one free GPU from nvidia-smi.' >&2; exit 2;; esac
case "$mode" in
  smoke)
    limit=10m
    options=(--tiny --sequence 32 --accumulation 1 --refresh 2 --cycles 1 --sizes 150m --case-timeout 90)
    ;;
  benchmark)
    limit=55m
    options=(--case-timeout 180)
    ;;
  *) echo 'Mode must be smoke or benchmark.' >&2; exit 2;;
esac

image=optimizer-research:ml2
docker image inspect "$image" >/dev/null
mkdir -p results
output=$(mktemp -d "$(pwd)/results/${mode}-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
container="optimizer-${USER:-user}-$(basename "$output")"
docker image inspect "$image" > "$output/image.json"
git rev-parse HEAD > "$output/revision.txt"
nvidia-smi -i "$gpu" > "$output/gpu.txt"

# Stop only this run on interruption. Results live on the host after --rm.
trap 'docker stop "$container" >/dev/null 2>&1 || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
printf 'Results: %s\nContainer: %s\n' "$output" "$container"
docker run --rm --init --name "$container" --stop-timeout 15 \
  --gpus "device=$gpu" --cpus 4 --memory 32g --shm-size 4g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/cache \
  --mount "type=bind,src=$output,dst=/results" \
  "$image" sh -c 'cp /app/environment.txt /results/environment.txt; exec "$@"' sh \
  timeout --kill-after=15s "$limit" python -u hardware_benchmark.py \
  "${options[@]}" --output /results/metrics 2>&1 | tee "$output/run.log"
