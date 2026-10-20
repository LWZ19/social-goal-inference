#!/usr/bin/env bash
# Start the pipeline (conda env sgi) and the simulator (conda env sgi-habitat) together; stop both on exit.
#   sim/demo.sh [--model ENTRY] [--goal constraints|direct_marking] [-- simulator options]
# Everything after -- goes to python -m sgi_habitat.run, e.g.
#   sim/demo.sh --model soclaas_qwen38_low -- --record runs/habitat/demo.mp4
set -euo pipefail
cd "$(dirname "$0")/.."

model=mock
goal=constraints
port=8765
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model="$2"; shift 2 ;;
    --goal) goal="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown option $1 (simulator options go after --)" >&2; exit 2 ;;
  esac
done

# conda may not be hooked into this shell (no `conda init`): fall back to the usual install folders.
base="$(conda info --base 2>/dev/null || true)"
for guess in "$base" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"; do
  [[ -n "$guess" && -f "$guess/etc/profile.d/conda.sh" ]] && { base="$guess"; break; }
done
[[ -f "$base/etc/profile.d/conda.sh" ]] || { echo "conda not found; install Miniconda or run: conda init" >&2; exit 2; }
source "$base/etc/profile.d/conda.sh"
if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
  echo "a pipeline is already listening on port $port (is another demo open?): close it, or pass --port 8766" >&2
  exit 2
fi
conda activate sgi
python -m sgi.serve --model "$model" --goal "$goal" --port "$port" &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT

for _ in $(seq 50); do                       # the server is up once /health answers
  curl -sf "http://127.0.0.1:$port/health" >/dev/null && break
  kill -0 $server 2>/dev/null || { echo "the pipeline did not start" >&2; exit 2; }
  sleep 0.2
done

conda activate sgi-habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
# ReplicaCAD's dataset file names navmeshes of scenes that are not downloaded: one harmless error line each.
python -m sgi_habitat.run --sgi-url "http://127.0.0.1:$port" "$@" \
  2> >(grep --line-buffered -v -e "navmesh_instances" -e "duplicate static plugin" >&2)
