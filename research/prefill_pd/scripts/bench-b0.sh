#!/usr/bin/env bash
# One condition per invocation. Restart workers before each invocation.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if [[ $# != 7 ]]; then
  echo 'Usage: bench-b0.sh NEW_RESULT_DIR INPUT OUTPUT QPS PREFIX_PERCENT REPEAT REQUESTS' >&2
  exit 2
fi
out=$1; input=$2; output=$3; rate=$4; prefix_pct=$5; repeat=$6; requests=$7
: "${MODEL:?}" "${BASELINE_COMMIT:?}"
[[ $(uname -s) == Linux ]] || { echo 'Linux NVIDIA server required' >&2; exit 1; }
[[ $(git rev-parse HEAD) == "$BASELINE_COMMIT" ]]
git diff --quiet HEAD -- vllm tests examples requirements
for value in "$input" "$output" "$prefix_pct" "$repeat" "$requests"; do
  [[ $value =~ ^[0-9]+$ ]] || exit 2
done
(( input > 0 && output > 0 && prefix_pct <= 75 && repeat > 0 && requests > 0 ))
(( input + output <= MAX_MODEL_LEN ))
prefix=$((input * prefix_pct / 100))
mkdir "$out" # Refuse to overwrite any prior run, including failed runs.
exec > >(tee "$out/console.log") 2>&1
cmd=(.venv/bin/vllm bench serve --backend vllm --base-url http://localhost:8192
  --model b0 --tokenizer "$MODEL" --dataset-name random
  --random-input-len "$((input-prefix))" --random-prefix-len "$prefix"
  --random-output-len "$output" --random-range-ratio 0
  --request-rate "$rate" --num-prompts "$requests" --seed "$((1000+repeat))"
  --num-warmups 16 --ignore-eos --save-result --save-detailed
  --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "50,99"
  --result-dir "$out" --result-filename result.json)
printf '%q ' "${cmd[@]}" > "$out/command.sh"
printf '\n' >> "$out/command.sh"
git rev-parse HEAD > "$out/revision.txt"
git status --porcelain > "$out/git-status.txt"
uv pip freeze > "$out/packages.txt"
nvidia-smi -q > "$out/gpu.txt"
nvidia-smi topo -m > "$out/topology.txt"
for name in MODEL MAX_MODEL_LEN MAX_BATCHED_TOKENS MAX_NUM_SEQS GPU_MEMORY_UTILIZATION DTYPE PREFILL_GPU DECODE_GPU UCX_TLS UCX_NET_DEVICES; do
  printf '%s=%q\n' "$name" "${!name}" >> "$out/config.env"
done
for port in 8100 8200; do
  curl --fail --silent "http://localhost:$port/health"
  curl --fail --silent "http://localhost:$port/metrics" > "$out/metrics-$port-before.txt"
done
nvidia-smi --query-gpu=timestamp,uuid,memory.used,utilization.gpu --format=csv -lms 200 > "$out/gpu-samples.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
"${cmd[@]}"
for port in 8100 8200; do
  curl --fail --silent "http://localhost:$port/metrics" > "$out/metrics-$port-after.txt"
done
touch "$out/FINISHED"
