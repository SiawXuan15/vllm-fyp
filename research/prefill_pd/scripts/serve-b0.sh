#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${MODEL:?Source completed b0 config first}"
: "${BASELINE_COMMIT:?}"
[[ $(uname -s) == Linux ]] || { echo 'Linux NVIDIA server required' >&2; exit 1; }
[[ $(git rev-parse HEAD) == "$BASELINE_COMMIT" ]]
git diff --quiet HEAD -- vllm tests examples requirements
common=("$MODEL" --served-model-name b0 --tokenizer "$MODEL"
  --dtype "${DTYPE:?}" --max-model-len "${MAX_MODEL_LEN:?}"
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:?}"
  --max-num-seqs "${MAX_NUM_SEQS:?}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:?}"
  --enable-prefix-caching --enable-chunked-prefill --scheduling-policy fcfs
  --seed 0)
case "${1:-}" in
  prefill) gpu=${PREFILL_GPU:?}; port=8100; side=5600; role=kv_producer ;;
  decode) gpu=${DECODE_GPU:?}; port=8200; side=5601; role=kv_consumer ;;
  proxy)
    exec .venv/bin/python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
      --port 8192 --prefiller-hosts localhost --prefiller-ports 8100 \
      --decoder-hosts localhost --decoder-ports 8200 ;;
  *) echo 'Usage: bash serve-b0.sh prefill|decode|proxy' >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=$gpu VLLM_NIXL_SIDE_CHANNEL_PORT=$side
exec .venv/bin/vllm serve "${common[@]}" --port "$port" \
  --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"$role\",\"kv_load_failure_policy\":\"fail\"}"
