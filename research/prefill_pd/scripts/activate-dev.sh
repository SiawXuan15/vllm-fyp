# Source this file from the repository root:
#   source research/prefill_pd/scripts/activate-dev.sh
# Changes apply only to the current shell.

# check if .venv exists 
if [[ ! -f .venv/bin/activate ]]; then
    echo 'Run this from the vllm repository root with .venv created.' >&2
    return 1
fi

#update path 
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate


unset VLLM_USE_PRECOMPILED VLLM_USE_PRECOMPILED_RUST
unset VLLM_PRECOMPILED_WHEEL_COMMIT VLLM_TARGET_DEVICE
