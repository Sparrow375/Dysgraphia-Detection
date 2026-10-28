#!/bin/bash
set -e

# Find all nvidia lib and bin directories in the venv
VENV_NVIDIA="/home/nitish/.venvs/inksight/lib/python3.11/site-packages/nvidia"
NV_LIBS=$(find "$VENV_NVIDIA" -type d -name "lib" 2>/dev/null | paste -sd ":" - || true)
NV_BINS=$(find "$VENV_NVIDIA" -type d -name "bin" 2>/dev/null | paste -sd ":" - || true)

export PATH="$NV_BINS:$PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$NV_LIBS:$LD_LIBRARY_PATH"
export TF_ENABLE_ONEDNN_OPTS=0
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=custom-kernel-fusion-rewriter,custom_kernel-fusion-autotuner"

exec /home/nitish/.venvs/inksight/bin/python "$@"
