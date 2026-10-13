#!/usr/bin/env bash
# HERD stream job entry: job-local CUDA environment and base checkpoint in /tmp, then `run.py stream`.
# Usage: bootstrap.sh <bundle-dir> <output-dir> [stream arguments]
# HERO_PYTHON selects the CPython 3.10 interpreter (default: python3.10 on PATH).
set -euo pipefail
bundle="$1"; output="$2"; shift 2
scratch=$(mktemp -d /tmp/hero-stream.XXXXXXXX)
trap 'rm -rf "$scratch"' EXIT
export HF_HOME="$scratch/hf" PIP_CACHE_DIR="$scratch/pip" TRITON_CACHE_DIR="$scratch/triton"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${HERO_PYTHON:-python3.10}" -m venv "$scratch/env"  # CPython 3.10
"$scratch/env/bin/python" -m pip install --quiet --disable-pip-version-check --require-hashes -r "$bundle/requirements-cuda.lock.txt"
"$scratch/env/bin/python" -m pip install --quiet --disable-pip-version-check --require-hashes --no-deps -r "$bundle/requirements-kernels.lock.txt"
"$scratch/env/bin/python" -m pip check
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$scratch/env/bin/python" "$bundle/run.py" stream --bundle "$bundle" --output "$output" \
  --checkpoint "$scratch/qwen" --fetch "$@"
