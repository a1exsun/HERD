#!/usr/bin/env bash
# Base control on a CUDA node: a job-local vLLM environment in node /tmp, then run.py with the vLLM engine.
# Run from a repository checkout that holds experiments/data and the pinned checkpoint with its registry.
# Usage: packages/baselines/base/bootstrap.sh --profile full|general --output DIR [run.py arguments...]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd -P)"
scratch=$(mktemp -d /tmp/base.XXXXXXXX)
trap 'rm -rf "$scratch"' EXIT
export HF_HOME="$scratch/hf" PIP_CACHE_DIR="$scratch/pip" TRITON_CACHE_DIR="$scratch/triton"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
"${BASE_PYTHON:-/apps/mambaforge/envs/dsks_2025.08/bin/python}" -m venv "$scratch/env"
"$scratch/env/bin/python" -m pip install --quiet --disable-pip-version-check --require-hashes -r "$here/requirements-eval.lock.txt"
"$scratch/env/bin/python" -m pip check
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$scratch/env/bin/python" "$here/run.py" --engine vllm --device cuda "$@"
