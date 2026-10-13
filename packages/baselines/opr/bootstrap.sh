#!/usr/bin/env bash
# OPR job entry: job-local CUDA environment in node /tmp, then one `run.py <phase> --stage N` unit.
# Usage: bootstrap.sh <bundle-dir> <output-dir> <train|evaluate|rollout> --stage N [unit arguments...]
set -euo pipefail
bundle="$1"; output="$2"; phase="$3"; shift 3
scratch=$(mktemp -d /tmp/opr.XXXXXXXX)
trap 'rm -rf "$scratch"' EXIT
export HF_HOME="$scratch/hf" PIP_CACHE_DIR="$scratch/pip" TRITON_CACHE_DIR="$scratch/triton"
export MODELSCOPE_CACHE="$scratch/modelscope"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
/apps/mambaforge/envs/dsks_2025.08/bin/python -m venv "$scratch/env"
# The method scripts call `swift` and `python` from PATH.
export PATH="$scratch/env/bin:$PATH"
if [ "$phase" = evaluate ]; then
  # Evaluation decodes with vLLM in the shared evaluation environment (requirements-eval.lock.txt).
  python -m pip install --quiet --disable-pip-version-check --require-hashes -r "$bundle/requirements-eval.lock.txt"
else
  # Training and rollouts run the method code: ms-swift and vLLM (requirements-cuda.lock.txt).
  python -m pip install --quiet --disable-pip-version-check --require-hashes -r "$bundle/requirements-cuda.lock.txt"
fi
python -m pip check
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python "$bundle/run.py" "$phase" --bundle "$bundle" --output "$output" --checkpoint "$scratch/qwen" --fetch "$@"
