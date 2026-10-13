#!/usr/bin/env bash
# OPHSD job entry: job-local CUDA environment in node /tmp, then `run.py train|evaluate`.
# Usage: bootstrap.sh <bundle-dir> <output-dir> <train|evaluate> [arguments...]
set -euo pipefail
bundle="$1"; output="$2"; command="$3"; shift 3
scratch=$(mktemp -d /tmp/ophsd.XXXXXXXX)
trap 'rm -rf "$scratch"' EXIT
export HF_HOME="$scratch/hf" PIP_CACHE_DIR="$scratch/pip" TRITON_CACHE_DIR="$scratch/triton"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/apps/mambaforge/envs/dsks_2025.08/bin/python -m venv "$scratch/env"
if [[ "$command" == evaluate ]]; then
  # Evaluation decodes with vLLM in its own locked environment (requirements-eval.lock.txt).
  "$scratch/env/bin/python" -m pip install --quiet --disable-pip-version-check --require-hashes -r "$bundle/requirements-eval.lock.txt"
else
  "$scratch/env/bin/python" -m pip install --quiet --disable-pip-version-check --require-hashes -r "$bundle/requirements-cuda.lock.txt"
  "$scratch/env/bin/python" -m pip install --quiet --disable-pip-version-check --require-hashes --no-deps -r "$bundle/requirements-kernels.lock.txt"
fi
"$scratch/env/bin/python" -m pip check
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
run() {
  "$scratch/env/bin/python" "$bundle/run.py" "$command" --bundle "$bundle" --output "$output" \
    --checkpoint "$scratch/qwen" --embedding "$scratch/bge" --fetch "$@"
}
# OPHSD_SPLIT_AT=N first stops after stage N, then resumes in a fresh process (resume check).
if [[ "$command" == train && -n "${OPHSD_SPLIT_AT:-}" ]]; then
  run --through-stage "$OPHSD_SPLIT_AT" "$@"
fi
run "$@"
