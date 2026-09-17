#!/bin/bash
#SBATCH --job-name=trellis2-step3b-accelerate
#SBATCH --partition=long
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/home/h/hongshan/cp4281-as2/logs/step3b-accelerate-%j.log
#SBATCH --error=/home/h/hongshan/cp4281-as2/logs/step3b-accelerate-%j.log

set -eo pipefail

PROJECT_ROOT="$HOME/cp4281-as2"
ENV_PREFIX="$PROJECT_ROOT/envs/trellis2"
ACCELERATE_VERSION="1.15.0"
WHEEL_NAME="accelerate-1.15.0-py3-none-any.whl"
WHEEL_SHA256="97eacca0b73e45cb867dbf8c5d5d4dc32219544300e0c8992c7334dc2ef33cec"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

LOCAL_JOB_ROOT=$(mktemp -d "/tmp/${USER}-trellis-step3b-accelerate-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$LOCAL_JOB_ROOT"' EXIT
export TMPDIR="$LOCAL_JOB_ROOT/tmp"
export PIP_CACHE_DIR="$LOCAL_JOB_ROOT/pip-cache"
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$LOCAL_JOB_ROOT/wheels" "$PROJECT_ROOT/logs"

echo "Start: $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
echo "Job: $SLURM_JOB_ID"
python --version
python -m pip --version
python -m pip freeze | LC_ALL=C sort \
    > "$PROJECT_ROOT/logs/step3b-accelerate-before-$SLURM_JOB_ID.txt"

python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for package in ("torch", "diffusers", "transformers", "huggingface-hub", "safetensors"):
    try:
        print(f"Before install: {package}=={version(package)}")
    except PackageNotFoundError:
        print(f"Before install: {package}=MISSING")

try:
    print(f"Before install: accelerate=={version('accelerate')}")
except PackageNotFoundError:
    print("Before install: accelerate=MISSING")
PY

python -m pip download \
    --no-deps \
    --only-binary=:all: \
    --dest "$LOCAL_JOB_ROOT/wheels" \
    "accelerate==$ACCELERATE_VERSION"

test -s "$LOCAL_JOB_ROOT/wheels/$WHEEL_NAME"
echo "$WHEEL_SHA256  $LOCAL_JOB_ROOT/wheels/$WHEEL_NAME" | sha256sum --check --strict

python -m pip install \
    --upgrade-strategy only-if-needed \
    "$LOCAL_JOB_ROOT/wheels/$WHEEL_NAME"

python - <<'PY'
from importlib.metadata import version

import accelerate
from diffusers import FluxPipeline

expected = "1.15.0"
actual = version("accelerate")
assert actual == expected, (actual, expected)
assert accelerate.__version__ == expected, accelerate.__version__
print(f"Accelerate import: PASSED ({actual})")
print(f"FluxPipeline import: PASSED ({FluxPipeline.__name__})")
PY

python -m pip check
python -m pip freeze | LC_ALL=C sort \
    > "$PROJECT_ROOT/logs/step3b-accelerate-pip-freeze-$SLURM_JOB_ID.txt"

echo "Packages added or changed by this job:"
LC_ALL=C comm -13 \
    "$PROJECT_ROOT/logs/step3b-accelerate-before-$SLURM_JOB_ID.txt" \
    "$PROJECT_ROOT/logs/step3b-accelerate-pip-freeze-$SLURM_JOB_ID.txt"

echo "Before manifest: $PROJECT_ROOT/logs/step3b-accelerate-before-$SLURM_JOB_ID.txt"
echo "Manifest: $PROJECT_ROOT/logs/step3b-accelerate-pip-freeze-$SLURM_JOB_ID.txt"
echo "STEP 3B ACCELERATE INSTALLATION: PASSED"
echo "Finish: $(date --iso-8601=seconds)"
