#!/bin/bash
#SBATCH --job-name=trellis2-step3b-text-verify
#SBATCH --partition=long
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/home/h/hongshan/cp4281-as2/logs/step3b-text-stack-verify-%j.log
#SBATCH --error=/home/h/hongshan/cp4281-as2/logs/step3b-text-stack-verify-%j.log

set -eo pipefail

PROJECT_ROOT="$HOME/cp4281-as2"
ENV_PREFIX="$PROJECT_ROOT/envs/trellis2"
HF_HOME="$PROJECT_ROOT/hf-cache"
HF_HUB_CACHE="$HF_HOME/hub"
FLUX_REV="741f7c3ce8b383c54771c7003378a50191e9efe9"
FLUX_SNAPSHOT="$HF_HUB_CACHE/models--black-forest-labs--FLUX.1-schnell/snapshots/$FLUX_REV"
MANIFEST="$PROJECT_ROOT/logs/step3b-text-stack-final-pip-freeze-$SLURM_JOB_ID.txt"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export HF_HOME HF_HUB_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

echo "Start: $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
echo "Job: $SLURM_JOB_ID"
python --version
python -m pip --version

test -f "$FLUX_SNAPSHOT/model_index.json"

python - "$FLUX_SNAPSHOT/model_index.json" <<'PY'
import json
import platform
import sys
from importlib.metadata import version

import accelerate
import diffusers
import psutil
import torch
from diffusers import FluxPipeline

expected = {
    "torch": "2.6.0+cu124",
    "diffusers": "0.40.0",
    "accelerate": "1.15.0",
    "psutil": "7.2.2",
    "transformers": "5.17.0",
    "huggingface-hub": "1.31.0",
    "safetensors": "0.8.0",
}
actual = {package: version(package) for package in expected}
assert actual == expected, {"expected": expected, "actual": actual}
assert diffusers.__version__ == expected["diffusers"], diffusers.__version__
assert accelerate.__version__ == expected["accelerate"], accelerate.__version__
assert psutil.__version__ == expected["psutil"], psutil.__version__
assert FluxPipeline.__name__ == "FluxPipeline", FluxPipeline.__name__

with open(sys.argv[1], encoding="utf-8") as handle:
    model_index = json.load(handle)
assert model_index.get("_class_name") == "FluxPipeline", model_index

print(f"Platform: {platform.platform()}")
for package, package_version in actual.items():
    print(f"{package}=={package_version}")
print("FluxPipeline import: PASSED")
print("FLUX model_index.json: PASSED")
PY

python -m pip check
python -m pip freeze | LC_ALL=C sort > "$MANIFEST"
test -s "$MANIFEST"

echo "Manifest: $MANIFEST"
echo "STEP 3B TEXT STACK VERIFICATION: PASSED"
echo "Finish: $(date --iso-8601=seconds)"
