#!/bin/bash
#SBATCH --job-name=trellis2-step3b-flux-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=xgpg[0-7]
#SBATCH --gpus=a100-40
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/h/hongshan/cp4281-as2/logs/step3b-flux-smoke-%j.log
#SBATCH --error=/home/h/hongshan/cp4281-as2/logs/step3b-flux-smoke-%j.log

set -eo pipefail

PROJECT_ROOT="$HOME/cp4281-as2"
ENV_PREFIX="$PROJECT_ROOT/envs/trellis2"
HF_HOME="$PROJECT_ROOT/hf-cache"
HF_HUB_CACHE="$HF_HOME/hub"

FLUX_REV="741f7c3ce8b383c54771c7003378a50191e9efe9"
FLUX_CACHE="$HF_HUB_CACHE/models--black-forest-labs--FLUX.1-schnell"
FLUX_SNAPSHOT="$FLUX_CACHE/snapshots/$FLUX_REV"
EXPERIMENT_SPEC="$PROJECT_ROOT/inputs/step3/experiment_spec.json"
OUTPUT_PARENT="$PROJECT_ROOT/outputs/step3/text_conditioned/flux_smoke"
OUTPUT_DIR="$OUTPUT_PARENT/txt02-toolbox-job-$SLURM_JOB_ID"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export HF_HOME HF_HUB_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

LOCAL_JOB_ROOT=$(mktemp -d "/tmp/${USER}-trellis-step3b-flux-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$LOCAL_JOB_ROOT"' EXIT
export TMPDIR="$LOCAL_JOB_ROOT/tmp"
export TRITON_CACHE_DIR="$LOCAL_JOB_ROOT/triton-cache"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$OUTPUT_PARENT"

echo "Start: $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
echo "Job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python --version

test -f "$FLUX_CACHE/refs/main"
ACTUAL_FLUX_REV=$(tr -d '\r\n' < "$FLUX_CACHE/refs/main")
if [ "$ACTUAL_FLUX_REV" != "$FLUX_REV" ]; then
    echo "ERROR: unexpected cached FLUX revision: $ACTUAL_FLUX_REV"
    exit 1
fi
test -f "$FLUX_SNAPSHOT/model_index.json"
test -f "$EXPERIMENT_SPEC"
test ! -e "$OUTPUT_DIR"

python -m json.tool "$EXPERIMENT_SPEC" > /dev/null
python -c 'import accelerate, diffusers, torch, transformers; print("diffusers", diffusers.__version__); print("accelerate", accelerate.__version__); print("transformers", transformers.__version__); print("torch", torch.__version__)'

python "$PROJECT_ROOT/scripts/step3b_flux_generate.py" \
    --case-id "txt02_toolbox" \
    --experiment-spec "$EXPERIMENT_SPEC" \
    --model-path "$FLUX_SNAPSHOT" \
    --output-dir "$OUTPUT_DIR" \
    --seed 42 \
    --height 1024 \
    --width 1024 \
    --num-inference-steps 4 \
    --guidance-scale 0.0 \
    --max-sequence-length 256

test -s "$OUTPUT_DIR/intermediate.png"
test -s "$OUTPUT_DIR/metadata.json"

python - "$OUTPUT_DIR/metadata.json" "$OUTPUT_DIR/intermediate.png" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

metadata_path = Path(sys.argv[1])
image_path = Path(sys.argv[2])
with metadata_path.open(encoding="utf-8") as handle:
    metadata = json.load(handle)

assert metadata["status"] == "completed", metadata
assert metadata["case_id"] == "txt02_toolbox", metadata
assert "{subject_description}" not in metadata["expanded_prompt"], metadata
assert metadata["seed"] == 42, metadata
assert metadata["generation_parameters"]["num_inference_steps"] == 4, metadata
assert metadata["generation_parameters"]["guidance_scale"] == 0.0, metadata
assert metadata["model"]["revision"] == "741f7c3ce8b383c54771c7003378a50191e9efe9", metadata
assert metadata["timings_seconds"]["text_to_image"] > 0, metadata

with Image.open(image_path) as image:
    assert image.size == (1024, 1024), image.size
    assert image.mode == "RGB", image.mode

digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
assert digest == metadata["output_files"]["intermediate_image_sha256"], metadata
print("FLUX smoke metadata verification: PASSED")
PY

echo "Output: $OUTPUT_DIR"
du -sh "$OUTPUT_DIR"
echo "Step 3b FLUX smoke test: PASSED"
echo "Finish: $(date --iso-8601=seconds)"
