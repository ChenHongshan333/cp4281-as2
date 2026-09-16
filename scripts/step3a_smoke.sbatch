#!/bin/bash
#SBATCH --job-name=trellis2-step3a-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=a100-40
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/h/hongshan/cp4281-as2/logs/step3a-smoke-%j.log
#SBATCH --error=/home/h/hongshan/cp4281-as2/logs/step3a-smoke-%j.log

set -eo pipefail

PROJECT_ROOT="$HOME/cp4281-as2"
ENV_PREFIX="$PROJECT_ROOT/envs/trellis2"
TRELLIS_SOURCE="$PROJECT_ROOT/src/TRELLIS.2"
HF_HOME="$PROJECT_ROOT/hf-cache"
HF_HUB_CACHE="$HF_HOME/hub"

TRELLIS_SOURCE_REV="75fbf0183001ed9876c8dbb35de6b68552ee08bd"
MODEL_REV="af44b45f2e35a493886929c6d786e563ec68364d"
TRELLIS_IMAGE_REV="25e0d31ffbebe4b5a97464dd851910efc3002d96"
DINOV3_REV="ea8dc2863c51be0a264bab82070e3e8836b02d51"
RMBG_REV="5df4c9c76d8170882c34f6986e848ee07fd0ba43"

MODEL_SNAPSHOT="$HF_HUB_CACHE/models--microsoft--TRELLIS.2-4B/snapshots/$MODEL_REV"
INPUT_IMAGE="$PROJECT_ROOT/inputs/step3/image_conditioned/img04_organic_shape/yellow_sofa.JPG"
INPUT_SHA256="b0288103d46d272465c51584361d506b0b05b48c4bb8d7a1a6ef3c8c091e4b3b"
OUTPUT_PARENT="$PROJECT_ROOT/outputs/step3/smoke"
OUTPUT_DIR="$OUTPUT_PARENT/img04-yellow-sofa-job-$SLURM_JOB_ID"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export HF_HOME HF_HUB_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$TRELLIS_SOURCE:${PYTHONPATH:-}"
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

LOCAL_JOB_ROOT=$(mktemp -d "/tmp/${USER}-trellis-step3a-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$LOCAL_JOB_ROOT"' EXIT
export TMPDIR="$LOCAL_JOB_ROOT/tmp"
export TRITON_CACHE_DIR="$LOCAL_JOB_ROOT/triton-cache"
export TORCH_EXTENSIONS_DIR="$LOCAL_JOB_ROOT/torch-extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$OUTPUT_PARENT"

check_ref() {
    repo_dir="$1"
    expected="$2"
    actual=$(tr -d '\r\n' < "$repo_dir/refs/main")
    if [ "$actual" != "$expected" ]; then
        echo "ERROR: unexpected cached revision in $repo_dir: $actual"
        exit 1
    fi
}

echo "Start: $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
echo "Job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python --version

test "$(git -C "$TRELLIS_SOURCE" rev-parse HEAD)" = "$TRELLIS_SOURCE_REV"
check_ref "$HF_HUB_CACHE/models--microsoft--TRELLIS.2-4B" "$MODEL_REV"
check_ref "$HF_HUB_CACHE/models--microsoft--TRELLIS-image-large" "$TRELLIS_IMAGE_REV"
check_ref "$HF_HUB_CACHE/models--facebook--dinov3-vitl16-pretrain-lvd1689m" "$DINOV3_REV"
check_ref "$HF_HUB_CACHE/models--briaai--RMBG-2.0" "$RMBG_REV"
test -f "$MODEL_SNAPSHOT/pipeline.json"
test -f "$INPUT_IMAGE"
test -f "$TRELLIS_SOURCE/assets/hdri/forest.exr"
test ! -e "$OUTPUT_DIR"
test "$(sha256sum "$INPUT_IMAGE" | cut -d ' ' -f 1)" = "$INPUT_SHA256"
echo "Environment map bytes: $(wc -c < "$TRELLIS_SOURCE/assets/hdri/forest.exr")"

cd "$TRELLIS_SOURCE"
python "$PROJECT_ROOT/scripts/step3a_generate.py" \
    --case-id "img04_organic_shape" \
    --input "$INPUT_IMAGE" \
    --output-dir "$OUTPUT_DIR" \
    --model-path "$MODEL_SNAPSHOT" \
    --trellis-source "$TRELLIS_SOURCE" \
    --envmap "$TRELLIS_SOURCE/assets/hdri/forest.exr" \
    --seed 42 \
    --pipeline-type 1024_cascade \
    --decimation-target 500000 \
    --texture-size 2048 \
    --render-resolution 768 \
    --render-views 4

test -s "$OUTPUT_DIR/asset.glb"
test -s "$OUTPUT_DIR/preprocessed.png"
test -s "$OUTPUT_DIR/metadata.json"
test "$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'view_*.png' | wc -l)" -ge 2

python - "$OUTPUT_DIR/metadata.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    metadata = json.load(handle)

assert metadata["status"] == "completed", metadata
assert metadata["background_removal_applied"] is True, metadata
assert metadata["timings_seconds"]["per_case_end_to_end"] > 0, metadata
assert metadata["output_files"]["render_key"] == "shaded", metadata
assert metadata["rendering"]["mode"].startswith("pbr_shaded"), metadata
print("Metadata verification: PASSED")
PY

echo "Output: $OUTPUT_DIR"
du -sh "$OUTPUT_DIR"
echo "Step 3a smoke test: PASSED"
echo "Finish: $(date --iso-8601=seconds)"
