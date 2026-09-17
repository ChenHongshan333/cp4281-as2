#!/bin/bash
#SBATCH --job-name=trellis2-step4-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=xgph10
#SBATCH --gpus=a100-80
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/h/hongshan/cp4281-as2/logs/step4-smoke-%j.log
#SBATCH --error=/home/h/hongshan/cp4281-as2/logs/step4-smoke-%j.log

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
EXPERIMENT_SPEC="$PROJECT_ROOT/inputs/step3/experiment_spec.json"
INPUT_IMAGE="$PROJECT_ROOT/inputs/step3/image_conditioned/img02_hard_surface/white_car.JPG"
INPUT_SHA256="6856e821856b00ec47360a0c7c924280dc63ed95753af89b1ac80960cb6505fc"
OUTPUT_PARENT="$PROJECT_ROOT/outputs/step4/smoke"
OUTPUT_DIR="$OUTPUT_PARENT/img02-hard-surface-job-$SLURM_JOB_ID"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export HF_HOME HF_HUB_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$PROJECT_ROOT:$TRELLIS_SOURCE:${PYTHONPATH:-}"
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

LOCAL_JOB_ROOT=$(mktemp -d "/tmp/${USER}-trellis-step4-smoke-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$LOCAL_JOB_ROOT"' EXIT
export TMPDIR="$LOCAL_JOB_ROOT/tmp"
export TRITON_CACHE_DIR="$LOCAL_JOB_ROOT/triton-cache"
export TORCH_EXTENSIONS_DIR="$LOCAL_JOB_ROOT/torch-extensions"
export PYTHONPYCACHEPREFIX="$LOCAL_JOB_ROOT/pycache"
mkdir -p \
    "$TMPDIR" \
    "$TRITON_CACHE_DIR" \
    "$TORCH_EXTENSIONS_DIR" \
    "$PYTHONPYCACHEPREFIX" \
    "$OUTPUT_PARENT" \
    "$PROJECT_ROOT/logs"

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
if ! nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; then
    echo "WARNING: nvidia-smi/NVML failed; validating CUDA directly with PyTorch."
fi
python --version
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch CUDA is unavailable")
probe = torch.ones(1, device="cuda")
torch.cuda.synchronize()
print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")
print(f"CUDA tensor probe: {probe}")
PY

test "$(git -C "$TRELLIS_SOURCE" rev-parse HEAD)" = "$TRELLIS_SOURCE_REV"
check_ref "$HF_HUB_CACHE/models--microsoft--TRELLIS.2-4B" "$MODEL_REV"
check_ref "$HF_HUB_CACHE/models--microsoft--TRELLIS-image-large" "$TRELLIS_IMAGE_REV"
check_ref "$HF_HUB_CACHE/models--facebook--dinov3-vitl16-pretrain-lvd1689m" "$DINOV3_REV"
check_ref "$HF_HUB_CACHE/models--briaai--RMBG-2.0" "$RMBG_REV"
test -f "$MODEL_SNAPSHOT/pipeline.json"
test -f "$EXPERIMENT_SPEC"
test -f "$INPUT_IMAGE"
test "$(sha256sum "$INPUT_IMAGE" | cut -d ' ' -f 1)" = "$INPUT_SHA256"
test ! -e "$OUTPUT_DIR"

python -m py_compile \
    "$PROJECT_ROOT/scripts/step4_capture.py" \
    "$PROJECT_ROOT/scripts/step4_decode.py" \
    "$PROJECT_ROOT/scripts/step4_generate.py"
python -B "$PROJECT_ROOT/scripts/step4_capture.py" --self-test
python -B "$PROJECT_ROOT/scripts/step4_decode.py" --self-test
python -B "$PROJECT_ROOT/scripts/step4_generate.py" \
    --self-test \
    --project-root "$PROJECT_ROOT" \
    --experiment-spec "$EXPERIMENT_SPEC"

cd "$TRELLIS_SOURCE"
python "$PROJECT_ROOT/scripts/step4_generate.py" \
    --case-id "img02_hard_surface" \
    --project-root "$PROJECT_ROOT" \
    --experiment-spec "$EXPERIMENT_SPEC" \
    --output-dir "$OUTPUT_DIR" \
    --model-path "$MODEL_SNAPSHOT" \
    --trellis-source "$TRELLIS_SOURCE" \
    --render-resolution 512

test -s "$OUTPUT_DIR/metadata.json"
test -s "$OUTPUT_DIR/input.png"
test -s "$OUTPUT_DIR/preprocessed.png"
test -s "$OUTPUT_DIR/final_render.png"
test "$(find "$OUTPUT_DIR/stage1/frames" -maxdepth 1 -type f -name 'frame_*.png' | wc -l)" -eq 5
test "$(find "$OUTPUT_DIR/stage2/frames" -maxdepth 1 -type f -name 'frame_*.png' | wc -l)" -eq 5
test "$(find "$OUTPUT_DIR/stage1/states" -maxdepth 1 -type f -name 'state_*.pt' | wc -l)" -eq 5
test "$(find "$OUTPUT_DIR/stage2_lr/states" -maxdepth 1 -type f -name 'state_*.pt' | wc -l)" -eq 5
test "$(find "$OUTPUT_DIR/stage2/states" -maxdepth 1 -type f -name 'state_*.pt' | wc -l)" -eq 5

python - "$OUTPUT_DIR/metadata.json" <<'PY'
import json
import math
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    metadata = json.load(handle)

assert metadata["status"] == "completed", metadata
assert metadata["case_id"] == "img02_hard_surface", metadata
assert metadata["pipeline_type"] == "1024_cascade", metadata
assert metadata["capture_policy"]["stage1"]["steps"] == 24, metadata
assert metadata["capture_policy"]["stage2"]["steps"] == 12, metadata
assert metadata["acceptance"]["stage1_frame_count"] == 5, metadata
assert metadata["acceptance"]["stage2_frame_count"] == 5, metadata
assert metadata["acceptance"]["all_true_timesteps_recorded"] is True, metadata
assert metadata["acceptance"]["formal_stage2_is_high_resolution_pass"] is True, metadata
assert metadata["acceptance"]["labelled_strips_created"] is False, metadata

stage1 = metadata["rendered_frames"]["stage1"]
stage2 = metadata["rendered_frames"]["stage2"]
expected_stage1 = [0.5, 5 / 12, 5 / 16, 5 / 28, 0.0]
expected_stage2 = [33 / 34, 3 / 4, 1 / 2, 3 / 14, 0.0]

assert len(stage1) == len(expected_stage1) == 5
assert len(stage2) == len(expected_stage2) == 5
for frame, expected in zip(stage1, expected_stage1):
    assert math.isclose(frame["timestep"], expected, abs_tol=1e-12), frame
    assert frame["measurement"]["structure_resolution"] == 32, frame
    assert frame["measurement"]["occupied_voxels"] >= 0, frame
for frame, expected in zip(stage2, expected_stage2):
    assert math.isclose(frame["timestep"], expected, abs_tol=1e-12), frame
    assert frame["model_label"] == "shape_slat_flow_model_1024", frame
    assert frame["measurement"]["vertices"] > 0, frame
    assert frame["measurement"]["faces"] > 0, frame

low_resolution = metadata["captures"]["stage2_low_resolution_evidence"]
high_resolution = metadata["captures"]["stage2_formal_high_resolution"]
assert low_resolution["model_label"] == "shape_slat_flow_model_512", low_resolution
assert high_resolution["model_label"] == "shape_slat_flow_model_1024", high_resolution
assert len(low_resolution["selected_states"]) == 5, low_resolution
assert len(high_resolution["selected_states"]) == 5, high_resolution

print("Metadata and true-timestep verification: PASSED")
print("Stage 1 occupied voxels:", [
    frame["measurement"]["occupied_voxels"] for frame in stage1
])
print("Stage 2 faces:", [
    frame["measurement"]["faces"] for frame in stage2
])
PY

echo "Output: $OUTPUT_DIR"
du -sh "$OUTPUT_DIR"
echo "STEP 4 SINGLE-CASE SMOKE TEST: PASSED"
echo "Finish: $(date --iso-8601=seconds)"
