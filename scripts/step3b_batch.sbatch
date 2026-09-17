#!/bin/bash
#SBATCH --job-name=trellis2-step3b-batch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=xgpg[0-7]
#SBATCH --gpus=a100-40
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/h/hongshan/cp4281-as2/logs/step3b-batch-%j.log
#SBATCH --error=/home/h/hongshan/cp4281-as2/logs/step3b-batch-%j.log

set -eo pipefail

PROJECT_ROOT="$HOME/cp4281-as2"
ENV_PREFIX="$PROJECT_ROOT/envs/trellis2"
TRELLIS_SOURCE="$PROJECT_ROOT/src/TRELLIS.2"
HF_HOME="$PROJECT_ROOT/hf-cache"
HF_HUB_CACHE="$HF_HOME/hub"

TRELLIS_SOURCE_REV="75fbf0183001ed9876c8dbb35de6b68552ee08bd"
TRELLIS_REV="af44b45f2e35a493886929c6d786e563ec68364d"
TRELLIS_IMAGE_REV="25e0d31ffbebe4b5a97464dd851910efc3002d96"
DINOV3_REV="ea8dc2863c51be0a264bab82070e3e8836b02d51"
RMBG_REV="5df4c9c76d8170882c34f6986e848ee07fd0ba43"
FLUX_REV="741f7c3ce8b383c54771c7003378a50191e9efe9"

TRELLIS_SNAPSHOT="$HF_HUB_CACHE/models--microsoft--TRELLIS.2-4B/snapshots/$TRELLIS_REV"
FLUX_SNAPSHOT="$HF_HUB_CACHE/models--black-forest-labs--FLUX.1-schnell/snapshots/$FLUX_REV"
EXPERIMENT_SPEC="$PROJECT_ROOT/inputs/step3/experiment_spec.json"
OUTPUT_PARENT="$PROJECT_ROOT/outputs/step3/text_conditioned"
OUTPUT_ROOT="$OUTPUT_PARENT/batch-$SLURM_JOB_ID"

CASE_IDS=(
    "txt01_desk_fan"
    "txt02_toolbox"
    "txt03_perfume_bottle"
    "txt04_root_sculpture"
    "txt05_lunar_rover"
)

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

LOCAL_JOB_ROOT=$(mktemp -d "/tmp/${USER}-trellis-step3b-batch-${SLURM_JOB_ID}-XXXXXX")
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
if ! nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; then
    echo "WARNING: nvidia-smi/NVML failed; validating CUDA directly with PyTorch."
fi
python --version
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch CUDA is unavailable")
probe = torch.ones(1, device="cuda")
print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")
print(f"CUDA tensor probe: {probe}")
PY

test "${#CASE_IDS[@]}" -eq 5
test "$(git -C "$TRELLIS_SOURCE" rev-parse HEAD)" = "$TRELLIS_SOURCE_REV"
check_ref "$HF_HUB_CACHE/models--microsoft--TRELLIS.2-4B" "$TRELLIS_REV"
check_ref "$HF_HUB_CACHE/models--microsoft--TRELLIS-image-large" "$TRELLIS_IMAGE_REV"
check_ref "$HF_HUB_CACHE/models--facebook--dinov3-vitl16-pretrain-lvd1689m" "$DINOV3_REV"
check_ref "$HF_HUB_CACHE/models--briaai--RMBG-2.0" "$RMBG_REV"
check_ref "$HF_HUB_CACHE/models--black-forest-labs--FLUX.1-schnell" "$FLUX_REV"
test -f "$TRELLIS_SNAPSHOT/pipeline.json"
test -f "$FLUX_SNAPSHOT/model_index.json"
test -f "$TRELLIS_SOURCE/assets/hdri/forest.exr"
test -f "$EXPERIMENT_SPEC"
test ! -e "$OUTPUT_ROOT"
python -m json.tool "$EXPERIMENT_SPEC" > /dev/null
python -c 'import accelerate, diffusers, psutil, torch, transformers; from diffusers import FluxPipeline; print("diffusers", diffusers.__version__); print("accelerate", accelerate.__version__); print("FluxPipeline", FluxPipeline.__name__)'

python - "$EXPERIMENT_SPEC" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    spec = json.load(handle)
text_spec = spec["text_conditioned"]
expected = [
    "txt01_desk_fan",
    "txt02_toolbox",
    "txt03_perfume_bottle",
    "txt04_root_sculpture",
    "txt05_lunar_rover",
]
assert text_spec["status"] in {
    "flux_smoke_complete_batch_ready",
    "batch_scripts_ready",
}, text_spec["status"]
assert text_spec["prompt_template_revision"] == 3, text_spec
assert text_spec["prompt_token_verification"]["status"] == "passed", text_spec
assert [case["case_id"] for case in text_spec["cases"]] == expected, text_spec
assert len(text_spec["smoke_test_history"]) >= 2, text_spec
assert text_spec["smoke_test_history"][-1]["accepted_for_batch"] is True, text_spec
print("Experiment specification preflight: PASSED")
PY

mkdir -p "$OUTPUT_ROOT"

echo "FLUX stage start: $(date --iso-8601=seconds)"
for case_id in "${CASE_IDS[@]}"; do
    case_root="$OUTPUT_ROOT/$case_id"
    flux_output="$case_root/flux"
    mkdir -p "$case_root"
    echo "FLUX case start: $case_id at $(date --iso-8601=seconds)"
    python "$PROJECT_ROOT/scripts/step3b_flux_generate.py" \
        --case-id "$case_id" \
        --experiment-spec "$EXPERIMENT_SPEC" \
        --model-path "$FLUX_SNAPSHOT" \
        --output-dir "$flux_output" \
        --seed 42 \
        --height 1024 \
        --width 1024 \
        --num-inference-steps 4 \
        --guidance-scale 0.0 \
        --max-sequence-length 256
    test -s "$flux_output/intermediate.png"
    test -s "$flux_output/metadata.json"
    echo "FLUX case finish: $case_id at $(date --iso-8601=seconds)"
done
echo "FLUX stage finish: $(date --iso-8601=seconds)"

echo "TRELLIS stage start: $(date --iso-8601=seconds)"
for case_id in "${CASE_IDS[@]}"; do
    case_root="$OUTPUT_ROOT/$case_id"
    flux_input="$case_root/flux/intermediate.png"
    trellis_output="$case_root/trellis"
    echo "TRELLIS case start: $case_id at $(date --iso-8601=seconds)"
    cd "$TRELLIS_SOURCE"
    python "$PROJECT_ROOT/scripts/step3a_generate.py" \
        --case-id "$case_id" \
        --input "$flux_input" \
        --output-dir "$trellis_output" \
        --model-path "$TRELLIS_SNAPSHOT" \
        --trellis-source "$TRELLIS_SOURCE" \
        --envmap "$TRELLIS_SOURCE/assets/hdri/forest.exr" \
        --seed 42 \
        --pipeline-type 1024_cascade \
        --decimation-target 500000 \
        --texture-size 2048 \
        --render-resolution 768 \
        --render-views 4
    test -s "$trellis_output/asset.glb"
    test -s "$trellis_output/metadata.json"
    test "$(find "$trellis_output" -maxdepth 1 -type f -name 'view_*.png' | wc -l)" -eq 4
    echo "TRELLIS case finish: $case_id at $(date --iso-8601=seconds)"
done
echo "TRELLIS stage finish: $(date --iso-8601=seconds)"

cd "$PROJECT_ROOT"
python "$PROJECT_ROOT/scripts/step3b_summarize.py" \
    --output-root "$OUTPUT_ROOT" \
    --experiment-spec "$EXPERIMENT_SPEC" \
    --slurm-job-id "$SLURM_JOB_ID"

test -s "$OUTPUT_ROOT/batch_summary.json"
test "$(find "$OUTPUT_ROOT" -mindepth 3 -maxdepth 3 -type f -name 'intermediate.png' | wc -l)" -eq 5
test "$(find "$OUTPUT_ROOT" -mindepth 3 -maxdepth 3 -type f -name 'asset.glb' | wc -l)" -eq 5
test "$(find "$OUTPUT_ROOT" -mindepth 3 -maxdepth 3 -type f -name 'view_*.png' | wc -l)" -eq 20

echo "Output: $OUTPUT_ROOT"
du -sh "$OUTPUT_ROOT"
echo "Step 3b text-conditioned batch: PASSED"
echo "Finish: $(date --iso-8601=seconds)"
