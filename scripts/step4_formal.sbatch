#!/bin/bash
#SBATCH --job-name=trellis2-step4-formal
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=xgph10
#SBATCH --gpus=a100-80
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/h/hongshan/cp4281-as2/logs/step4-formal-%j.log
#SBATCH --error=/home/h/hongshan/cp4281-as2/logs/step4-formal-%j.log

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
SMOKE_OUTPUT="$PROJECT_ROOT/outputs/step4/smoke/img02-hard-surface-job-856096"
OUTPUT_PARENT="$PROJECT_ROOT/outputs/step4/formal"
BATCH_ROOT="$OUTPUT_PARENT/batch-$SLURM_JOB_ID"

ARMCHAIR_INPUT="$PROJECT_ROOT/inputs/step3/image_conditioned/img04_organic_shape/yellow_sofa.JPG"
ARMCHAIR_SHA256="b0288103d46d272465c51584361d506b0b05b48c4bb8d7a1a6ef3c8c091e4b3b"
BOOKSHELF_INPUT="$PROJECT_ROOT/inputs/step3/image_conditioned/img05_complex_shape/bookshelf.JPG"
BOOKSHELF_SHA256="598578420aad664d8a66a8f0a37c5c9ce3ef8d46f85dc8d75df7221eb4811358"

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

LOCAL_JOB_ROOT=$(mktemp -d "/tmp/${USER}-trellis-step4-formal-${SLURM_JOB_ID}-XXXXXX")
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

validate_output() {
    output_dir="$1"
    expected_case_id="$2"
    test -s "$output_dir/metadata.json"
    test -s "$output_dir/input.png"
    test -s "$output_dir/preprocessed.png"
    test -s "$output_dir/final_render.png"
    test -s "$output_dir/visualizations/stage1_diffusion_strip.png"
    test -s "$output_dir/visualizations/stage2_diffusion_strip.png"
    test "$(find "$output_dir/stage1/frames" -maxdepth 1 -type f -name 'frame_*.png' | wc -l)" -eq 5
    test "$(find "$output_dir/stage2/frames" -maxdepth 1 -type f -name 'frame_*.png' | wc -l)" -eq 5
    test "$(find "$output_dir/stage1/states" -maxdepth 1 -type f -name 'state_*.pt' | wc -l)" -eq 5
    test "$(find "$output_dir/stage2_lr/states" -maxdepth 1 -type f -name 'state_*.pt' | wc -l)" -eq 5
    test "$(find "$output_dir/stage2/states" -maxdepth 1 -type f -name 'state_*.pt' | wc -l)" -eq 5

    python - "$output_dir/metadata.json" "$expected_case_id" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1]).resolve()
expected_case_id = sys.argv[2]
output_root = metadata_path.parent
with metadata_path.open(encoding="utf-8") as handle:
    metadata = json.load(handle)

assert metadata["status"] == "completed", metadata
assert metadata["case_id"] == expected_case_id, metadata
assert metadata["pipeline_type"] == "1024_cascade", metadata
assert metadata["capture_policy"]["stage1"]["steps"] == 24, metadata
assert metadata["capture_policy"]["stage2"]["steps"] == 12, metadata
assert metadata["acceptance"]["stage1_frame_count"] == 5, metadata
assert metadata["acceptance"]["stage2_frame_count"] == 5, metadata
assert metadata["acceptance"]["all_true_timesteps_recorded"] is True, metadata
assert metadata["acceptance"]["formal_stage2_is_high_resolution_pass"] is True, metadata
assert metadata["acceptance"]["labelled_strips_created"] is True, metadata

expected = {
    "stage1": [0.5, 5 / 12, 5 / 16, 5 / 28, 0.0],
    "stage2": [33 / 34, 3 / 4, 1 / 2, 3 / 14, 0.0],
}
for stage, expected_timesteps in expected.items():
    frames = metadata["rendered_frames"][stage]
    assert len(frames) == len(expected_timesteps) == 5
    for frame, timestep in zip(frames, expected_timesteps):
        assert math.isclose(frame["timestep"], timestep, abs_tol=1e-12), frame

for stage in ("stage1", "stage2"):
    artifact = metadata["labelled_strips"][stage]
    path = (output_root / artifact["path"]).resolve()
    path.relative_to(output_root)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == artifact["sha256"], artifact
    assert artifact["panel_count"] == 5, artifact

low_resolution = metadata["captures"]["stage2_low_resolution_evidence"]
high_resolution = metadata["captures"]["stage2_formal_high_resolution"]
assert low_resolution["model_label"] == "shape_slat_flow_model_512"
assert high_resolution["model_label"] == "shape_slat_flow_model_1024"
assert len(low_resolution["selected_states"]) == 5
assert len(high_resolution["selected_states"]) == 5
print(f"Validated Step 4 output: {expected_case_id}")
PY
}

run_case() {
    case_id="$1"
    output_dir="$BATCH_ROOT/$case_id"
    test ! -e "$output_dir"
    cd "$TRELLIS_SOURCE"
    python "$PROJECT_ROOT/scripts/step4_generate.py" \
        --case-id "$case_id" \
        --project-root "$PROJECT_ROOT" \
        --experiment-spec "$EXPERIMENT_SPEC" \
        --output-dir "$output_dir" \
        --model-path "$MODEL_SNAPSHOT" \
        --trellis-source "$TRELLIS_SOURCE" \
        --render-resolution 512
    python "$PROJECT_ROOT/scripts/step4_make_strips.py" \
        --output-dir "$output_dir"
    validate_output "$output_dir" "$case_id"
}

echo "Start: $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
echo "Job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
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
test -f "$ARMCHAIR_INPUT"
test -f "$BOOKSHELF_INPUT"
test "$(sha256sum "$ARMCHAIR_INPUT" | cut -d ' ' -f 1)" = "$ARMCHAIR_SHA256"
test "$(sha256sum "$BOOKSHELF_INPUT" | cut -d ' ' -f 1)" = "$BOOKSHELF_SHA256"
test -d "$SMOKE_OUTPUT"
test ! -e "$BATCH_ROOT"

python -m py_compile \
    "$PROJECT_ROOT/scripts/step4_capture.py" \
    "$PROJECT_ROOT/scripts/step4_decode.py" \
    "$PROJECT_ROOT/scripts/step4_generate.py" \
    "$PROJECT_ROOT/scripts/step4_make_strips.py"
python -B "$PROJECT_ROOT/scripts/step4_capture.py" --self-test
python -B "$PROJECT_ROOT/scripts/step4_decode.py" --self-test
python -B "$PROJECT_ROOT/scripts/step4_generate.py" \
    --self-test \
    --project-root "$PROJECT_ROOT" \
    --experiment-spec "$EXPERIMENT_SPEC"
python -B "$PROJECT_ROOT/scripts/step4_make_strips.py" --self-test

mkdir -p "$BATCH_ROOT"

test ! -e "$SMOKE_OUTPUT/visualizations/stage1_diffusion_strip.png"
test ! -e "$SMOKE_OUTPUT/visualizations/stage2_diffusion_strip.png"
python "$PROJECT_ROOT/scripts/step4_make_strips.py" \
    --output-dir "$SMOKE_OUTPUT"
validate_output "$SMOKE_OUTPUT" "img02_hard_surface"

run_case "img04_organic_shape"
run_case "img05_complex_shape"

python - "$BATCH_ROOT" "$SMOKE_OUTPUT" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

batch_root = Path(sys.argv[1]).resolve()
case_paths = {
    "img02_hard_surface": Path(sys.argv[2]).resolve(),
    "img04_organic_shape": batch_root / "img04_organic_shape",
    "img05_complex_shape": batch_root / "img05_complex_shape",
}
cases = []
for case_id, path in case_paths.items():
    with (path / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["status"] == "completed"
    assert metadata["case_id"] == case_id
    assert metadata["acceptance"]["labelled_strips_created"] is True
    cases.append(
        {
            "case_id": case_id,
            "role": "accepted_smoke" if case_id == "img02_hard_surface" else "formal_batch",
            "output": str(path),
            "stage1_strip": metadata["labelled_strips"]["stage1"]["path"],
            "stage2_strip": metadata["labelled_strips"]["stage2"]["path"],
            "final_render": metadata["final_render"]["path"],
        }
    )

summary = {
    "schema_version": 1,
    "stage": "Step 4 - Visualise the diffusion process",
    "status": "completed",
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "case_count": len(cases),
    "cases": cases,
}
with (batch_root / "batch_summary.json").open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(summary, handle, indent=2)
    handle.write("\n")
print("Step 4 batch summary: PASSED")
PY

test -s "$BATCH_ROOT/batch_summary.json"
echo "Batch output: $BATCH_ROOT"
du -sh "$BATCH_ROOT"
echo "STEP 4 FORMAL THREE-CASE VERIFICATION: PASSED"
echo "Finish: $(date --iso-8601=seconds)"
