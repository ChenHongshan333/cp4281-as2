#!/bin/bash
#SBATCH --job-name=trellis2-step3a-batch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=xgpg[0-7]
#SBATCH --gpus=a100-40
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/h/hongshan/cp4281-as2/logs/step3a-batch-%j.log
#SBATCH --error=/home/h/hongshan/cp4281-as2/logs/step3a-batch-%j.log

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
OUTPUT_PARENT="$PROJECT_ROOT/outputs/step3/image_conditioned"
OUTPUT_ROOT="$OUTPUT_PARENT/batch-$SLURM_JOB_ID"

CASE_IDS=(
    "img01_thin_structure"
    "img02_hard_surface"
    "img03_transparency"
    "img04_organic_shape"
    "img05_complex_shape"
)

INPUT_IMAGES=(
    "$PROJECT_ROOT/inputs/step3/image_conditioned/img01_thin_structure/earphone.JPG"
    "$PROJECT_ROOT/inputs/step3/image_conditioned/img02_hard_surface/white_car.JPG"
    "$PROJECT_ROOT/inputs/step3/image_conditioned/img03_transparency/glass_cup.JPG"
    "$PROJECT_ROOT/inputs/step3/image_conditioned/img04_organic_shape/yellow_sofa.JPG"
    "$PROJECT_ROOT/inputs/step3/image_conditioned/img05_complex_shape/bookshelf.JPG"
)

INPUT_SHA256S=(
    "2aa20de79ed0bc817f4319235f18c57a7d032025c6b470ec23b4aca26c27728c"
    "6856e821856b00ec47360a0c7c924280dc63ed95753af89b1ac80960cb6505fc"
    "a140109cbccf2bd427372c04544d66e7d01424dc5171d18af7027e9271d251e0"
    "b0288103d46d272465c51584361d506b0b05b48c4bb8d7a1a6ef3c8c091e4b3b"
    "598578420aad664d8a66a8f0a37c5c9ce3ef8d46f85dc8d75df7221eb4811358"
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

LOCAL_JOB_ROOT=$(mktemp -d "/tmp/${USER}-trellis-step3a-batch-${SLURM_JOB_ID}-XXXXXX")
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

test "${#CASE_IDS[@]}" -eq 5
test "${#INPUT_IMAGES[@]}" -eq "${#CASE_IDS[@]}"
test "${#INPUT_SHA256S[@]}" -eq "${#CASE_IDS[@]}"
test "$(git -C "$TRELLIS_SOURCE" rev-parse HEAD)" = "$TRELLIS_SOURCE_REV"
check_ref "$HF_HUB_CACHE/models--microsoft--TRELLIS.2-4B" "$MODEL_REV"
check_ref "$HF_HUB_CACHE/models--microsoft--TRELLIS-image-large" "$TRELLIS_IMAGE_REV"
check_ref "$HF_HUB_CACHE/models--facebook--dinov3-vitl16-pretrain-lvd1689m" "$DINOV3_REV"
check_ref "$HF_HUB_CACHE/models--briaai--RMBG-2.0" "$RMBG_REV"
test -f "$MODEL_SNAPSHOT/pipeline.json"
test -f "$EXPERIMENT_SPEC"
test -f "$TRELLIS_SOURCE/assets/hdri/forest.exr"
test ! -e "$OUTPUT_ROOT"

for index in "${!CASE_IDS[@]}"; do
    input_path="${INPUT_IMAGES[$index]}"
    expected_sha256="${INPUT_SHA256S[$index]}"
    test -f "$input_path"
    test "$(sha256sum "$input_path" | cut -d ' ' -f 1)" = "$expected_sha256"
done

mkdir -p "$OUTPUT_ROOT"

for index in "${!CASE_IDS[@]}"; do
    case_id="${CASE_IDS[$index]}"
    input_path="${INPUT_IMAGES[$index]}"
    case_output="$OUTPUT_ROOT/$case_id"

    echo "Case start: $case_id at $(date --iso-8601=seconds)"
    cd "$TRELLIS_SOURCE"
    python "$PROJECT_ROOT/scripts/step3a_generate.py" \
        --case-id "$case_id" \
        --input "$input_path" \
        --output-dir "$case_output" \
        --model-path "$MODEL_SNAPSHOT" \
        --trellis-source "$TRELLIS_SOURCE" \
        --envmap "$TRELLIS_SOURCE/assets/hdri/forest.exr" \
        --experiment-spec "$EXPERIMENT_SPEC" \
        --seed 42 \
        --pipeline-type 1024_cascade \
        --decimation-target 500000 \
        --texture-size 2048 \
        --render-resolution 768 \
        --render-views 4
    echo "Case finish: $case_id at $(date --iso-8601=seconds)"
done

cd "$PROJECT_ROOT"
python - "$OUTPUT_ROOT" "$SLURM_JOB_ID" <<'PY'
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

output_root = Path(sys.argv[1]).resolve()
job_id = sys.argv[2]
expected_case_ids = [
    "img01_thin_structure",
    "img02_hard_surface",
    "img03_transparency",
    "img04_organic_shape",
    "img05_complex_shape",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


cases = []
for case_id in expected_case_ids:
    case_dir = output_root / case_id
    metadata_path = case_dir / "metadata.json"
    glb_path = case_dir / "asset.glb"
    preprocessed_path = case_dir / "preprocessed.png"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["case_id"] == case_id, metadata
    assert metadata["status"] == "completed", metadata
    assert metadata["background_removal_applied"] is True, metadata
    assert metadata["source_provenance"], metadata
    assert metadata["challenge"], metadata
    assert metadata["selection_rationale"], metadata
    assert metadata["output_files"]["render_key"] == "shaded", metadata
    assert metadata["rendering"]["mode"].startswith("pbr_shaded"), metadata
    assert metadata["timings_seconds"]["per_case_end_to_end"] > 0, metadata
    assert glb_path.is_file() and glb_path.stat().st_size > 0
    assert preprocessed_path.is_file() and preprocessed_path.stat().st_size > 0

    view_paths = sorted(case_dir.glob("view_*.png"))
    assert len(view_paths) == 4, view_paths
    assert all(path.stat().st_size > 0 for path in view_paths)

    cases.append(
        {
            "case_id": case_id,
            "challenge": metadata["challenge"],
            "selection_rationale": metadata["selection_rationale"],
            "source_input": metadata["source_input"],
            "source_sha256": metadata["source_sha256"],
            "background_removal_applied": metadata[
                "background_removal_applied"
            ],
            "gpu_name": metadata["runtime"]["gpu_name"],
            "seed": metadata["seed"],
            "timings_seconds": metadata["timings_seconds"],
            "peak_cuda_memory_bytes": metadata["peak_cuda_memory_bytes"],
            "rendering": metadata["rendering"],
            "glb": {
                "path": str(glb_path),
                "size_bytes": glb_path.stat().st_size,
                "sha256": sha256(glb_path),
            },
            "rendered_views": [str(path) for path in view_paths],
            "metadata_path": str(metadata_path),
        }
    )

summary = {
    "schema_version": 1,
    "stage": "Step 3a - Image-conditioned generation",
    "status": "completed",
    "slurm_job_id": job_id,
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "case_count": len(cases),
    "total_primary_generation_seconds": round(
        sum(case["timings_seconds"]["per_case_end_to_end"] for case in cases),
        4,
    ),
    "cases": cases,
}
summary_path = output_root / "batch_summary.json"
temporary_path = summary_path.with_suffix(".json.tmp")
temporary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
temporary_path.replace(summary_path)

assert len(list(output_root.glob("*/asset.glb"))) == 5
print("Batch summary:", summary_path)
for case in cases:
    print(
        case["case_id"],
        f'{case["timings_seconds"]["per_case_end_to_end"]:.4f}s',
        case["rendering"]["mode"],
        case["glb"]["sha256"],
    )
print("STEP 3A IMAGE-CONDITIONED BATCH: PASSED")
PY

test -s "$OUTPUT_ROOT/batch_summary.json"
echo "Output: $OUTPUT_ROOT"
du -sh "$OUTPUT_ROOT"
echo "Finish: $(date --iso-8601=seconds)"
