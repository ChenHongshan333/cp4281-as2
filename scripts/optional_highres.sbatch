#!/bin/bash
#SBATCH --job-name=trellis2-opt-highres
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=xgph10,xgpj0
#SBATCH --gpus=a100-80
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/h/hongshan/cp4281-as2/logs/optional-highres-%j.log
#SBATCH --error=/home/h/hongshan/cp4281-as2/logs/optional-highres-%j.log

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
EXPERIMENT_SPEC="$PROJECT_ROOT/inputs/optional/high_resolution_spec.json"
INPUT_IMAGE="$PROJECT_ROOT/inputs/step3/image_conditioned/img05_complex_shape/bookshelf.JPG"
INPUT_SHA256="598578420aad664d8a66a8f0a37c5c9ce3ef8d46f85dc8d75df7221eb4811358"
BASELINE_DIR="$PROJECT_ROOT/outputs/step3/image_conditioned/batch-853514/img05_complex_shape"
OUTPUT_ROOT="$PROJECT_ROOT/outputs/optional/high_resolution/job-$SLURM_JOB_ID"
OUTPUT_DIR="$OUTPUT_ROOT/img05_complex_shape_1536"

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

LOCAL_JOB_ROOT=$(mktemp -d "/tmp/${USER}-trellis-opt-highres-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$LOCAL_JOB_ROOT"' EXIT
export TMPDIR="$LOCAL_JOB_ROOT/tmp"
export TRITON_CACHE_DIR="$LOCAL_JOB_ROOT/triton-cache"
export TORCH_EXTENSIONS_DIR="$LOCAL_JOB_ROOT/torch-extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$(dirname "$OUTPUT_ROOT")"

check_ref() {
    repo_dir="$1"
    expected="$2"
    actual=$(tr -d '\r\n' < "$repo_dir/refs/main")
    if [ "$actual" != "$expected" ]; then
        echo "ERROR: unexpected cached revision in $repo_dir: $actual" >&2
        exit 1
    fi
}

echo "Start: $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
echo "Job: $SLURM_JOB_ID"
echo "Experiment: bookshelf 1024_cascade baseline versus 1536_cascade"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python --version

test "$(git -C "$TRELLIS_SOURCE" rev-parse HEAD)" = "$TRELLIS_SOURCE_REV"
check_ref "$HF_HUB_CACHE/models--microsoft--TRELLIS.2-4B" "$MODEL_REV"
check_ref "$HF_HUB_CACHE/models--microsoft--TRELLIS-image-large" "$TRELLIS_IMAGE_REV"
check_ref "$HF_HUB_CACHE/models--facebook--dinov3-vitl16-pretrain-lvd1689m" "$DINOV3_REV"
check_ref "$HF_HUB_CACHE/models--briaai--RMBG-2.0" "$RMBG_REV"

test -s "$MODEL_SNAPSHOT/pipeline.json"
test -s "$EXPERIMENT_SPEC"
test -s "$INPUT_IMAGE"
test -s "$BASELINE_DIR/metadata.json"
test -s "$BASELINE_DIR/asset.glb"
test -s "$PROJECT_ROOT/scripts/step3a_generate.py"
test -s "$TRELLIS_SOURCE/assets/hdri/forest.exr"
test ! -e "$OUTPUT_ROOT"
test "$(sha256sum "$INPUT_IMAGE" | cut -d ' ' -f 1)" = "$INPUT_SHA256"

python -m py_compile "$PROJECT_ROOT/scripts/step3a_generate.py"
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch CUDA is unavailable")
print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")
print(f"CUDA tensor probe: {torch.ones(1, device='cuda')}")
PY

cd "$TRELLIS_SOURCE"
python "$PROJECT_ROOT/scripts/step3a_generate.py" \
    --case-id img05_complex_shape \
    --input "$INPUT_IMAGE" \
    --output-dir "$OUTPUT_DIR" \
    --model-path "$MODEL_SNAPSHOT" \
    --trellis-source "$TRELLIS_SOURCE" \
    --envmap "$TRELLIS_SOURCE/assets/hdri/forest.exr" \
    --experiment-spec "$EXPERIMENT_SPEC" \
    --seed 42 \
    --pipeline-type 1536_cascade \
    --decimation-target 500000 \
    --texture-size 2048 \
    --render-resolution 768 \
    --render-views 4

cd "$PROJECT_ROOT"
python - "$BASELINE_DIR" "$OUTPUT_DIR" "$OUTPUT_ROOT" "$SLURM_JOB_ID" <<'PY'
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

baseline_dir = Path(sys.argv[1]).resolve()
highres_dir = Path(sys.argv[2]).resolve()
output_root = Path(sys.argv[3]).resolve()
job_id = sys.argv[4]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


baseline = read_json(baseline_dir / "metadata.json")
highres = read_json(highres_dir / "metadata.json")

assert baseline["status"] == "completed"
assert highres["status"] == "completed"
assert baseline["case_id"] == highres["case_id"] == "img05_complex_shape"
assert baseline["source_sha256"] == highres["source_sha256"]
assert baseline["seed"] == highres["seed"] == 42

baseline_parameters = dict(baseline["generation_parameters"])
highres_parameters = dict(highres["generation_parameters"])
assert baseline_parameters.pop("pipeline_type") == "1024_cascade"
assert highres_parameters.pop("pipeline_type") == "1536_cascade"
assert baseline_parameters == highres_parameters

assert highres["mesh"]["voxel_size"] < baseline["mesh"]["voxel_size"]
assert highres["peak_cuda_memory_bytes"] > 0
assert len(highres["output_files"]["rendered_views"]) == 4

artifact_names = [
    "asset.glb",
    "metadata.json",
    "preprocessed.png",
    "view_00.png",
    "view_01.png",
    "view_02.png",
    "view_03.png",
]
for name in artifact_names:
    assert (highres_dir / name).is_file() and (highres_dir / name).stat().st_size > 0


def metrics(metadata: dict) -> dict:
    timings = metadata["timings_seconds"]
    return {
        "pipeline_type": metadata["generation_parameters"]["pipeline_type"],
        "trellis_generation_seconds": timings["trellis_generation"],
        "glb_export_seconds": timings["glb_export"],
        "per_case_end_to_end_seconds": timings["per_case_end_to_end"],
        "peak_cuda_memory_bytes": metadata["peak_cuda_memory_bytes"],
        "mesh": metadata["mesh"],
    }


baseline_metrics = metrics(baseline)
highres_metrics = metrics(highres)
comparison = {
    "schema_version": 1,
    "stage": "Optional extension - higher-resolution generation",
    "status": "completed_pending_visual_qa",
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "slurm_job_id": job_id,
    "case_id": "img05_complex_shape",
    "controlled_variables": {
        "source_sha256": highres["source_sha256"],
        "seed": 42,
        "generation_parameters_except_pipeline_type": highres_parameters,
    },
    "baseline": baseline_metrics,
    "high_resolution": highres_metrics,
    "ratios_highres_over_baseline": {
        "trellis_generation_time": highres_metrics["trellis_generation_seconds"]
        / baseline_metrics["trellis_generation_seconds"],
        "peak_cuda_memory": highres_metrics["peak_cuda_memory_bytes"]
        / baseline_metrics["peak_cuda_memory_bytes"],
        "vertices": highres_metrics["mesh"]["vertices"]
        / baseline_metrics["mesh"]["vertices"],
        "faces": highres_metrics["mesh"]["faces"]
        / baseline_metrics["mesh"]["faces"],
        "voxel_size": highres_metrics["mesh"]["voxel_size"]
        / baseline_metrics["mesh"]["voxel_size"],
    },
    "high_resolution_artifacts": {
        name: {
            "size_bytes": (highres_dir / name).stat().st_size,
            "sha256": sha256(highres_dir / name),
        }
        for name in artifact_names
    },
}

manifest_path = output_root / "comparison_manifest.json"
manifest_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")

print(f"Baseline voxel size: {baseline_metrics['mesh']['voxel_size']}")
print(f"High-resolution voxel size: {highres_metrics['mesh']['voxel_size']}")
print(
    "Generation time ratio: "
    f"{comparison['ratios_highres_over_baseline']['trellis_generation_time']:.3f}x"
)
print(
    "Peak CUDA memory ratio: "
    f"{comparison['ratios_highres_over_baseline']['peak_cuda_memory']:.3f}x"
)
print(f"Comparison manifest: {manifest_path}")
print("OPTIONAL HIGH-RESOLUTION ARTIFACT VERIFICATION: PASSED")
PY

du -sh "$OUTPUT_ROOT"
echo "OPTIONAL HIGH-RESOLUTION JOB: PASSED"
echo "Finish: $(date --iso-8601=seconds)"
