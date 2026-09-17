#!/bin/bash
#SBATCH --job-name=trellis2-step5-formal
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=xgph10,xgpj0
#SBATCH --gpus=a100-80
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/h/hongshan/cp4281-as2/logs/step5-formal-%j.log
#SBATCH --error=/home/h/hongshan/cp4281-as2/logs/step5-formal-%j.log

set -eo pipefail

PROJECT_ROOT="$HOME/cp4281-as2"
ENVIRONMENT="$PROJECT_ROOT/envs/trellis2"
SCENE_SPEC="$PROJECT_ROOT/inputs/step5/scene_spec.json"
OUTPUT_DIR="$PROJECT_ROOT/outputs/step5/formal/job-$SLURM_JOB_ID"
JOB_TMP="$PROJECT_ROOT/tmp/step5-formal-$SLURM_JOB_ID"

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/outputs/step5/formal" "$JOB_TMP"
trap 'rm -rf -- "$JOB_TMP"' EXIT

export TMPDIR="$JOB_TMP"
export XDG_CACHE_HOME="$JOB_TMP/cache"
export TORCH_HOME="$PROJECT_ROOT/cache/torch"
export HF_HOME="$PROJECT_ROOT/cache/huggingface"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENVIRONMENT"

echo "Start: $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
echo "Job: $SLURM_JOB_ID"
echo "Output: $OUTPUT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python --version

if [ -e "$OUTPUT_DIR" ]; then
  echo "ERROR: refusing to overwrite $OUTPUT_DIR" >&2
  exit 1
fi

test -s "$SCENE_SPEC"
test -s "$PROJECT_ROOT/scripts/step5_backend_diag.py"
test -s "$PROJECT_ROOT/scripts/step5_compose.py"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch CUDA is unavailable")
print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")
print(f"CUDA tensor probe: {torch.ones(1, device='cuda')}")
PY

python -m py_compile \
  "$PROJECT_ROOT/scripts/step5_backend_diag.py" \
  "$PROJECT_ROOT/scripts/step5_compose.py"

python -B "$PROJECT_ROOT/scripts/step5_backend_diag.py" \
  --self-test \
  --spec "$SCENE_SPEC" \
  --project-root "$PROJECT_ROOT"

python -B "$PROJECT_ROOT/scripts/step5_compose.py" \
  --self-test \
  --spec "$SCENE_SPEC" \
  --project-root "$PROJECT_ROOT"

python -B "$PROJECT_ROOT/scripts/step5_compose.py" \
  --spec "$SCENE_SPEC" \
  --project-root "$PROJECT_ROOT" \
  --output-dir "$OUTPUT_DIR"

test -s "$OUTPUT_DIR/scene.glb"
test -s "$OUTPUT_DIR/hero_view.png"
test -s "$OUTPUT_DIR/scene_manifest.json"
test -s "$OUTPUT_DIR/scene_spec.json"

python - "$OUTPUT_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
manifest = json.loads((output_dir / "scene_manifest.json").read_text(encoding="utf-8"))

assert manifest["status"] == "completed"
assert manifest["generated_asset_count"] == 6
assert len(manifest["assets"]) == 6
assert manifest["scene"]["geometry_count"] >= 15
assert manifest["scene"]["reload_geometry_count"] == manifest["scene"]["geometry_count"]

acceptance = manifest["acceptance"]
assert acceptance["minimum_generated_asset_count"] == 5
assert acceptance["generated_asset_count_passed"] is True
assert acceptance["all_source_hashes_verified"] is True
assert acceptance["all_support_contacts_verified"] is True
assert acceptance["single_scene_file_created"] is True
assert acceptance["rendered_view_count"] >= 1

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

assert sha256(output_dir / "scene.glb") == manifest["scene"]["sha256"]
assert sha256(output_dir / "hero_view.png") == manifest["render"]["sha256"]
assert manifest["render"]["resolution"] == [1280, 720]
assert manifest["render"]["face_count"] > 0

print("Generated asset count: 6")
print(f"Scene geometry count: {manifest['scene']['geometry_count']}")
print(f"Rendered faces: {manifest['render']['face_count']}")
print(f"Scene SHA256: {manifest['scene']['sha256']}")
print(f"Render SHA256: {manifest['render']['sha256']}")
print("STEP 5 FORMAL ARTIFACT VERIFICATION: PASSED")
PY

du -sh "$OUTPUT_DIR"
echo "STEP 5 FORMAL SCENE JOB: PASSED"
echo "Finish: $(date --iso-8601=seconds)"
