#!/bin/bash
#SBATCH --job-name=trellis2-step5-diag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=xgph10,xgpj0
#SBATCH --gpus=a100-80
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/home/h/hongshan/cp4281-as2/logs/step5-backend-diag-%j.log
#SBATCH --error=/home/h/hongshan/cp4281-as2/logs/step5-backend-diag-%j.log

set -eo pipefail

PROJECT_ROOT="$HOME/cp4281-as2"
ENV_PREFIX="$PROJECT_ROOT/envs/trellis2"
SCENE_SPEC="$PROJECT_ROOT/inputs/step5/scene_spec.json"
PERSISTENT_OUTPUT="$PROJECT_ROOT/outputs/step5/diagnostics/job-$SLURM_JOB_ID"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export PYGLET_HEADLESS=true
export PYOPENGL_PLATFORM=egl

LOCAL_JOB_ROOT=$(mktemp -d "/tmp/${USER}-trellis-step5-diag-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$LOCAL_JOB_ROOT"' EXIT
export TMPDIR="$LOCAL_JOB_ROOT/tmp"
export PYTHONPYCACHEPREFIX="$LOCAL_JOB_ROOT/pycache"
mkdir -p "$TMPDIR" "$PYTHONPYCACHEPREFIX" "$PROJECT_ROOT/logs"

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

test -f "$SCENE_SPEC"
test ! -e "$PERSISTENT_OUTPUT"

python -m py_compile "$PROJECT_ROOT/scripts/step5_backend_diag.py"
python -B "$PROJECT_ROOT/scripts/step5_backend_diag.py" \
    --self-test \
    --spec "$SCENE_SPEC" \
    --project-root "$PROJECT_ROOT"

python "$PROJECT_ROOT/scripts/step5_backend_diag.py" \
    --spec "$SCENE_SPEC" \
    --project-root "$PROJECT_ROOT" \
    --output-dir "$LOCAL_JOB_ROOT/output"

test -s "$LOCAL_JOB_ROOT/output/backend_report.json"
mkdir -p "$PERSISTENT_OUTPUT"
cp "$LOCAL_JOB_ROOT/output/backend_report.json" "$PERSISTENT_OUTPUT/backend_report.json"
if [ -s "$LOCAL_JOB_ROOT/output/trimesh_preview.png" ]; then
    cp "$LOCAL_JOB_ROOT/output/trimesh_preview.png" "$PERSISTENT_OUTPUT/trimesh_preview.png"
fi

python - "$PERSISTENT_OUTPUT/backend_report.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)

assert report["status"] == "diagnostic_complete", report
assert report["composition"]["asset_count"] == 6, report
assert report["composition"]["combined_geometry_count"] > 0, report
assert (
    report["composition"]["reload_geometry_count"]
    == report["composition"]["combined_geometry_count"]
), report
print("Composition GLB round trip: PASSED")
print("Package versions:", report["package_versions"])
print("System commands:", report["system_commands"])
print("Trimesh render probe:", report["trimesh_render_probe"])
PY

echo "Diagnostic output: $PERSISTENT_OUTPUT"
echo "STEP 5 BACKEND DIAGNOSTIC JOB: PASSED"
echo "Finish: $(date --iso-8601=seconds)"
