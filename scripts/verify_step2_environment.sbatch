#!/bin/bash
#SBATCH --job-name=trellis2-verify
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/home/h/hongshan/cp4281-as2/logs/step2-final-%j.log

set -eo pipefail

PROJECT_ROOT="$HOME/cp4281-as2"
ENV_PREFIX="$PROJECT_ROOT/envs/trellis2"
TRELLIS_SOURCE="$PROJECT_ROOT/src/TRELLIS.2"
EXTENSION_ROOT="$PROJECT_ROOT/src/extensions"
VERSIONS_FILE="$PROJECT_ROOT/logs/step2-final-versions.txt"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$TRELLIS_SOURCE:${PYTHONPATH:-}"
export OPENCV_IO_ENABLE_OPENEXR=1

cd "$TRELLIS_SOURCE"

echo "Start: $(date --iso-8601=seconds)"
echo "Node: $(hostname)"

echo
echo "=== FULL TRELLIS.2 IMPORT VERIFICATION ==="
python - <<'PY'
import importlib
import torch

compiled_modules = [
    "flash_attn",
    "nvdiffrast.torch",
    "nvdiffrec_render.renderutils._C",
    "cumesh._C",
    "cumesh._cubvh",
    "cumesh._cumesh_xatlas",
    "o_voxel._C",
    "flex_gemm.kernels.cuda",
]

for module_name in compiled_modules:
    importlib.import_module(module_name)
    print(f"{module_name}: PASSED")

import trellis2
from trellis2.pipelines import (
    Trellis2ImageTo3DPipeline,
    Trellis2TexturingPipeline,
)
from trellis2.renderers import EnvMap
from trellis2.representations import Mesh
from trellis2.utils import render_utils

assert Trellis2ImageTo3DPipeline is not None
assert Trellis2TexturingPipeline is not None
assert EnvMap is not None
assert Mesh is not None
assert render_utils is not None
assert torch.cuda.is_available()

x = torch.randn(512, 512, device="cuda")
y = x @ x
torch.cuda.synchronize()
assert y.is_cuda and torch.isfinite(y).all().item()

print("trellis2 package:", trellis2.__file__)
print("TRELLIS.2 pipeline imports: PASSED")
print("CUDA tensor execution: PASSED")
print("GPU:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
PY

echo
echo "=== PACKAGE CONSISTENCY ==="
python -m pip check

echo
echo "=== FINAL VERSION MANIFEST ==="
{
    echo "Generated: $(date --iso-8601=seconds)"
    echo "Environment: $CONDA_PREFIX"
    echo "TRELLIS.2 source: $TRELLIS_SOURCE"
    echo
    python --version
    python -m pip --version
    nvcc --version
    cmake --version
    ninja --version
    echo
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo
    python - <<'PY'
from importlib.metadata import version

packages = [
    "torch",
    "torchvision",
    "flash-attn",
    "nvdiffrast",
    "nvdiffrec-render",
    "cumesh",
    "o-voxel",
    "flex-gemm",
    "transformers",
    "gradio",
    "trimesh",
]

for package in packages:
    print(f"{package}: {version(package)}")
PY
    echo
    echo "TRELLIS.2 commit: $(git -C "$TRELLIS_SOURCE" rev-parse HEAD)"
    echo "nvdiffrast commit: $(git -C "$EXTENSION_ROOT/nvdiffrast" rev-parse HEAD)"
    echo "nvdiffrec commit: $(git -C "$EXTENSION_ROOT/nvdiffrec" rev-parse HEAD)"
    echo "CuMesh commit: $(git -C "$EXTENSION_ROOT/CuMesh" rev-parse HEAD)"
    echo "FlexGEMM commit: $(git -C "$EXTENSION_ROOT/FlexGEMM" rev-parse HEAD)"
    echo "Eigen submodule: $(git -C "$TRELLIS_SOURCE" submodule status o-voxel/third_party/eigen)"
} | tee "$VERSIONS_FILE"

python -m pip freeze > "$PROJECT_ROOT/logs/step2-final-pip-freeze.txt"
conda list --explicit > "$PROJECT_ROOT/logs/step2-final-conda-explicit.txt"
conda list > "$PROJECT_ROOT/logs/step2-final-conda-list.txt"

echo
echo "Step 2 environment verification: PASSED"
echo "Version manifest: $VERSIONS_FILE"
echo "Finish: $(date --iso-8601=seconds)"
