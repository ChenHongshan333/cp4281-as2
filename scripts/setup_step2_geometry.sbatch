#!/bin/bash
#SBATCH --job-name=trellis2-geometry
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/h/hongshan/cp4281-as2/logs/step2-geometry-%j.log

set -eo pipefail

PROJECT_ROOT="$HOME/cp4281-as2"
ENV_PREFIX="$PROJECT_ROOT/envs/trellis2"
TRELLIS_SOURCE="$PROJECT_ROOT/src/TRELLIS.2"
EXTENSION_ROOT="$PROJECT_ROOT/src/extensions"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib/stubs:${LIBRARY_PATH:-}"
export LDFLAGS="-L$CUDA_HOME/lib/stubs ${LDFLAGS:-}"
export TORCH_CUDA_ARCH_LIST="8.0"
export BUILD_TARGET="cuda"
export MAX_JOBS=4

LOCAL_BUILD_ROOT=$(mktemp -d "/tmp/${USER}-trellis-geometry-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$LOCAL_BUILD_ROOT"' EXIT
export TMPDIR="$LOCAL_BUILD_ROOT/tmp"
export PIP_CACHE_DIR="$LOCAL_BUILD_ROOT/pip-cache"
export TRITON_CACHE_DIR="$LOCAL_BUILD_ROOT/triton-cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" "$EXTENSION_ROOT"

echo "Start: $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
echo "Local build root: $LOCAL_BUILD_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python --version
nvcc --version

if [ ! -d "$EXTENSION_ROOT/CuMesh/.git" ]; then
    if [ -e "$EXTENSION_ROOT/CuMesh" ]; then
        echo "ERROR: CuMesh exists but is not a Git repository"
        exit 1
    fi
    git clone --recursive https://github.com/JeffreyXiang/CuMesh.git \
        "$EXTENSION_ROOT/CuMesh"
else
    git -C "$EXTENSION_ROOT/CuMesh" submodule update --init --recursive
fi

if [ ! -d "$EXTENSION_ROOT/FlexGEMM/.git" ]; then
    if [ -e "$EXTENSION_ROOT/FlexGEMM" ]; then
        echo "ERROR: FlexGEMM exists but is not a Git repository"
        exit 1
    fi
    git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git \
        "$EXTENSION_ROOT/FlexGEMM"
else
    git -C "$EXTENSION_ROOT/FlexGEMM" submodule update --init --recursive
fi

test -d "$TRELLIS_SOURCE/o-voxel"
test -f "$TRELLIS_SOURCE/o-voxel/third_party/eigen/Eigen/Core"

echo
echo "=== SOURCE REVISIONS ==="
echo "TRELLIS.2: $(git -C "$TRELLIS_SOURCE" rev-parse HEAD)"
echo "o-voxel Eigen: $(git -C "$TRELLIS_SOURCE" submodule status o-voxel/third_party/eigen)"
echo "CuMesh: $(git -C "$EXTENSION_ROOT/CuMesh" rev-parse HEAD)"
echo "FlexGEMM: $(git -C "$EXTENSION_ROOT/FlexGEMM" rev-parse HEAD)"

mkdir -p \
    "$LOCAL_BUILD_ROOT/src/CuMesh" \
    "$LOCAL_BUILD_ROOT/src/o-voxel" \
    "$LOCAL_BUILD_ROOT/src/FlexGEMM"
cp -r "$EXTENSION_ROOT/CuMesh/." "$LOCAL_BUILD_ROOT/src/CuMesh/"
cp -r "$TRELLIS_SOURCE/o-voxel/." "$LOCAL_BUILD_ROOT/src/o-voxel/"
cp -r "$EXTENSION_ROOT/FlexGEMM/." "$LOCAL_BUILD_ROOT/src/FlexGEMM/"

echo
echo "=== INSTALL CUMESH ==="
python -m pip install "$LOCAL_BUILD_ROOT/src/CuMesh" --no-build-isolation

echo
echo "=== INSTALL O-VOXEL ==="
python -m pip install "$LOCAL_BUILD_ROOT/src/o-voxel" --no-build-isolation

echo
echo "=== INSTALL FLEXGEMM ==="
python -m pip install "$LOCAL_BUILD_ROOT/src/FlexGEMM" --no-build-isolation

echo
echo "=== GEOMETRY EXTENSION VERIFICATION ==="
python - <<'PY'
import importlib
import torch

assert torch.cuda.is_available()

compiled_modules = [
    "cumesh._C",
    "cumesh._cubvh",
    "cumesh._cumesh_xatlas",
    "o_voxel._C",
    "flex_gemm.kernels.cuda",
]

for module_name in compiled_modules:
    module = importlib.import_module(module_name)
    assert module is not None
    print(f"{module_name}: PASSED")

import cumesh
import o_voxel
import flex_gemm

assert cumesh is not None
assert o_voxel is not None
assert flex_gemm is not None

print("torch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("All geometry extension imports: PASSED")
PY

python -m pip show cumesh o-voxel flex-gemm || true
python -m pip check
python -m pip freeze > "$PROJECT_ROOT/logs/step2-geometry-pip-freeze.txt"

echo "Finish: $(date --iso-8601=seconds)"
