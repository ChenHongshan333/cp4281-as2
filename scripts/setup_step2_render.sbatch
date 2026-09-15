#!/bin/bash
#SBATCH --job-name=trellis2-render
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/home/h/hongshan/cp4281-as2/logs/step2-render-%j.log

set -eo pipefail

PROJECT_ROOT="$HOME/cp4281-as2"
ENV_PREFIX="$PROJECT_ROOT/envs/trellis2"
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
export MAX_JOBS=4

LOCAL_BUILD_ROOT=$(mktemp -d "/tmp/${USER}-trellis-render-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$LOCAL_BUILD_ROOT"' EXIT
export TMPDIR="$LOCAL_BUILD_ROOT/tmp"
export PIP_CACHE_DIR="$LOCAL_BUILD_ROOT/pip-cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$EXTENSION_ROOT"

echo "Start: $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
echo "Local build root: $LOCAL_BUILD_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python --version
nvcc --version

if [ ! -d "$EXTENSION_ROOT/nvdiffrast/.git" ]; then
    if [ -e "$EXTENSION_ROOT/nvdiffrast" ]; then
        echo "ERROR: nvdiffrast exists but is not a Git repository"
        exit 1
    fi
    git clone --branch v0.4.0 --depth 1 \
        https://github.com/NVlabs/nvdiffrast.git \
        "$EXTENSION_ROOT/nvdiffrast"
fi

if [ ! -d "$EXTENSION_ROOT/nvdiffrec/.git" ]; then
    if [ -e "$EXTENSION_ROOT/nvdiffrec" ]; then
        echo "ERROR: nvdiffrec exists but is not a Git repository"
        exit 1
    fi
    git clone --branch renderutils --depth 1 \
        https://github.com/JeffreyXiang/nvdiffrec.git \
        "$EXTENSION_ROOT/nvdiffrec"
fi

echo
echo "=== SOURCE REVISIONS ==="
echo "nvdiffrast: $(git -C "$EXTENSION_ROOT/nvdiffrast" rev-parse HEAD)"
echo "nvdiffrec:  $(git -C "$EXTENSION_ROOT/nvdiffrec" rev-parse HEAD)"

mkdir -p "$LOCAL_BUILD_ROOT/src/nvdiffrast" "$LOCAL_BUILD_ROOT/src/nvdiffrec"
cp -r "$EXTENSION_ROOT/nvdiffrast/." "$LOCAL_BUILD_ROOT/src/nvdiffrast/"
cp -r "$EXTENSION_ROOT/nvdiffrec/." "$LOCAL_BUILD_ROOT/src/nvdiffrec/"

echo
echo "=== INSTALL NVDIFFRAST ==="
python -m pip install "$LOCAL_BUILD_ROOT/src/nvdiffrast" --no-build-isolation

echo
echo "=== INSTALL NVDIFFREC RENDERUTILS ==="
python -m pip install "$LOCAL_BUILD_ROOT/src/nvdiffrec" --no-build-isolation

echo
echo "=== RENDERING EXTENSION VERIFICATION ==="
python - <<'PY'
import torch
import nvdiffrast.torch as dr
from nvdiffrec_render.renderutils import _C

assert torch.cuda.is_available()

ctx = dr.RasterizeCudaContext()
positions = torch.tensor(
    [[[-0.8, -0.8, 0.0, 1.0],
      [ 0.8, -0.8, 0.0, 1.0],
      [ 0.0,  0.8, 0.0, 1.0]]],
    dtype=torch.float32,
    device="cuda",
)
triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
raster, _ = dr.rasterize(ctx, positions, triangles, resolution=[64, 64])

assert raster.is_cuda
assert (raster[..., 3] > 0).any().item()
assert _C is not None

print("torch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("nvdiffrast CUDA rasterization: PASSED")
print("nvdiffrec_render CUDA extension import: PASSED")
PY

python -m pip show nvdiffrast nvdiffrec-render || true
python -m pip check
python -m pip freeze > "$PROJECT_ROOT/logs/step2-render-pip-freeze.txt"

echo "Finish: $(date --iso-8601=seconds)"
