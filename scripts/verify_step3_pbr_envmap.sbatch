#!/bin/bash
#SBATCH --job-name=trellis2-pbr-env-check
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=xgpg[0-7]
#SBATCH --gpus=a100-40
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/home/h/hongshan/cp4281-as2/logs/step3-pbr-env-check-%j.log
#SBATCH --error=/home/h/hongshan/cp4281-as2/logs/step3-pbr-env-check-%j.log

set -eo pipefail

PROJECT_ROOT="$HOME/cp4281-as2"
ENV_PREFIX="$PROJECT_ROOT/envs/trellis2"
TRELLIS_SOURCE="$PROJECT_ROOT/src/TRELLIS.2"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export PYTHONPATH="$PROJECT_ROOT:$TRELLIS_SOURCE:${PYTHONPATH:-}"
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "Start: $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
echo "Job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python --version

cd "$PROJECT_ROOT"
python - <<'PY'
import torch
import torch.nn.functional as F

from scripts.step3a_generate import build_neutral_studio_environment
from trellis2.renderers import EnvMap

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = torch.device("cuda")
torch.cuda.reset_peak_memory_stats(device)

environment = build_neutral_studio_environment(device)
assert environment.shape == (256, 512, 3), environment.shape
assert environment.dtype == torch.float32, environment.dtype
assert environment.is_contiguous()
assert torch.isfinite(environment).all()
assert environment.min().item() > 0

# EnvMap.sample() lazily converts the lat-long image to a cubemap, builds the
# nvdiffrec mip chain, and samples it through nvdiffrast.
envmap = EnvMap(environment)
torch.manual_seed(0)
directions = F.normalize(
    torch.randn((8, 16, 3), dtype=torch.float32, device=device), dim=-1
)
sampled = envmap.sample(directions)
torch.cuda.synchronize()

assert sampled.shape == directions.shape, (sampled.shape, directions.shape)
assert sampled.dtype == torch.float32, sampled.dtype
assert torch.isfinite(sampled).all()
assert sampled.max().item() > sampled.min().item()

print(
    "Environment:",
    tuple(environment.shape),
    environment.dtype,
    f"range=[{environment.min().item():.6f}, {environment.max().item():.6f}]",
)
print(
    "Cubemap samples:",
    tuple(sampled.shape),
    sampled.dtype,
    f"range=[{sampled.min().item():.6f}, {sampled.max().item():.6f}]",
)
print(
    "Peak CUDA allocation bytes:",
    torch.cuda.max_memory_allocated(device),
)
print("STEP 3 PBR ENVMAP VERIFICATION: PASSED")
PY

echo "Finish: $(date --iso-8601=seconds)"
