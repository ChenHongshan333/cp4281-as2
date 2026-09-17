#!/bin/bash
#SBATCH --job-name=trellis2-cuda-diag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=a100-40
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/home/h/hongshan/cp4281-as2/logs/step4-cuda-diag-%j.log
#SBATCH --error=/home/h/hongshan/cp4281-as2/logs/step4-cuda-diag-%j.log

set -eo pipefail

PROJECT_ROOT="$HOME/cp4281-as2"
ENV_PREFIX="$PROJECT_ROOT/envs/trellis2"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

echo "Start: $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
echo "Job: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>}"
echo "SLURM_GPUS=${SLURM_GPUS:-<unset>}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

type -a python
python --version
nvidia-smi -L || true
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader || true
ls -l /dev/nvidia* 2>&1 || true
ldconfig -p 2>/dev/null | grep -E 'libcuda\.so|libnvidia-ml\.so' || true

python - <<'PY'
import ctypes
import os
import traceback
import warnings

print("PYTHON CUDA DIAGNOSTICS")
for name in (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_GPUS",
    "LD_LIBRARY_PATH",
):
    print(f"{name}={os.environ.get(name, '<unset>')}")

try:
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    result = cuda.cuInit(0)
    print(f"libcuda.so.1 cuInit(0) result={result}")
    if result == 0:
        version = ctypes.c_int()
        version_result = cuda.cuDriverGetVersion(ctypes.byref(version))
        print(
            "libcuda.so.1 cuDriverGetVersion "
            f"result={version_result} version={version.value}"
        )
except Exception:
    print("libcuda.so.1 probe raised an exception:")
    traceback.print_exc()

try:
    import torch

    print(f"torch.__version__={torch.__version__}")
    print(f"torch.__file__={torch.__file__}")
    print(f"torch.version.cuda={torch.version.cuda}")
    print(f"torch.backends.cuda.is_built()={torch.backends.cuda.is_built()}")
    compiled_version = getattr(torch._C, "_cuda_getCompiledVersion", None)
    if compiled_version is not None:
        print(f"torch compiled CUDA version={compiled_version()}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        available = torch.cuda.is_available()
        count = torch.cuda.device_count()
    print(f"torch.cuda.is_available()={available}")
    print(f"torch.cuda.device_count()={count}")
    for warning in caught:
        print(f"PyTorch warning: {warning.message}")

    try:
        torch.cuda.init()
        probe = torch.ones(1, device="cuda")
        torch.cuda.synchronize()
        print(f"torch CUDA device={torch.cuda.get_device_name(0)}")
        print(f"torch CUDA tensor probe={probe}")
    except Exception:
        print("PyTorch CUDA initialization raised an exception:")
        traceback.print_exc()
except Exception:
    print("PyTorch import/diagnostics raised an exception:")
    traceback.print_exc()

print("STEP 4 CUDA DIAGNOSTICS: COMPLETE")
PY

echo "End: $(date --iso-8601=seconds)"
