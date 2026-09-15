#!/usr/bin/env bash
#SBATCH --job-name=cp4281-models
#SBATCH --partition=long
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/h/hongshan/cp4281-as2/logs/download-models-%j.log
#SBATCH --error=/home/h/hongshan/cp4281-as2/logs/download-models-%j.log

set -euo pipefail

PROJECT_ROOT="/home/h/hongshan/cp4281-as2"

export HF_HOME="$PROJECT_ROOT/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TMPDIR="$PROJECT_ROOT/tmp"

source "$PROJECT_ROOT/tools/hf-download-env/bin/activate"

echo "Start: $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
python --version
hf --version
hf auth whoami

echo "Downloading TRELLIS.2-4B"
hf download microsoft/TRELLIS.2-4B

echo "Downloading sparse-structure decoder"
hf download microsoft/TRELLIS-image-large \
  --include "ckpts/ss_dec_conv3d_16l8_fp16.*"

echo "Downloading DINOv3 conditioner"
hf download facebook/dinov3-vitl16-pretrain-lvd1689m

echo "Downloading RMBG-2.0"
hf download briaai/RMBG-2.0

echo "Downloading FLUX.1-schnell Diffusers pipeline"
hf download black-forest-labs/FLUX.1-schnell \
  --exclude "flux1-schnell.safetensors"

echo "Downloaded cache size:"
du -sh "$HF_HOME"

echo "Incomplete files, if any:"
find "$HF_HOME" -type f -name "*.incomplete" -print

echo "Finish: $(date --iso-8601=seconds)"
