#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/h/hongshan/cp4281-as2"
export HF_HOME="$PROJECT_ROOT/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"

check_repo() {
    repo_id="$1"
    cache_name="$2"
    repo_dir="$HF_HUB_CACHE/$cache_name"

    if [[ ! -f "$repo_dir/refs/main" ]]; then
        echo "FAILED: $repo_id has no cached main revision"
        return 1
    fi

    revision="$(cat "$repo_dir/refs/main")"
    snapshot="$repo_dir/snapshots/$revision"

    if [[ ! -d "$snapshot" ]]; then
        echo "FAILED: snapshot directory missing for $repo_id"
        return 1
    fi

    size="$(du -sh "$repo_dir" | cut -f1)"
    weight_count="$(find -L "$snapshot" -type f -name "*.safetensors" | wc -l)"

    echo
    echo "Repository: $repo_id"
    echo "Revision:   $revision"
    echo "Cache size: $size"
    echo "Snapshot:   $snapshot"
    echo "Safetensors files: $weight_count"

    find -L "$snapshot" \
        -type f -name "*.safetensors" \
        -printf "  %P\n" | sort

    if [[ "$weight_count" -eq 0 ]]; then
        echo "FAILED: no safetensors found for $repo_id"
        return 1
    fi
}

require_file() {
    path="$1"

    if [[ ! -e "$path" ]]; then
        echo "FAILED: required file missing: $path"
        return 1
    fi

    echo "Verified: $path"
}

check_repo \
    "microsoft/TRELLIS.2-4B" \
    "models--microsoft--TRELLIS.2-4B"

check_repo \
    "microsoft/TRELLIS-image-large" \
    "models--microsoft--TRELLIS-image-large"

check_repo \
    "facebook/dinov3-vitl16-pretrain-lvd1689m" \
    "models--facebook--dinov3-vitl16-pretrain-lvd1689m"

check_repo \
    "briaai/RMBG-2.0" \
    "models--briaai--RMBG-2.0"

check_repo \
    "black-forest-labs/FLUX.1-schnell" \
    "models--black-forest-labs--FLUX.1-schnell"

TRELLIS_IMAGE_DIR="$HF_HUB_CACHE/models--microsoft--TRELLIS-image-large"
TRELLIS_IMAGE_REV="$(cat "$TRELLIS_IMAGE_DIR/refs/main")"
TRELLIS_IMAGE_SNAPSHOT="$TRELLIS_IMAGE_DIR/snapshots/$TRELLIS_IMAGE_REV"

require_file \
    "$TRELLIS_IMAGE_SNAPSHOT/ckpts/ss_dec_conv3d_16l8_fp16.json"

require_file \
    "$TRELLIS_IMAGE_SNAPSHOT/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"

FLUX_DIR="$HF_HUB_CACHE/models--black-forest-labs--FLUX.1-schnell"
FLUX_REV="$(cat "$FLUX_DIR/refs/main")"
FLUX_SNAPSHOT="$FLUX_DIR/snapshots/$FLUX_REV"

require_file "$FLUX_SNAPSHOT/model_index.json"

if [[ -e "$FLUX_SNAPSHOT/flux1-schnell.safetensors" ]]; then
    echo "WARNING: duplicate FLUX single-file checkpoint was downloaded"
else
    echo "Verified: duplicate FLUX single-file checkpoint was excluded"
fi

incomplete="$(find "$HF_HOME" -type f -name "*.incomplete" -print -quit)"

if [[ -n "$incomplete" ]]; then
    echo "FAILED: incomplete download found: $incomplete"
    exit 1
fi

echo
echo "Total Hugging Face cache:"
du -sh "$HF_HOME"

echo
echo "MODEL VERIFICATION PASSED"
