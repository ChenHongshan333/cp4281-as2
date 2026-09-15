# TRELLIS.2 Conditional 3D Generation

## Overview

This project deploys `microsoft/TRELLIS.2-4B` on an NVIDIA GPU and uses it to generate 3D assets from image and text inputs. Since TRELLIS.2 is image-conditioned, text-to-3D generation is implemented as a text-to-image stage followed by the standard image-to-3D pipeline.

The project also investigates the model's two-stage flow-matching process by extracting, decoding, and rendering intermediate sampler states. Finally, generated assets are assembled into a composed 3D scene.

Current progress: **Step 1 complete — awaiting approval to start Step 2**

## Table of Contents

- [Step 1 — Download Model Weights](#step-1--download-model-weights)
- [Step 2 — Build the Environment](#step-2--build-the-environment)
- [Step 3 — Deploy and Generate](#step-3--deploy-and-generate)
- [Step 4 — Visualise the Diffusion Process](#step-4--visualise-the-diffusion-process)
- [Step 5 — Composite a Scene](#step-5--composite-a-scene)
- [Submission Checklist](#submission-checklist)

## Step 1 — Download Model Weights

### Objective

Obtain all model checkpoints required by the image-to-3D and text-to-3D pipelines.

Required models:

| Model | Purpose |
|---|---|
| `microsoft/TRELLIS.2-4B` | Main two-stage 3D generation model; mandatory |
| `microsoft/TRELLIS-image-large` | Sparse-structure decoder checkpoint: `ckpts/ss_dec_conv3d_16l8_fp16.*` |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | Image-condition encoder |
| `briaai/RMBG-2.0` or BiRefNet | Input-image background removal |
| `black-forest-labs/FLUX.1-schnell` or an equivalent model | Converts text prompts into single-object images for text-to-3D generation |

### Implementation

The weights were downloaded on the NUS SoC Compute Cluster. A CPU job in the Slurm `long` partition performed the download because this stage required network and storage resources rather than a GPU. A read-only Hugging Face token was used after access had been granted to the gated DINOv3, RMBG-2.0, and FLUX repositories.

The download tools used Python 3.13.13, Hugging Face CLI 1.31.0, and `hf_xet`. The cache was placed in persistent shared storage:

```text
/home/h/hongshan/cp4281-as2/hf-cache
```

The main download commands were:

```bash
hf download microsoft/TRELLIS.2-4B
hf download microsoft/TRELLIS-image-large \
  --include "ckpts/ss_dec_conv3d_16l8_fp16.*"
hf download facebook/dinov3-vitl16-pretrain-lvd1689m
hf download briaai/RMBG-2.0
hf download black-forest-labs/FLUX.1-schnell \
  --exclude "flux1-schnell.safetensors"
```

The standalone `flux1-schnell.safetensors` file was excluded because the required Diffusers-format transformer, text encoders, and VAE were downloaded from the same repository. This avoided storing a duplicate 23.8 GB FLUX checkpoint.

### Results

Step 1 was completed successfully. Slurm job `849112` finished with `COMPLETED`, exit code `0:0`, in 13 minutes 29 seconds. The resulting Hugging Face cache occupies 53 GB, and no `.incomplete` files were found.

| Model | Revision | Cache size | Verification |
|---|---|---:|---|
| `microsoft/TRELLIS.2-4B` | `af44b45f2e35a493886929c6d786e563ec68364d` | 16 GB | 9 safetensors files found |
| `microsoft/TRELLIS-image-large` | `25e0d31ffbebe4b5a97464dd851910efc3002d96` | 141 MB | Required JSON and safetensors decoder files found |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | `ea8dc2863c51be0a264bab82070e3e8836b02d51` | 1.2 GB | `model.safetensors` found |
| `briaai/RMBG-2.0` | `5df4c9c76d8170882c34f6986e848ee07fd0ba43` | 3.9 GB | `model.safetensors` found |
| `black-forest-labs/FLUX.1-schnell` | `741f7c3ce8b383c54771c7003378a50191e9efe9` | 32 GB | 8 Diffusers safetensors files and `model_index.json` found |

The full verification manifest is stored on the cluster at:

```text
/home/h/hongshan/cp4281-as2/logs/model-manifest.txt
```

All weights were obtained from their official Hugging Face repositories:

- [`microsoft/TRELLIS.2-4B`](https://huggingface.co/microsoft/TRELLIS.2-4B)
- [`microsoft/TRELLIS-image-large`](https://huggingface.co/microsoft/TRELLIS-image-large)
- [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
- [`briaai/RMBG-2.0`](https://huggingface.co/briaai/RMBG-2.0)
- [`black-forest-labs/FLUX.1-schnell`](https://huggingface.co/black-forest-labs/FLUX.1-schnell)

No substitute 3D model was used. RMBG-2.0 and FLUX.1-schnell were selected from the implementations suggested by the assignment.

## Step 2 — Build the Environment

Not started. This step will create an isolated Python environment and install PyTorch, the CUDA toolchain, and all required compiled CUDA extensions.

## Step 3 — Deploy and Generate

Not started. This step will produce five image-conditioned assets and five text-conditioned assets, with two rendered viewpoints and generation timing for each asset.

## Step 4 — Visualise the Diffusion Process

Not started. This step will capture and decode five intermediate sampler states from each of TRELLIS.2's two flow-matching stages for three image-conditioned cases.

## Step 5 — Composite a Scene

Not started. This step will arrange at least five generated assets into one intentionally composed 3D scene.

## Submission Checklist

### Report

- [x] Step 1: downloaded models, official sources, revisions, storage location, and substitution statement
- [ ] Step 2: ordered copy-pasteable environment commands and exact versions of Python, PyTorch, CUDA, and every compiled extension
- [ ] Step 3a: five original input images, selection rationale, five `.glb` assets, at least two rendered viewpoints per asset, and individual generation times
- [ ] Step 3b: five original prompts, the prompt template, intermediate text-to-image outputs, five `.glb` assets, at least two rendered viewpoints per asset, and individual generation times
- [ ] Step 3b: explain that TRELLIS.2 never receives text directly and include at least one case where the text-to-image stage limits the final result
- [ ] Step 4: select three Step 3a cases and show the input and final rendered asset for each
- [ ] Step 4: two labelled strips per case, with five Stage 1 and five Stage 2 renders each, for 30 intermediate renders in total
- [ ] Step 4: label every frame with its actual stage, true timestep, and a measurement
- [ ] Step 4: sample Stage 1 across `t = 0.5` to `0.0`, Stage 2 across `t = 1.0` to `0.0`, and report the increased Stage 1 Euler step count
- [ ] Step 4: explain both sampling locations, captured `x_t`, timestep schedules, and the separate decoding method used for each latent space
- [ ] Step 5: describe the scene concept, assembly tool, asset selection, scale, orientation, placement, and surface contact
- [ ] Step 5: include at least one rendered scene view
- [ ] Optional: document either higher-resolution generation or a production-quality creative scene, if attempted

### Files

- [ ] One PDF report containing all report items above
- [ ] Ten generated `.glb` assets: five image-conditioned and five text-conditioned
- [ ] One composed scene containing at least five generated assets (`.glb` preferred; `.blend` or `.usd` accepted)

### Oral Preparation

- [ ] Be able to explain the complete Step 3 conditioning chain
- [ ] Be able to explain both Step 4 flow-matching stages, sampler states, schedules, and decoders
- [ ] Be able to explain the Step 5 scene assembly and placement decisions
