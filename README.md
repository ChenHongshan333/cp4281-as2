# TRELLIS.2 Conditional 3D Generation

## Overview

This project deploys `microsoft/TRELLIS.2-4B` on an NVIDIA GPU and uses it to generate 3D assets from image and text inputs. Since TRELLIS.2 is image-conditioned, text-to-3D generation is implemented as a text-to-image stage followed by the standard image-to-3D pipeline.

The project also investigates the model's two-stage flow-matching process by extracting, decoding, and rendering intermediate sampler states. Finally, generated assets are assembled into a composed 3D scene.

Current progress: **Step 1 — Download model weights**

## Table of Contents

- [Step 1 — Download Model Weights](#step-1--download-model-weights)
- [Step 2 — Build the Environment](#step-2--build-the-environment)
- [Step 3 — Deploy and Generate](#step-3--deploy-and-generate)
- [Step 4 — Visualise the Diffusion Process](#step-4--visualise-the-diffusion-process)
- [Step 5 — Composite a Scene](#step-5--composite-a-scene)
- [Final Deliverables](#final-deliverables)

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

1. Select a Linux server with an NVIDIA GPU and sufficient storage.
2. Obtain access to any gated Hugging Face repositories.
3. Download the checkpoints to a persistent directory on the target server.
4. Download only the required sparse-structure decoder files from `TRELLIS-image-large`.
5. Record the source repository, revision, storage path, and size of each downloaded model.
6. Verify that all checkpoint files are complete and readable.

### Results

In progress. The target GPU server and model storage location have not yet been confirmed, so no model checkpoints have been downloaded.

## Step 2 — Build the Environment

Not started. This step will create an isolated Python environment and install PyTorch, the CUDA toolchain, and all required compiled CUDA extensions.

## Step 3 — Deploy and Generate

Not started. This step will produce five image-conditioned assets and five text-conditioned assets, with two rendered viewpoints and generation timing for each asset.

## Step 4 — Visualise the Diffusion Process

Not started. This step will capture and decode five intermediate sampler states from each of TRELLIS.2's two flow-matching stages for three image-conditioned cases.

## Step 5 — Composite a Scene

Not started. This step will arrange at least five generated assets into one intentionally composed 3D scene.

## Final Deliverables

- One PDF report
- Ten generated `.glb` assets
- One composed scene file (`.glb` preferred; `.blend` or `.usd` accepted)
