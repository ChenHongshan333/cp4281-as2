# TRELLIS.2 Conditional 3D Generation

## Overview

This project deploys `microsoft/TRELLIS.2-4B` on an NVIDIA GPU and uses it to generate 3D assets from image and text inputs. Since TRELLIS.2 is image-conditioned, text-to-3D generation is implemented as a text-to-image stage followed by the standard image-to-3D pipeline.

The project also investigates the model's two-stage flow-matching process by extracting, decoding, and rendering intermediate sampler states. Finally, generated assets are assembled into a composed 3D scene.

Current progress: **Steps 1–2 and Step 3a complete — awaiting approval to start Step 3b**

## Table of Contents

- [Step 1 — Download Model Weights](#step-1--download-model-weights)
- [Step 2 — Build the Environment](#step-2--build-the-environment)
- [Step 3 — Deploy and Generate](#step-3--deploy-and-generate)
  - [Step 3a — Image-Conditioned Generation](#step-3a--image-conditioned-generation)
  - [Step 3b — Text-Conditioned Generation](#step-3b--text-conditioned-generation)
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

The complete reproducible Slurm download job and verification procedure are preserved in [`scripts/download_models.sbatch`](scripts/download_models.sbatch) and [`scripts/verify_models.sh`](scripts/verify_models.sh).

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

### Cluster Artifacts

| Cluster path | Purpose | Keep for submission? |
|---|---|---|
| `~/cp4281-as2/download_models.sbatch` | Slurm job that downloaded all required checkpoints | Keep as reproducibility evidence; not a submitted file |
| `~/cp4281-as2/verify_models.sh` | Verifies revisions, required files, weight counts, and incomplete downloads | Keep as reproducibility evidence; not a submitted file |
| `~/cp4281-as2/hf-cache/` | The 53 GB model cache used by later inference jobs | Keep on the cluster; do not submit |
| `~/cp4281-as2/logs/download-models-849112.log` | Download timing, node, versions, and completion log | Keep as evidence; do not submit |
| `~/cp4281-as2/logs/model-manifest.txt` | Verified revisions, sizes, and checkpoint inventory | Keep for the report; do not submit separately |
| `~/cp4281-as2/access-check/` | Small files used only to test gated-repository access | Temporary; not needed for submission |
| `~/cp4281-as2/tools/hf-download-env/` | Lightweight Hugging Face download environment | Re-creatable; not the Step 2 runtime environment |
| `~/cp4281-as2/tmp/` | Temporary download workspace | Not needed for submission after successful verification |

## Step 2 — Build the Environment

### Objective

Create an isolated, reproducible runtime for TRELLIS.2 with its required CUDA toolchain and source-built native extensions. No package was installed into the cluster's system Python.

### Implementation

The environment was created at:

```text
/home/h/hongshan/cp4281-as2/envs/trellis2
```

Miniconda was loaded explicitly because non-interactive Slurm shells do not automatically source the user's `.bashrc`:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$HOME/cp4281-as2/envs/trellis2"
```

The installation was performed in the following order:

1. Create a Python 3.10 environment and install the CUDA 12.4 compiler toolkit, GCC/G++ 12, CMake, and Ninja.
2. Install PyTorch 2.6.0 and torchvision 0.21.0 using their CUDA 12.4 builds.
3. Install the standard Python dependencies required by TRELLIS.2.
4. Clone TRELLIS.2 recursively and retain its exact Git revision.
5. Compile and test FlashAttention.
6. Compile and test `nvdiffrast` and `nvdiffrec_render`.
7. Compile and load-test CuMesh, O-Voxel, and FlexGEMM.
8. Import the complete TRELLIS.2 pipeline and export final pip, conda, version, and source-revision manifests.

Compilation ran in Slurm GPU jobs against an NVIDIA A100 80 GB PCIe MIG 3g.40gb instance. Temporary build trees and pip caches were placed together under `/tmp` to avoid cross-filesystem wheel operations. CUDA extensions were built for compute capability 8.0. The CUDA driver stub directory was supplied only to the linker for extensions that link against `libcuda.so`.

The reproducible scripts retained in this repository are:

| Script | Purpose |
|---|---|
| [`scripts/setup_step2_render.sbatch`](scripts/setup_step2_render.sbatch) | Build and functionally test `nvdiffrast` and `nvdiffrec_render` |
| [`scripts/setup_step2_geometry.sbatch`](scripts/setup_step2_geometry.sbatch) | Build and load-test CuMesh, O-Voxel, and FlexGEMM |
| [`scripts/verify_step2_environment.sbatch`](scripts/verify_step2_environment.sbatch) | Perform the final TRELLIS.2 import, CUDA execution, dependency, version, and revision checks |

### Results

The isolated environment passed the complete TRELLIS.2 pipeline import, all compiled-extension imports, a CUDA tensor execution test, and `pip check`.

| Component | Installed version or build identity |
|---|---|
| Python | 3.10.21 |
| pip | 26.2.1 |
| CUDA toolkit / NVCC | 12.4.131 |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| CMake | 4.4.3 |
| Ninja | 1.13.2 |
| FlashAttention | 2.7.3; CUDA kernel test passed |
| nvdiffrast | 0.4.0; commit `253ac4fcea7de5f396371124af597e6cc957bfae` |
| nvdiffrec_render | Source build at commit `b296927cc7fd01c2ac1087c8065c4d7248f72da4` |
| CuMesh | Source build at commit `12289e1062f0603f2f0d0771b02e1395d247f26f` |
| O-Voxel | Source build from the pinned TRELLIS.2 repository |
| FlexGEMM | Source build at commit `6dd94a859c26ee8246888502eada3dd8ad85532e` |
| Pillow | 12.3.0 |
| Transformers | 5.17.0 |
| Gradio | 6.0.1 |
| Trimesh | 5.1.0 |
| utils3d | 0.0.2 |

Source identities used by the final environment:

| Source | Revision |
|---|---|
| TRELLIS.2 | `75fbf0183001ed9876c8dbb35de6b68552ee08bd` |
| nvdiffrast | `253ac4fcea7de5f396371124af597e6cc957bfae` |
| nvdiffrec | `b296927cc7fd01c2ac1087c8065c4d7248f72da4` |
| CuMesh | `12289e1062f0603f2f0d0771b02e1395d247f26f` |
| FlexGEMM | `6dd94a859c26ee8246888502eada3dd8ad85532e` |
| Eigen submodule | `21e4582d1739107337a03460c81412981130373e` |

The final verification completed on 15 September 2026 at 20:28:36 +08:00. Detailed package inventories are retained on the cluster at:

```text
~/cp4281-as2/logs/step2-final-versions.txt
~/cp4281-as2/logs/step2-final-pip-freeze.txt
~/cp4281-as2/logs/step2-final-conda-explicit.txt
~/cp4281-as2/logs/step2-final-conda-list.txt
```

## Step 3 — Deploy and Generate

Step 3a is complete. Step 3b has not started.

### Step 3a — Image-Conditioned Generation

#### Objective

Generate five textured 3D assets from original photographs captured on location by the author. The photographs stress different single-view reconstruction capabilities: thin structures, hard surfaces, transparency, curved upholstered geometry, and repeated concave compartments. Each result must include a `.glb`, its background-removed conditioning image, at least two rendered viewpoints, and an individually recorded generation time.

#### Implementation

The native image-conditioned chain was deployed as:

```text
original photograph -> RMBG-2.0 -> DINOv3 ViT-L/16 -> TRELLIS.2-4B -> textured GLB
```

All five cases used seed `42`, the `1024_cascade` pipeline, 12 sampling steps in each TRELLIS stage, a 500,000-triangle export target, 2,048-pixel textures, and four 768-pixel rendered viewpoints. EXIF orientation was applied before preprocessing. The source images, fixed SHA-256 values, selection rationales, and Step 4 candidate flags are recorded in [`inputs/step3/experiment_spec.json`](inputs/step3/experiment_spec.json).

The reproducible implementation is preserved in:

- [`scripts/step3a_generate.py`](scripts/step3a_generate.py), which performs preprocessing, generation, GLB export, rendering, timing, and per-case metadata capture.
- [`scripts/step3a_batch.sbatch`](scripts/step3a_batch.sbatch), which verifies all revisions and input hashes and runs the five cases sequentially on one A100-40 GPU.
- [`scripts/verify_step3_pbr_envmap.sbatch`](scripts/verify_step3_pbr_envmap.sbatch), which validates the procedural PBR environment through the CUDA cubemap and mipmap path.

The pinned TRELLIS.2 revision required a runtime compatibility adapter for the Transformers 5 DINOv3 encoder layout. The adapter preserves TRELLIS.2's expected pre-normalisation feature extraction without changing the pinned source checkout. OpenCV could not decode the repository's DWAB-compressed EXR environment map on the cluster, so a deterministic procedural neutral studio environment was used for shaded PBR previews. This fallback was independently validated on an A100 before the batch run.

The final sequential batch was Slurm job `853514` on an NVIDIA A100-PCIE-40GB. The primary per-case metric covers image preprocessing, TRELLIS generation, and GLB export; model initialisation and preview rendering are recorded separately in each `metadata.json`.

#### Results

| Case | Capability tested | Primary time | Qualitative result |
|---|---|---:|---|
| Wired earphones | Thin structures | 359.43 s | Cables, both earbuds, inline controls, and the connector remained connected; one viewpoint clips an earbud at the frame edge. |
| White hatchback | Hard surfaces | 201.95 s | Coherent body, four wheels, glazing, and major lights; small text and unseen rear details are blurred or reconstructed. |
| Translucent glass cup | Transparency | 126.48 s | The opening, handle, walls, and ribbed base are preserved; transparency is foggy and printed decoration is inconsistently projected. |
| Yellow armchair | Curved upholstered form and thin legs | 240.47 s | Strong geometry and material consistency across all viewpoints, with only minor seam and surface noise. |
| Freestanding bookshelf | Concavities and repeated compartments | 111.16 s | Repeated shelves and recesses remain three-dimensional; unseen sides and the top are completed asymmetrically. |

The five primary times total 1,039.48 seconds (17 minutes 19 seconds). Every asset has four shaded PBR viewpoints, a valid glTF 2.0 GLB with embedded WebP textures, a background-removed input, and complete metadata. Structural and visual quality assurance passed for all five cases. The accepted local batch is stored under:

```text
outputs/step3/image_conditioned/batch-853514/
```

The selected Step 4 candidates remain the car, armchair, and bookshelf because they combine stable final geometry with distinct hard-surface, organic, and repeated-concavity behaviour.

### Step 3b — Text-Conditioned Generation

Not started. This stage will use FLUX.1-schnell to create five isolated single-object images from the original prompts in `experiment_spec.json`, then pass those intermediate images through the same validated image-to-3D chain.

## Step 4 — Visualise the Diffusion Process

Not started. This step will capture and decode five intermediate sampler states from each of TRELLIS.2's two flow-matching stages for three image-conditioned cases.

## Step 5 — Composite a Scene

Not started. This step will arrange at least five generated assets into one intentionally composed 3D scene.

## Submission Checklist

### Report

- [x] Step 1: downloaded models, official sources, revisions, storage location, and substitution statement
- [x] Step 2: ordered copy-pasteable environment commands and exact versions of Python, PyTorch, CUDA, and every compiled extension
- [x] Step 3a: five original input images, selection rationale, five `.glb` assets, at least two rendered viewpoints per asset, and individual generation times
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
- [x] Five image-conditioned `.glb` assets
- [ ] Five text-conditioned `.glb` assets
- [ ] One composed scene containing at least five generated assets (`.glb` preferred; `.blend` or `.usd` accepted)

### Oral Preparation

- [ ] Be able to explain the complete Step 3 conditioning chain
- [ ] Be able to explain both Step 4 flow-matching stages, sampler states, schedules, and decoders
- [ ] Be able to explain the Step 5 scene assembly and placement decisions
