# CP4281 Assignment 2: Deploying TRELLIS.2 for Conditional 3D Generation

## Overview

This project deploys `microsoft/TRELLIS.2-4B` on the NUS SoC Compute Cluster and evaluates image-conditioned and chained text-conditioned 3D generation. It also captures and decodes the running flow-matching states from both TRELLIS.2 geometry stages, assembles six generated assets into one scene, and tests a higher-resolution `1536_cascade` extension. All reported outputs were produced by the pinned local model snapshots and scripts in this repository.

## 1. Model Weights

All weights came from their official Hugging Face repositories and were stored under `/home/h/hongshan/cp4281-as2/hf-cache`. Download job `849112` completed in 13 minutes 29 seconds; the verified cache occupied 53 GB and contained no incomplete files.

| Model and official source | Revision | Role |
|---|---|---|
| [`microsoft/TRELLIS.2-4B`](https://huggingface.co/microsoft/TRELLIS.2-4B) | `af44b45f2e35a493886929c6d786e563ec68364d` | Mandatory two-stage 3D generator and shape/texture VAEs |
| [`microsoft/TRELLIS-image-large`](https://huggingface.co/microsoft/TRELLIS-image-large) | `25e0d31ffbebe4b5a97464dd851910efc3002d96` | Sparse-structure decoder checkpoint |
| [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | `ea8dc2863c51be0a264bab82070e3e8836b02d51` | Image conditioner |
| [`briaai/RMBG-2.0`](https://huggingface.co/briaai/RMBG-2.0) | `5df4c9c76d8170882c34f6986e848ee07fd0ba43` | Input background removal |
| [`black-forest-labs/FLUX.1-schnell`](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | `741f7c3ce8b383c54771c7003378a50191e9efe9` | Text-to-image front end for Step 3b |

No substitute 3D model was used. RMBG-2.0 and FLUX.1-schnell were assignment-approved choices. Only the Diffusers-format FLUX files were retained; the redundant standalone `flux1-schnell.safetensors` was excluded. Gated repositories were accessed with an authorised read-only token, which was never stored in the repository.

## 2. Isolated Environment

The runtime was installed at `/home/h/hongshan/cp4281-as2/envs/trellis2`; no system interpreter was modified. The condensed, copy-pasteable installation order was:

```bash
PROJECT_ROOT="$HOME/cp4281-as2"
ENV_PREFIX="$PROJECT_ROOT/envs/trellis2"

conda create --prefix "$ENV_PREFIX" python=3.10 pip -y
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

conda install -c nvidia -c conda-forge \
  cuda-toolkit=12.4 cuda-nvcc=12.4 gcc=12 gxx=12 cmake ninja -y

python -m pip install \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

git clone --recursive https://github.com/microsoft/TRELLIS.2.git \
  "$PROJECT_ROOT/src/TRELLIS.2"
git -C "$PROJECT_ROOT/src/TRELLIS.2" checkout \
  75fbf0183001ed9876c8dbb35de6b68552ee08bd

python -m pip install -r "$PROJECT_ROOT/src/TRELLIS.2/requirements.txt"
python -m pip install flash-attn==2.7.3 --no-build-isolation

sbatch scripts/setup_step2_render.sbatch
sbatch scripts/setup_step2_geometry.sbatch
sbatch scripts/verify_step2_environment.sbatch
sbatch scripts/setup_step3b_accelerate.sbatch
sbatch scripts/setup_step3b_diffusers.sbatch
sbatch scripts/verify_step3b_text_stack.sbatch
```

Each Slurm stage was allowed to complete and pass before the next was submitted. The scripts contain the exact compiler flags, pinned source revisions, temporary build paths, installation commands, and functional tests. CUDA extensions were compiled for compute capability 8.0 with CUDA 12.4, GCC/G++ 12, and local `/tmp` build directories.

| Component | Version / identity |
|---|---|
| Python / pip | 3.10.21 / 26.2.1 |
| CUDA toolkit / NVCC | 12.4.131 |
| PyTorch / torchvision | 2.6.0+cu124 / 0.21.0+cu124 |
| FlashAttention | 2.7.3; CUDA kernel test passed |
| nvdiffrast | 0.4.0; `253ac4fcea7de5f396371124af597e6cc957bfae` |
| nvdiffrec_render | `b296927cc7fd01c2ac1087c8065c4d7248f72da4` |
| CuMesh | `12289e1062f0603f2f0d0771b02e1395d247f26f` |
| O-Voxel | Source build from pinned TRELLIS.2 |
| FlexGEMM | `6dd94a859c26ee8246888502eada3dd8ad85532e` |
| Transformers / Pillow / Trimesh | 5.17.0 / 12.3.0 / 5.1.0 |
| Diffusers / Accelerate / psutil | 0.40.0 / 1.15.0 / 7.2.2 |

The final environment passed full pipeline imports, CUDA execution, all compiled-extension tests, FLUX loading from the offline snapshot, and `pip check`. Complete `pip freeze`, explicit Conda, and package-list manifests are retained under `~/cp4281-as2/logs/step2-final-*` and `step3b-text-stack-final-pip-freeze-855195.txt`.

## 3. Conditional 3D Generation

### 3.1 Image-conditioned generation

The native path was:

```text
photograph -> RMBG-2.0 -> DINOv3 ViT-L/16 -> TRELLIS.2-4B -> textured GLB
```

Five photographs captured by the author were chosen to stress distinct reconstruction properties. All cases used seed `42`, `1024_cascade`, 12 steps in each sampler, a 500,000-triangle export target, 2,048-pixel textures, and four fixed 768-pixel previews. The primary time includes preprocessing, TRELLIS generation, and GLB export, but excludes shared model initialisation and preview rendering.

| Case | Stress tested | Primary time | Result |
|---|---|---:|---|
| Wired earphones | Thin cables and connectors | 359.43 s | Connected earbuds, cable, controls, and plug; one view clips an earbud |
| White hatchback | Hard surfaces | 201.95 s | Coherent body, wheels, glazing, and lights; fine text is blurred |
| Glass cup | Transparency | 126.48 s | Opening, handle, walls, and base survive; material becomes foggy |
| Yellow armchair | Curves and thin legs | 240.47 s | Strong upholstery and leg geometry with minor surface noise |
| Bookshelf | Repeated recesses | 111.16 s | Shelves remain three-dimensional; hidden sides are inferred asymmetrically |

#### Image inputs and two-view previews

<table>
<tr><th>Case and input</th><th>Rendered view 00</th><th>Rendered view 02</th></tr>
<tr><td><b>Wired earphones</b><br><img src="inputs/step3/image_conditioned/img01_thin_structure/earphone.JPG" width="220"></td><td><img src="outputs/step3/image_conditioned/batch-853514/img01_thin_structure/view_00.png" width="220"></td><td><img src="outputs/step3/image_conditioned/batch-853514/img01_thin_structure/view_02.png" width="220"></td></tr>
<tr><td><b>White hatchback</b><br><img src="inputs/step3/image_conditioned/img02_hard_surface/white_car.JPG" width="220"></td><td><img src="outputs/step3/image_conditioned/batch-853514/img02_hard_surface/view_00.png" width="220"></td><td><img src="outputs/step3/image_conditioned/batch-853514/img02_hard_surface/view_02.png" width="220"></td></tr>
<tr><td><b>Glass cup</b><br><img src="inputs/step3/image_conditioned/img03_transparency/glass_cup.JPG" width="220"></td><td><img src="outputs/step3/image_conditioned/batch-853514/img03_transparency/view_00.png" width="220"></td><td><img src="outputs/step3/image_conditioned/batch-853514/img03_transparency/view_02.png" width="220"></td></tr>
<tr><td><b>Yellow armchair</b><br><img src="inputs/step3/image_conditioned/img04_organic_shape/yellow_sofa.JPG" width="220"></td><td><img src="outputs/step3/image_conditioned/batch-853514/img04_organic_shape/view_00.png" width="220"></td><td><img src="outputs/step3/image_conditioned/batch-853514/img04_organic_shape/view_02.png" width="220"></td></tr>
<tr><td><b>Bookshelf</b><br><img src="inputs/step3/image_conditioned/img05_complex_shape/bookshelf.JPG" width="220"></td><td><img src="outputs/step3/image_conditioned/batch-853514/img05_complex_shape/view_00.png" width="220"></td><td><img src="outputs/step3/image_conditioned/batch-853514/img05_complex_shape/view_02.png" width="220"></td></tr>
</table>

Formal Slurm job `853514` generated all five GLBs and twenty previews. The five primary times total 1,039.48 seconds (17 minutes 19 seconds).

### 3.2 Text-conditioned generation

TRELLIS.2 has no text conditioner, so I built this chain:

```text
prompt -> FLUX.1-schnell -> intermediate image -> RMBG-2.0
       -> DINOv3 ViT-L/16 -> TRELLIS.2-4B -> textured GLB
```

The shared prompt template was:

```text
Studio image of {subject_description}. Isolated, centered, fully visible,
three-quarter view, plain gray background, diffuse lighting, realistic
material. No people, props, text, logo, watermark, cropping, or duplicates.
```

The five subject descriptions were generated with AI assistance, then manually reviewed and approved. They are project-specific and are not represented as solely student-authored. Every expanded prompt was checked against the CLIP 77-token and T5 256-token limits before generation.

| Case | Exact `{subject_description}` substitution | FLUX | TRELLIS | Combined |
|---|---|---:|---:|---:|
| Desk fan | `a vintage mint-green metal desk fan with five blades, a circular wire cage, and a compact rounded base` | 113.55 s | 150.46 s | 264.01 s |
| Toolbox | `a compact teal steel toolbox with a raised folding handle, two silver latches, and slightly rounded corners` | 37.58 s | 148.54 s | 186.12 s |
| Perfume bottle | `an asymmetric translucent amber glass perfume bottle with a faceted clear stopper and a curved silhouette` | 39.24 s | 108.82 s | 148.06 s |
| Root sculpture | `a small freestanding sculpture shaped like an intertwined weathered tree root with branching knots and a broad stable base` | 39.23 s | 69.42 s | 108.65 s |
| Lunar rover | `a stylized compact lunar rover toy with four ribbed wheels, a small tilted dish antenna, and an asymmetric instrument box` | 46.67 s | 159.23 s | 205.90 s |

#### FLUX intermediates and two-view previews

<table>
<tr><th>Case and FLUX intermediate</th><th>Rendered view 00</th><th>Rendered view 02</th></tr>
<tr><td><b>Desk fan</b><br><img src="outputs/step3/text_conditioned/batch-855432/txt01_desk_fan/flux/intermediate.png" width="220"></td><td><img src="outputs/step3/text_conditioned/batch-855432/txt01_desk_fan/trellis/view_00.png" width="220"></td><td><img src="outputs/step3/text_conditioned/batch-855432/txt01_desk_fan/trellis/view_02.png" width="220"></td></tr>
<tr><td><b>Toolbox</b><br><img src="outputs/step3/text_conditioned/batch-855432/txt02_toolbox/flux/intermediate.png" width="220"></td><td><img src="outputs/step3/text_conditioned/batch-855432/txt02_toolbox/trellis/view_00.png" width="220"></td><td><img src="outputs/step3/text_conditioned/batch-855432/txt02_toolbox/trellis/view_02.png" width="220"></td></tr>
<tr><td><b>Perfume bottle</b><br><img src="outputs/step3/text_conditioned/batch-855432/txt03_perfume_bottle/flux/intermediate.png" width="220"></td><td><img src="outputs/step3/text_conditioned/batch-855432/txt03_perfume_bottle/trellis/view_00.png" width="220"></td><td><img src="outputs/step3/text_conditioned/batch-855432/txt03_perfume_bottle/trellis/view_02.png" width="220"></td></tr>
<tr><td><b>Root sculpture</b><br><img src="outputs/step3/text_conditioned/batch-855432/txt04_root_sculpture/flux/intermediate.png" width="220"></td><td><img src="outputs/step3/text_conditioned/batch-855432/txt04_root_sculpture/trellis/view_00.png" width="220"></td><td><img src="outputs/step3/text_conditioned/batch-855432/txt04_root_sculpture/trellis/view_02.png" width="220"></td></tr>
<tr><td><b>Lunar rover</b><br><img src="outputs/step3/text_conditioned/batch-855432/txt05_lunar_rover/flux/intermediate.png" width="220"></td><td><img src="outputs/step3/text_conditioned/batch-855432/txt05_lunar_rover/trellis/view_00.png" width="220"></td><td><img src="outputs/step3/text_conditioned/batch-855432/txt05_lunar_rover/trellis/view_02.png" width="220"></td></tr>
</table>

Formal job `855432` produced five FLUX images, five GLBs, and twenty previews. The combined primary time was 912.73 seconds (15 minutes 13 seconds). The toolbox had the strongest planar result; fan cage wires warped, glass cues in the perfume became opaque, and fine root tips occasionally detached.

The lunar rover most clearly exposes the chained system's limitation. FLUX supplied only one three-quarter image and omitted the underside, rear attachments, and hidden instrument layout. TRELLIS.2 never received the text, so it inferred those regions only from pixels, yielding a recognisable rover with simplified and partly invented rear geometry. The perfume case similarly shows that painted highlights do not fully specify glass thickness or transmission.

## 4. Visualising Both Flow-Matching Stages

Three image-conditioned cases were selected: the car for hard surfaces, armchair for curved upholstery and thin legs, and bookshelf for repeated concavities. Each case contains five decoded states from each stage, for 30 intermediate renders total.

### Capture and decoding method

TRELLIS.2 first samples a dense sparse-structure latent that determines occupied geometry, then samples a sparse shape-SLat latent that resolves the detailed surface. [`scripts/step4_capture.py`](scripts/step4_capture.py) temporarily wraps `FlowEulerSampler.sample` at runtime; the pinned upstream checkout is unchanged. The wrapper captures the running iterate `x_t` carried after each Euler update. Because history entry `i` is the post-update state, it is labelled with `schedule[i + 1]`, not the pre-update `schedule[i]`.

| Stage | Formal model | Euler steps | Actual retained timesteps | Decoding |
|---|---|---:|---|---|
| 1: coarse sparse structure | `sparse_structure_flow_model` | 24 | `0.500000, 0.416667, 0.312500, 0.178571, 0.000000` | `sparse_structure_decoder`, then occupied cells rendered as voxels |
| 2: detailed shape-SLat | `shape_slat_flow_model_1024` | 12 | `0.970588, 0.750000, 0.500000, 0.214286, 0.000000` | Rebuild `SparseTensor`, reverse normalisation, decode a 1024 mesh, render normals |

Stage 1 was increased from 12 to 24 Euler steps because its output is empty above approximately `t=0.5`; 24 steps provide five distinct useful post-update states across `0.5` to `0.0`. The `1024_cascade` invokes Stage 2 at 512 and then 1024 resolution. Both calls were recorded, but formal strips use only the final 1024 pass. Fixed cameras and true timesteps make progression comparable.

### White hatchback

<p><img src="outputs/step4/smoke/img02-hard-surface-job-856096/preprocessed.png" width="260"> <img src="outputs/step4/smoke/img02-hard-surface-job-856096/final_render.png" width="260"></p>

*Left: background-removed conditioning input. Right: final rendered asset.*

![White hatchback Stage 1](outputs/step4/smoke/img02-hard-surface-job-856096/visualizations/stage1_diffusion_strip.png)

![White hatchback Stage 2](outputs/step4/smoke/img02-hard-surface-job-856096/visualizations/stage2_diffusion_strip.png)

The footprint grows from 51 to 2,008 occupied voxels; the final Stage 2 state has 4,631,952 decoded faces. Stage 1 establishes the car volume, while Stage 2 resolves windows, wheels, and body panels.

### Yellow armchair

<p><img src="outputs/step4/formal/batch-856142/img04_organic_shape/preprocessed.png" width="260"> <img src="outputs/step4/formal/batch-856142/img04_organic_shape/final_render.png" width="260"></p>

*Left: background-removed conditioning input. Right: final rendered asset.*

![Yellow armchair Stage 1](outputs/step4/formal/batch-856142/img04_organic_shape/visualizations/stage1_diffusion_strip.png)

![Yellow armchair Stage 2](outputs/step4/formal/batch-856142/img04_organic_shape/visualizations/stage2_diffusion_strip.png)

The final structure has 4,340 occupied voxels and the last decoded state has 8,940,018 faces. The curved enclosing back and thin legs become coherent, although inferred rear fabric grooves are exaggerated.

### Bookshelf

<p><img src="outputs/step4/formal/batch-856142/img05_complex_shape/preprocessed.png" width="260"> <img src="outputs/step4/formal/batch-856142/img05_complex_shape/final_render.png" width="260"></p>

*Left: background-removed conditioning input. Right: final rendered asset.*

![Bookshelf Stage 1](outputs/step4/formal/batch-856142/img05_complex_shape/visualizations/stage1_diffusion_strip.png)

![Bookshelf Stage 2](outputs/step4/formal/batch-856142/img05_complex_shape/visualizations/stage2_diffusion_strip.png)

The final structure has 3,207 occupied voxels and the last Stage 2 decode has 10,655,286 faces. The tall volume stabilises first; repeated shelves and recesses separate later. Early Stage 2 face counts are not monotonic because each noisy latent is independently decoded while its topology changes.

The smoke case ran as job `856096`; armchair and bookshelf ran as job `856142`, both on A100 80 GB GPUs. Generation-and-capture times excluding model initialisation were 43.11, 59.25, and 40.30 seconds respectively.

## 5. Composite Scene

**Adaptive Maker Garage** combines a vehicle-maintenance bay on the left with a reading/design lounge on the right. Six generated assets were assembled programmatically with Trimesh: white hatchback, teal toolbox, yellow armchair, bookshelf, mint desk fan, and amber perfume bottle. Each GLB was uniformly scaled from its bounds to a chosen metre-scale extent, centred in X/Z, rotated by an explicit yaw, and translated until its minimum Y coordinate contacted the floor, rug, or table top. Procedural walls, floor, rug, and table provide context and physical support.

| Asset | Target extent | Position `(x,y,z)` m | Yaw | Support |
|---|---:|---:|---:|---|
| Car | Z = 4.20 m | `(-1.35, 0.00, -0.35)` | 20 degrees | Floor |
| Toolbox | X = 0.55 m | `(-2.75, 0.00, 2.25)` | 5 degrees | Floor |
| Armchair | X = 0.90 m | `(1.65, 0.02, -0.55)` | -20 degrees | Rug |
| Bookshelf | Y = 1.90 m | `(3.15, 0.00, -3.05)` | -8 degrees | Floor |
| Fan | Y = 0.45 m | `(2.70, 0.66, 0.25)` | -18 degrees | Table |
| Perfume | Y = 0.25 m | `(3.08, 0.66, 0.45)` | 12 degrees | Table |

![Adaptive Maker Garage](outputs/step5/formal/job-856482/hero_view.png)

Formal job `856482` exported all six assets plus nine support geometries as one 116,793,092-byte GLB and reloaded all 15 geometries successfully. Every support-contact error was at most `1e-5` m. Nvdiffrast rendered the 1280-by-720 view off-screen because Blender and Pyrender were unavailable in the validated environment. Visual inspection found all six assets identifiable, no primary subject cropped, and no obvious floating or severe interpenetration.

## Optional Extension: 1536 Resolution

The bookshelf was regenerated with `1536_cascade` to test whether higher resolution causes a visible improvement. Source image, seed, sampling/guidance parameters, export target, texture size, and cameras matched the accepted `1024_cascade` baseline; only pipeline resolution changed.

| Metric | 1024 baseline | 1536 | Ratio |
|---|---:|---:|---:|
| TRELLIS generation | 61.5257 s | 101.1527 s | 1.644x |
| End-to-end case time | 111.1562 s | 149.6901 s | 1.347x |
| Peak CUDA allocation | 6.49 GiB | 14.20 GiB | 2.188x |
| Working vertices | 5,620,620 | 8,092,547 | 1.440x |
| Working faces | 11,429,892 | 16,344,616 | 1.430x |
| Voxel size | 0.0009765625 | 0.0006510417 | 0.667x |

![All matched 1024 and 1536 views](outputs/optional/high_resolution/job-857824/figures/comparison_all_views.png)

![Matched structural detail crops](outputs/optional/high_resolution/job-857824/figures/comparison_detail_crops.png)

The 1536 output has straighter shelf edges, more continuous dividers, and more regular recess and panel-seam spacing. However, book covers and small labels remain unreadable, some textures become noisier, and hidden surfaces are still guesses. Higher resolution therefore improved geometric regularity and separation of repeated elements, but not semantic texture fidelity or unseen-surface ambiguity. It cost 64.4% more generation time and 118.8% more peak allocation. Timing is a practical rather than strict hardware benchmark because the original baseline ran on an A100 40 GB allocation and job `857824` ran on an A100 80 GB PCIe GPU.

## Deliverables and Reproducibility

The submitted files comprise the ten Step 3 GLBs, the composite [`scene.glb`](outputs/step5/formal/job-856482/scene.glb), and this report converted to PDF. Per-case metadata records inputs, prompts, revisions, parameters, time, hashes, and artifact paths. The main reproducibility entry points are `step3a_batch.sbatch`, `step3b_batch.sbatch`, `step4_formal.sbatch`, `step5_formal.sbatch`, and `optional_highres.sbatch` under [`scripts/`](scripts/).

AI assistance was used for scripting and report drafting, and for the five text subject descriptions. All generated text and code were manually reviewed against logs, metadata, source code, and rendered outputs. The observations above report actual outcomes, including visible limitations and failed assumptions, rather than treating higher numerical resolution as automatically better.
