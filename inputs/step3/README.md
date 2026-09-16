# Step 3 Input Specification

This directory contains the source inputs and the experiment specification for
Step 3. It must not contain generated previews or model outputs.

## Image-conditioned inputs

Place one original source image in each case directory:

```text
image_conditioned/
  img01_thin_structure/source.<ext>
  img02_hard_surface/source.<ext>
  img03_transparency/source.<ext>
  img04_organic_shape/source.<ext>
  img05_complex_shape/source.<ext>
```

Use the highest-quality original image available. Do not pre-remove the
background or overwrite the source image. A derived, background-removed image
will be saved with the generated outputs later. Record the image provenance and
selection rationale in `experiment_spec.json` before generation.

For reliable single-view reconstruction, prefer a single fully visible object,
clear separation from the background, limited occlusion, and diffuse lighting.
At least three cases should also have clear silhouettes suitable for the Step 4
intermediate-state visualisation.

## Text-conditioned inputs

The five original subject descriptions and the common FLUX prompt template are
stored in `experiment_spec.json`. During generation, the exact expanded prompt
and random seed must be copied into each case's output metadata.

## Source/output separation

- `inputs/step3/` stores immutable source inputs and experiment intent.
- `outputs/step3/` will store derived images, GLB files, renders, and metadata.
- `logs/step3/` on the cluster will store Slurm and runtime logs.

