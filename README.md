# Automated Semilandmark Detection in Three-Spined Stickleback

Automated detection of landmarks and semilandmarks on 2D lateral images of
three-spined stickleback (*Gasterosteus aculeatus*), benchmarked against
manually digitized landmarks and compared across classical image-analysis and
machine-learning / deep-learning (AI) approaches.

This repository contains the pipeline developed for my MSc thesis in
bioinformatics.

## Overview

The pipeline takes standardized lateral photographs of stickleback,
automatically segments the fish from the background, extracts the body outline,
and places anatomical landmarks together with equidistant semilandmarks along
that outline. The automatically generated configurations are then compared
against a manually digitized ground truth (TPS format) to quantify accuracy on
a per-landmark basis.

## Approaches compared

**Image analysis (contour-based).**
Deep-learning background removal (`rembg`) → contour extraction (OpenCV) →
contour smoothing (Savitzky–Golay filter) → landmark anchoring and semilandmark
placement along the outline.

**Machine learning / AI.**
Models trained to predict landmark coordinates directly from the images or
derived features: k-nearest neighbours (KNN), multilayer perceptron (MLP),
convolutional neural network (CNN), graph neural network (GNN) and vision
transformer (ViT). The model is selected via `model_type` in `config.yaml`
(`"hybrid"` is an alias for the CNN).

## Repository structure

| Path | Contents |
|------|----------|
| `cluster/` | Shell submission scripts for the University of Bern UBELIX HPC cluster |
| `debug/` | Debugging and visualization utilities |
| `DL_model/` | Storage of trained models and loss plots |
| `ImageAnalysis_deprecated/` | Image Analysis approach |
| `pipeline_helpers/` | Helper modules for segmentation, contour handling and landmark placement |
| `scripts/` | Main pipeline entry points |
| `config.yaml` | Central configuration (paths, processing flags, model and training settings) |
| `requirements_*.txt` | Python dependencies |

* different requirement.txt files may be loaded. Either with fixed versions or bare libraries names

## Installation

```bash
git clone https://github.com/Kweisensee93/Master_Thesis.git
cd Master_Thesis
python -m venv .venv && source .venv/bin/activate   # or a conda env
pip install -r requirements.txt
```

For exact reproduction see the environment_full_PipFreeze.txt

## Usage

All behaviour is driven by `config.yaml`. Set the input paths (`FISH_DIR`,
`TPS_FILE`, `OUTPUT_DIR`), choose the `model_type`, and adjust the processing
flags, then run the pipeline entry point in `scripts/`. Key options:

- `DefinedFile` — process a single image (e.g. `"CC21L003"`) or `"all"`.
- `Keep_landmarks_as_anchors` — keep true landmarks fixed, or snap them to the
  nearest contour point.
- `Fast_Mode` — trade a small amount of accuracy for faster processing.
- `Number_of_worst_performers_review` — export the *n* worst-performing images
  for manual inspection.
- `epochs`, `batch_size`, `lr`, `k` — training hyperparameters (KNN uses only
  `k`).

## Note on a dropped approach

An earlier version of the segmentation stage explored building the fish outline
by applying a bank of several classical image filters and combining their
outputs. This approach was **dropped**: running several filters per image did
not scale to the full dataset, making it computationally too expensive relative
to the deep-learning background removal that replaced it. It is documented here
for completeness and is not part of the final pipeline.

## Data and compute

Images originate from the 2D stickleback morphometrics collection of the
Institute of Ecology and Evolution, University of Bern. The pipeline was
developed and executed on the University of Bern UBELIX HPC cluster.
