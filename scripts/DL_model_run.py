# DL_models_predict_v01.py
# ---------------------------------------------------------------------------
# Pure INFERENCE driver (no ground truth required).
#
# Loads an already-trained model and predicts semi-landmarks for a new
# dataset that only contains the fixed landmarks (LM= block in the TPS,
# no CURVES). Nothing here reads "semi_landmarks", so a landmark-only TPS
# works fine.
#
#   model_type in config: knn | mlp | cnn | gnn | vit   (hybrid == cnn)
#
# Output CSV schema (no ground-truth columns, no distance):
#   img_name, model_type, index, X_pred, Y_pred
# ---------------------------------------------------------------------------

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd

from pipeline_helpers import KNNSemiLandmarkRegressor  # noqa: F401 (used via pickle)
from pipeline_helpers import SimpleFishMLP, HybridFishNet, FishGNN, FishViT  # noqa: F401
from DL_models_learning import model_path_for, canonical
from pipeline_helpers import (
    flip_y, parse_tps, denormalise_landmarks_relative,
    parse_args, load_config,
)

# Reuse the exact preprocessing + per-model inference from the eval driver
# so predictions are produced identically to how the model was validated.
from DL_models_run_v01 import load_model, predict_new


# ── Prediction loop ──────────────────────────────────────────────────────────
def run_prediction(model, model_type, image_names, fish_dir, tps_data, output_dir):
    mt   = canonical(model_type)
    rows = []
    skipped = 0

    for img_name in image_names:
        img_path = fish_dir / img_name

        data = tps_data.get(img_name, {})
        lmarks = data.get("landmarks", [])
        if len(lmarks) < 13:                      # need LM1 (idx 0) and LM13 (idx 12)
            print(f"  Skipping {img_name} – needs >=13 landmarks, found {len(lmarks)}")
            skipped += 1
            continue

        # (n_semi, 2) predictions in the LM1->LM13 relative basis
        preds_relative = predict_new(img_path, tps_data, model, mt)

        img_h       = cv2.imread(str(img_path)).shape[0]
        lmarks_full = flip_y(lmarks, img_h)
        pts_pred    = denormalise_landmarks_relative(preds_relative, lmarks_full)

        for j, p in enumerate(pts_pred):
            rows.append({
                "img_name":   img_name,
                "model_type": mt,
                "index":      j,
                "X_pred":     round(float(p[0]), 5),
                "Y_pred":     round(float(p[1]), 5),
            })

    results_df  = pd.DataFrame(rows)
    results_csv = output_dir / f"{mt}_predictions.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\n  Predictions saved -> {results_csv}")
    print(f"  {len(image_names) - skipped} image(s) predicted, {skipped} skipped, "
          f"{len(rows)} landmark rows written.")
    return results_df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)

    PROJECT_DIR = Path(cfg.get("PROJECT_DIR", "C:/Users/korbi/Desktop/A_Master_Thesis/"))
    FISH_DIR    = Path(cfg.get("FISH_DIR",    PROJECT_DIR / "rawdata"))
    TPS_FILE    = Path(cfg.get("TPS_FILE",    FISH_DIR / "landmark01.TPS"))
    OUTPUT_DIR  = Path(cfg.get("OUTPUT_DIR",  PROJECT_DIR / "output/tmp"))
    MODEL_DIR   = Path(cfg.get("DL_MODEL_DIR", PROJECT_DIR / "DL_model/"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_type = canonical(cfg.get("model_type", "cnn"))
    model_path = model_path_for(MODEL_DIR, model_type)

    # Inference only: the model MUST already exist. No auto-training here.
    if not model_path.exists():
        sys.exit(
            f"ERROR: no trained {model_type.upper()} model at {model_path}.\n"
            f"       Train it first, or fix 'model_type' / 'DL_MODEL_DIR' in the config."
        )

    model = load_model(model_type, model_path)
    print(f"Loaded trained {model_type.upper()} model from {model_path}")

    tps_data    = parse_tps(TPS_FILE)
    fish_images = sorted(
        f for f in os.listdir(FISH_DIR)
        if f.lower().endswith(".jpg") and f in tps_data
    )

    if not fish_images:
        sys.exit(
            f"ERROR: no .jpg images in {FISH_DIR} matched entries in {TPS_FILE.name}.\n"
            f"       Check FISH_DIR and TPS_FILE point at the NEW dataset."
        )

    print(f"\nRunning inference on {len(fish_images)} image(s) …")
    run_prediction(model, model_type, fish_images, FISH_DIR, tps_data, OUTPUT_DIR)