# KNN_v05.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_helpers import (
    parse_tps, build_dataset, KNNSemiLandmarkRegressor,
    denormalise_landmarks,normalise_landmarks_relative , denormalise_landmarks_relative,
    to_crop_space, euclidean, flip_y
)
import os
import cv2
import numpy as np
import pandas as pd

# ── Settings ──────────────────────────────────────────────────────────────────
AddInfo     = False     # Set to True to enable additional info prints
GetInfo     = True      # Set to True to enable detailed info/CSV files
GetImages   = True      # Set to True to enable evaluation images
DefinedFile = "all"     # Specific stem e.g. "CC21L003" or "all"
K           = 5         # Number of neighbours for KNN

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path("C:/Users/korbi/Desktop/A_Master_Thesis/")
FISH_DIR    = PROJECT_DIR / "rawdata"
TPS_FILE    = FISH_DIR / "landmark01.TPS"
OUTPUT_DIR  = PROJECT_DIR / "output/tmp1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # 0. Load TPS and Image List
    tps_data = parse_tps(TPS_FILE)
    fish_images = sorted(f for f in os.listdir(FISH_DIR) if f.lower().endswith(".jpg"))
    if len(tps_data) != len(fish_images):
        print(f"WARNING: Number of specimens in TPS ({len(tps_data)}) does not match number of images in storage ({len(fish_images)}).")
    if AddInfo: print(f"Found {len(fish_images)} images and {len(tps_data)} TPS entries.")

    # 1. Split and Build; for now every 4th image is validation
    fish_np = np.array(fish_images)     #all image names in a numpy array for easier indexing
    mask = np.arange(len(fish_np)) % 4 == 3
    train_np = fish_np[~mask]
    val_np = fish_np[mask]
    if AddInfo: print(f"  Train: {len(train_np)}  |  Val: {len(val_np)}")

    # Build datasets; crop box is given in helper function
    # We get both the "bare" (_crop) and "relative" (LM1->LM13 corrected) semi-landmark coords for training;
    x_train, y_train_crop, y_train_RelativeToLM, _, _, train_names_ok = build_dataset(train_np, FISH_DIR, tps_data)
    x_val, y_val_crop, y_val_RelativeToLM, val_boxes, val_lmarks_full, val_names_ok = build_dataset(val_np, FISH_DIR, tps_data)

    # 2. Train Two Regressors
    if AddInfo: print(f"\nTraining KNN regressor (k={K}) …")
    reg_crop = KNNSemiLandmarkRegressor(k=K)
    reg_crop.train(x_train, y_train_crop)

    reg_RelativeToLM = KNNSemiLandmarkRegressor(k=K)
    reg_RelativeToLM.train(x_train, y_train_RelativeToLM)

    rows_crop, rows_RelativeToLM = [], []

    for i, val_img in enumerate(x_val):
        img_name = val_names_ok[i]
        img_h = cv2.imread(str(FISH_DIR / img_name)).shape[0]

        # Predictions
        pred_crop, _, _ = reg_crop.predict(val_img)
        pred_RelativeToLM, _, _ = reg_RelativeToLM.predict(val_img)

        # Ground Truth in Full Image Space (Original Coordinates)
        gt_full = flip_y(tps_data[img_name]["semi_landmarks"][0], img_h) #

        # --- CSV 1: Crop space (No LM1/LM13 correction, just crop-based) ---
        # Map [0,1] crop-coords back to full image pixels
        pts_pred_crop = denormalise_landmarks(pred_crop, val_boxes[i])
        
        for j, (p, g) in enumerate(zip(pts_pred_crop, gt_full)):
            rows_crop.append({
                "img_name": img_name, "index": j,
                "X_new": round(p[0], 3), "Y_new": round(p[1], 3),
                "X_old": round(g[0], 3), "Y_old": round(g[1], 3),
                "dist_Old_to_New": round(euclidean(p, g), 3)
            })

        # --- CSV 2: Corrected (LM1->LM13 basis correction) ---
        # Map (t, d) relative coords back to full image pixels
        pts_pred_RelativeToLM = denormalise_landmarks_relative(pred_RelativeToLM, val_lmarks_full[i])
        
        for j, (p, g) in enumerate(zip(pts_pred_RelativeToLM, gt_full)):
            rows_RelativeToLM.append({
                "img_name": img_name, "index": j,
                "X_new": round(p[0], 3), "Y_new": round(p[1], 3),
                "X_old": round(g[0], 3), "Y_old": round(g[1], 3),
                "dist_Old_to_New": round(euclidean(p, g), 3)
            })

    # Save outputs
    pd.DataFrame(rows_crop).to_csv(OUTPUT_DIR / "knn_results_crop.csv", index=False)
    pd.DataFrame(rows_RelativeToLM).to_csv(OUTPUT_DIR / "knn_results_corrected.csv", index=False)
    if AddInfo: print("CSVs generated in original image coordinate system.")
    