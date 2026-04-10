# KNN_v04.py
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
AddInfo = True
K       = 5

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path("C:/Users/korbi/Desktop/A_Master_Thesis/")
FISH_DIR    = PROJECT_DIR / "rawdata"
TPS_FILE    = FISH_DIR / "landmark01.TPS"
OUTPUT_DIR  = PROJECT_DIR / "output/tmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    tps_data = parse_tps(TPS_FILE)
    fish_images = sorted(f for f in os.listdir(FISH_DIR) if f.lower().endswith(".jpg"))

    # 1. Split and Build
    fish_np = np.array(fish_images)
    mask = np.arange(len(fish_np)) % 4 == 3
    
    # Train set
    x_train, y_train_bare, y_train_rel, _, _, train_names_ok = build_dataset(fish_np[~mask], FISH_DIR, tps_data)
    # Val set
    x_val, y_val_bare, y_val_rel, val_boxes, val_lmarks, val_names_ok = build_dataset(fish_np[mask], FISH_DIR, tps_data)

    # 2. Train Two Regressors
    reg_bare = KNNSemiLandmarkRegressor(k=K)
    reg_bare.train(x_train, y_train_bare)

    reg_rel = KNNSemiLandmarkRegressor(k=K)
    reg_rel.train(x_train, y_train_rel)

    rows_bare, rows_norm = [], []

    for i, val_img in enumerate(x_val):
        img_name = val_names_ok[i]
        img_h = cv2.imread(str(FISH_DIR / img_name)).shape[0]

        # Predictions
        pred_b_norm, _, _ = reg_bare.predict(val_img)
        pred_r_norm, _, _ = reg_rel.predict(val_img)

        # Ground Truth in Full Image Space (Original Coordinates)
        gt_px_full = flip_y(tps_data[img_name]["semi_landmarks"][0], img_h) #

        # --- CSV 1: Bare (No LM1/LM13 correction, just crop-based) ---
        # Map [0,1] crop-coords back to full image pixels
        pts_pred_bare = denormalise_landmarks(pred_b_norm, val_boxes[i])
        
        for j, (p, g) in enumerate(zip(pts_pred_bare, gt_px_full)):
            rows_bare.append({
                "img_name": img_name, "index": j,
                "X_new": round(p[0], 3), "Y_new": round(p[1], 3),
                "X_old": round(g[0], 3), "Y_old": round(g[1], 3),
                "dist_Old_to_New": round(euclidean(p, g), 3)
            })

        # --- CSV 2: Corrected (LM1->LM13 basis correction) ---
        # Map (t, d) relative coords back to full image pixels
        pts_pred_rel = denormalise_landmarks_relative(pred_r_norm, val_lmarks[i])
        
        for j, (p, g) in enumerate(zip(pts_pred_rel, gt_px_full)):
            rows_norm.append({
                "img_name": img_name, "index": j,
                "X_new": round(p[0], 3), "Y_new": round(p[1], 3),
                "X_old": round(g[0], 3), "Y_old": round(g[1], 3),
                "dist_Old_to_New": round(euclidean(p, g), 3)
            })

    # Save outputs
    pd.DataFrame(rows_bare).to_csv(OUTPUT_DIR / "knn_results_bare.csv", index=False)
    pd.DataFrame(rows_norm).to_csv(OUTPUT_DIR / "knn_results_corrected.csv", index=False)
    print("CSVs generated in original image coordinate system.")