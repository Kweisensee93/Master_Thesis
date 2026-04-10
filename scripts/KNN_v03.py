# KNN_v02.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_helpers import parse_tps                                              # data_io.py
from pipeline_helpers import build_dataset, KNNSemiLandmarkRegressor                # preprocessing.py
from pipeline_helpers import normalise_landmarks, denormalise_landmarks            # preprocessing.py (Bare)
from pipeline_helpers import normalise_landmarks_relative, denormalise_landmarks_relative  # normalisation.py (LM Basis)
from pipeline_helpers import euclidean, flip_y, to_crop_space                       # geometry.py

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
    tps_data    = parse_tps(TPS_FILE)
    fish_images = sorted(f for f in os.listdir(FISH_DIR) if f.lower().endswith(".jpg"))

    # 1. Train / Val Split
    fish_np     = np.array(fish_images)
    mask        = np.arange(len(fish_np)) % 4 == 3
    train_names = fish_np[~mask]
    val_names   = fish_np[mask]

    # 2. Build Datasets
    # Version A: Bare Normalized [0,1] within crop
    # Version B: Relative to LM1-LM13 basis
    x_train, y_train_bare, _, train_lmarks, train_names_ok = build_dataset(
        train_names, FISH_DIR, tps_data, include_semi=True
    )
    
    # Generate Version B training targets (LM1->LM13 relative)
    y_train_relative = []
    for i, name in enumerate(train_names_ok):
        img_h = cv2.imread(str(FISH_DIR / name)).shape[0]
        semi_px = flip_y(tps_data[name]["semi_landmarks"][0], img_h)
        lms_px  = flip_y(tps_data[name]["landmarks"], img_h)
        y_train_relative.append(normalise_landmarks_relative(semi_px, lms_px))
    y_train_relative = np.array(y_train_relative)

    x_val, _, val_boxes, val_lmarks, val_names_ok = build_dataset(
        val_names, FISH_DIR, tps_data, include_semi=False
    )

    # 3. Prediction
    # Model for Bare coordinates
    reg_bare = KNNSemiLandmarkRegressor(k=K)
    reg_bare.train(x_train, y_train_bare)

    # Model for Relative coordinates
    reg_rel = KNNSemiLandmarkRegressor(k=K)
    reg_rel.train(x_train, y_train_relative)

    rows_bare = []
    rows_norm = []

    for i, val_img in enumerate(x_val):
        img_name = val_names_ok[i]
        
        # Predict Bare [0,1] coordinates
        pred_bare, _, _ = reg_bare.predict(val_img)
        # Predict Relative (t, d) coordinates
        pred_rel, _, _  = reg_rel.predict(val_img)

        # Convert Bare to Crop Pixel Space
        # (Assuming denormalise_landmarks maps [0,1] back to the crop box)
        pts_bare_px = denormalise_landmarks(pred_bare, val_boxes[i])
        
        # Convert Relative to Full Image Pixel Space then to Crop Space
        pts_rel_full = denormalise_landmarks_relative(pred_rel, val_lmarks[i])
        pts_rel_crop = to_crop_space(pts_rel_full, val_boxes[i])

        # Log results...
        # (Data collection logic similar to your monolith KNN_v02.py)