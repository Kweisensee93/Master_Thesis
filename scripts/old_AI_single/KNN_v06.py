# KNN_v05.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_helpers import (
    parse_tps, parse_args, build_dataset, build_dataset_resized, KNNSemiLandmarkRegressor, denormalise_landmarks,
    denormalise_landmarks_relative, euclidean, flip_y, load_config, load_train_KNN_model
    )
import os
import cv2
import numpy as np
import pandas as pd

# Number of neighbours for KNN
K = 5

if __name__ == "__main__":
    args = parse_args() # Variables and paths are defined via YAML or JSON file
    cfg = load_config(args.config) #Fallback to default values, if either no value is

    AddInfo         = cfg.get("AddInfo",            False)     # Set to True to enable additional info prints
    GetInfo         = cfg.get("GetInfo",            True)      # Set to True to enable detailed info files
    GetImages       = cfg.get("GetImages",          True)      # Set to True to enable images with contour and landmarks drawn as output
    DefinedFile     = cfg.get("DefinedFile",        "all")     # Set to specific image name (without extension) e.g. "CC21L003" or "all"
    # Define Paths for the files needed.
    PROJECT_DIR     = Path(cfg.get("PROJECT_DIR",   "C:/Users/korbi/Desktop/A_Master_Thesis/"))
    FISH_DIR        = Path(cfg.get("FISH_DIR",      PROJECT_DIR / "rawdata"))
    TPS_FILE        = Path(cfg.get("TPS_FILE",      FISH_DIR / "landmark01.TPS"))
    OUTPUT_DIR      = Path(cfg.get("OUTPUT_DIR",    PROJECT_DIR / "output/tmp"))
    OUTPUT_DIR.mkdir(parents=False, exist_ok=True)
    # Check for Regressor:
    REGRESSOR_DIR   = Path(cfg.get("KNN_MODEL_DIR", "/storage/homefs/kw23y068/Master_Thesis/KNN_model/"))

    # 0. Load TPS and Image List
    tps_data = parse_tps(TPS_FILE)
    fish_images = sorted(f for f in os.listdir(FISH_DIR)
                         if f.lower().endswith(".jpg") and f in tps_data)
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
    x_train, y_train_crop, y_train_RelativeToLM, _, _, train_names_ok = build_dataset_resized(train_np, FISH_DIR, tps_data, target_size=(1024, 256))
    x_val, y_val_crop, y_val_RelativeToLM, val_boxes, val_lmarks_full, val_names_ok = build_dataset_resized(val_np, FISH_DIR, tps_data, target_size=(1024, 256))

    model_name_reg = f"reg_crop_k{K}.pkl"
    model_name_rel = f"reg_relative_k{K}.pkl"
    reg_crop_path = REGRESSOR_DIR / model_name_reg
    reg_rel_path = REGRESSOR_DIR / model_name_rel
    print(f"reg_crop:         {'LOAD' if reg_crop_path.exists() else 'TRAIN'} → {reg_crop_path}")
    print(f"reg_RelativeToLM: {'LOAD' if reg_rel_path.exists() else 'TRAIN'} → {reg_rel_path}")

    # 2. Train Two Regressors
    if AddInfo: print(f"\nLoading or Training KNN regressor (k={K}) …")
    reg_crop = load_train_KNN_model(
        reg_crop_path, KNNSemiLandmarkRegressor,
        x_train, y_train_crop, K, model_name_reg, AddInfo
        )
    #reg_crop.train(x_train, y_train_crop)

    reg_RelativeToLM = load_train_KNN_model(
        reg_rel_path, KNNSemiLandmarkRegressor,
        x_train, y_train_RelativeToLM, K, model_name_rel, AddInfo
        )
    #reg_RelativeToLM.train(x_train, y_train_RelativeToLM)

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
    
    # --- KNN Summary and Worst Performers Report ---
    # Combine all results for analysis
    all_knn_results = pd.DataFrame(rows_RelativeToLM) # Focus on the corrected version

    if not all_knn_results.empty:
        # 1. Generate Summary CSV
        knn_summary = pd.DataFrame({
            "Metric": ["Euclidean_Distance_Error"],
            "mean":   [all_knn_results["dist_Old_to_New"].mean()],
            "median": [all_knn_results["dist_Old_to_New"].median()],
            "std":    [all_knn_results["dist_Old_to_New"].std()],
            "max":    [all_knn_results["dist_Old_to_New"].max()]
        }).round(5)
        knn_summary.to_csv(OUTPUT_DIR / "knn_summary_metrics.csv", index=False)

        # 2. Identify Worst Performers
        # We group by image and find the one with the highest average error
        worst_knn = (
            all_knn_results.groupby("img_name")["dist_Old_to_New"]
            .mean()
            .nlargest(5) # Top 5 worst
            .reset_index()
            .rename(columns={"dist_Old_to_New": "mean_error_pixels"})
        )
        worst_knn.round(5).to_csv(OUTPUT_DIR / "knn_worst_performers.csv", index=False)