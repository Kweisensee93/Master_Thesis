# knn_approach.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_helpers import parse_tps                                              # data_io.py
from pipeline_helpers import build_dataset, KNNSemiLandmarkRegressor                # preprocessing.py
from pipeline_helpers import euclidean, flip_y, to_crop_space                       # geometry.py
from pipeline_helpers import normalise_landmarks_relative, denormalise_landmarks_relative  # normalisation.py

import os
import cv2
import numpy as np
import pandas as pd

# ── Settings ──────────────────────────────────────────────────────────────────
AddInfo = False
K       = 5

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path("C:/Users/korbi/Desktop/A_Master_Thesis/")
FISH_DIR    = PROJECT_DIR / "rawdata"
TPS_FILE    = FISH_DIR / "landmark01.TPS"
OUTPUT_DIR  = PROJECT_DIR / "output/tmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_img_h(img_path: Path) -> int:
    """Return image height without loading full pixel data."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    return img.shape[0]

if __name__ == "__main__":

    # 1. Parse TPS + discover images
    tps_data    = parse_tps(TPS_FILE)
    fish_images = sorted(f for f in os.listdir(FISH_DIR) if f.lower().endswith(".jpg"))
    if len(tps_data) != len(fish_images):
        print(f"WARNING: TPS has {len(tps_data)} entries but found {len(fish_images)} images.")

    # 2. Train / val split (every 4th image → validation)
    fish_np     = np.array(fish_images)
    mask        = np.arange(len(fish_np)) % 4 == 3
    val_names   = fish_np[mask]
    train_names = fish_np[~mask]
    if AddInfo: print(f"Train: {len(train_names)}  |  Val: {len(val_names)}")

    # 3. Build datasets
    # Returns: images, semi_arr, crop_boxes, landmarks_list, names
    x_train, y_train, _,         train_lmarks, train_names_ok = build_dataset(
        train_names, FISH_DIR, tps_data, include_semi=True
    )
    x_val,   _,       val_boxes, val_lmarks,   val_names_ok   = build_dataset(
        val_names, FISH_DIR, tps_data, include_semi=False
    )
    if AddInfo:
        print(f"x_train: {x_train.shape}  y_train: {y_train.shape}")
        print(f"x_val  : {x_val.shape}")

    # 4. Train KNN
    regressor = KNNSemiLandmarkRegressor(k=K)
    regressor.train(x_train, y_train)

    # 5. Predict on validation set (output in normalised t,d space)
    predicted_semi = []
    for i, val_img in enumerate(x_val):
        pred, k_dists, k_idxs = regressor.predict(val_img)
        predicted_semi.append(pred)
        if AddInfo:
            neighbour_names = ", ".join(train_names_ok[j] for j in k_idxs)
            print(f"  [{i+1}/{len(x_val)}] {val_names_ok[i]:20s}  "
                  f"neighbours: [{neighbour_names}]  dists: {np.round(k_dists, 2)}")
    predicted_semi = np.stack(predicted_semi, axis=0)   # (N, n_semi*2)

    # 6. Load GT semi-landmarks for evaluation
    _, y_val_true, _, _, _ = build_dataset(
        val_names_ok, FISH_DIR, tps_data, include_semi=True
    )

    # 7. Build per-point output rows – unnormalised (px) and normalised (t,d)
    rows_unnorm = []
    rows_norm   = []

    for i, img_name in enumerate(val_names_ok):
        data   = tps_data.get(img_name, {})
        lmarks = val_lmarks[i]     # y-flipped fixed landmarks, full image space
        box    = val_boxes[i]      # (min_x, max_x, min_y, max_y)
        img_h  = get_img_h(FISH_DIR / img_name)

        # Predicted: denormalise (t,d) → full image px → crop px
        pred_full = denormalise_landmarks_relative(predicted_semi[i], lmarks)
        pred_crop = to_crop_space(pred_full, box)

        # GT: y-flip → full image px → crop px
        gt_full = flip_y(data.get("semi_landmarks", [[]])[0], img_h)
        gt_crop = to_crop_space(gt_full, box)

        # GT in normalised (t,d) space – matches predicted_semi[i] coordinate frame
        gt_norm_arr = normalise_landmarks_relative(gt_full, lmarks)

        # ── Version A: unnormalised (crop pixel space) ────────────────────────
        pts_pred_crop = np.array(pred_crop, dtype=np.float32)
        pts_gt_crop   = np.array(gt_crop,   dtype=np.float32)

        for j, (p, g) in enumerate(zip(pts_pred_crop, pts_gt_crop)):
            rows_unnorm.append({
                "image_name":      img_name,
                "point_index":     j,
                "x_new":           round(float(p[0]), 3),
                "y_new":           round(float(p[1]), 3),
                "x_old":           round(float(g[0]), 3),
                "y_old":           round(float(g[1]), 3),
                "Dist_New_to_Old": round(float(euclidean(p, g)), 3),
            })

        # ── Version B: normalised (t,d relative to LM1→LM13) ─────────────────
        pts_pred_norm = predicted_semi[i].reshape(-1, 2)
        pts_gt_norm   = gt_norm_arr.reshape(-1, 2)

        for j, (p, g) in enumerate(zip(pts_pred_norm, pts_gt_norm)):
            rows_norm.append({
                "image_name":      img_name,
                "point_index":     j,
                "x_new":           round(float(p[0]), 6),   # t_predicted
                "y_new":           round(float(p[1]), 6),   # d_predicted
                "x_old":           round(float(g[0]), 6),   # t_groundtruth
                "y_old":           round(float(g[1]), 6),   # d_groundtruth
                "Dist_New_to_Old": round(float(euclidean(p, g)), 6),
            })

    # 8. Save CSVs
    df_unnorm = pd.DataFrame(rows_unnorm)
    df_norm   = pd.DataFrame(rows_norm)

    df_unnorm.to_csv(OUTPUT_DIR / "knn_results_unnormalised.csv", index=False)
    df_norm.to_csv(  OUTPUT_DIR / "knn_results_normalised.csv",   index=False)
    print(f"Saved unnormalised -> {OUTPUT_DIR / 'knn_results_unnormalised.csv'}")
    print(f"Saved normalised   -> {OUTPUT_DIR / 'knn_results_normalised.csv'}")

    print(f"\nMean Dist_New_to_Old  (px, unnormalised) : {df_unnorm['Dist_New_to_Old'].mean():.2f}")
    print(f"Mean Dist_New_to_Old  (t/d, normalised)  : {df_norm['Dist_New_to_Old'].mean():.6f}")