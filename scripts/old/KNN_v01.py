#KNN_v01.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_helpers import parse_tps, load_image # from data_io.py
from pipeline_helpers import crop_to_landmarks, build_dataset, KNNSemiLandmarkRegressor # from preprocessing.py
from pipeline_helpers import euclidean # from geometry.py

import os
import numpy as np
import pandas as pd

AddInfo     = False     # Set to True to enable additional info prints
GetInfo     = True      # Set to True to enable detailed info/CSV files
GetImages   = True      # Set to True to enable evaluation images
DefinedFile = "all"     # Specific stem e.g. "CC21L003" or "all"
K           = 5         # Number of neighbours for KNN

# Paths - may be replaced by a YAML config file later
PROJECT_DIR = Path("C:/Users/korbi/Desktop/A_Master_Thesis/")
FISH_DIR    = PROJECT_DIR / "rawdata"
TPS_FILE    = FISH_DIR / "landmark01.TPS"
OUTPUT_DIR  = PROJECT_DIR / "output/tmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":

    # 1. Get rawdata (images and TPS)
    tps_data = parse_tps(TPS_FILE)
    fish_images = sorted(f for f in os.listdir(FISH_DIR) if f.lower().endswith(".jpg"))
    if len(tps_data) != len(fish_images):
        print(f"WARNING: Number of specimens in TPS ({len(tps_data)}) does not match number of images in storage ({len(fish_images)}).")
    if AddInfo: print(f"Found {len(fish_images)} images and {len(tps_data)} TPS entries.")

    # 2. Train / val split  (every 4th image → validation)
    fish_np      = np.array(fish_images)
    fish_np = fish_np[0:80] # TEMPORARY: limit to 20 images for quick testing
    mask         = np.arange(len(fish_np)) % 4 == 3
    val_names    = fish_np[mask]
    train_names  = fish_np[~mask]
    if AddInfo: print(f"  Train: {len(train_names)}  |  Val: {len(val_names)}")

    # 3. build datasets
    x_train, y_train, train_boxes, train_names_ok = build_dataset(
        train_names, FISH_DIR, tps_data, include_semi=True
    )
    print(f"  x_train : {x_train.shape}  (N, H, W, C)")
    print(f"  y_train : {y_train.shape}  (N, n_semi*2)  ← normalised semi-landmark coords")

    # Validation set OMITS semi-landmarks (withheld – to be predicted)
    if AddInfo: print("\nBuilding validation dataset …")
    x_val, _, val_boxes, val_names_ok = build_dataset(
        val_names, FISH_DIR, tps_data, include_semi=False
    )

    # 4. Train KNN regressor
    if AddInfo: print(f"\nTraining KNN regressor (k={K}) …")
    regressor = KNNSemiLandmarkRegressor(k=K)
    regressor.train(x_train, y_train)
    if AddInfo: print("  Done (lazy learner – training is just storing the data).")

    # 5. Predict on validation set
    if AddInfo: print("\nPredicting semi-landmarks for validation set …")
    predicted_semi  = []
    all_k_distances = []
    all_k_indices   = []

    for i, val_img in enumerate(x_val):
        pred, k_dists, k_idxs = regressor.predict(val_img)
        predicted_semi.append(pred)
        all_k_distances.append(k_dists)
        all_k_indices.append(k_idxs)

        neighbour_names = ", ".join(train_names_ok[j] for j in k_idxs)
        print(f"  [{i+1}/{len(x_val)}] {val_names_ok[i]:20s}  "
              f"k neighbours: [{neighbour_names}]  "
              f"dists: {np.round(k_dists, 2)}")

    predicted_semi = np.stack(predicted_semi, axis=0)

    # 6. Evaluate against ground truth
    _, y_val_true, _, val_lmarks_true, _ = build_dataset(
        val_names_ok, FISH_DIR, tps_data, include_semi=True
    )
    
    old_to_new_dist = euclidean(predicted_semi, y_val_true)

    # Export: CSV + evaluation image via the shared export_results function
    geometry_results = {

        }
    export_results(geometry_results, OUTPUT_DIR, GetInfo, GetImages, DefinedFile, debug_01=False)

    # 8. Master CSV (same format as remg_approach.py)
    if all_master_results:
        master_df = pd.DataFrame(all_master_results)
        master_df.to_csv(OUTPUT_DIR / "master_semilandmark_metrics.csv", index=False)
        print(f"\nMaster CSV saved -> {OUTPUT_DIR / 'master_semilandmark_metrics.csv'}")
