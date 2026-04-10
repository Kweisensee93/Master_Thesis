# remg approach
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_helpers.config import SEMILANDMARK_CURVE # from config.py
from pipeline_helpers import parse_tps, load_image # from data_io.py
from pipeline_helpers import flip_y, find_anchor_points, get_contour_subset, resample_points # from geometry.py
from pipeline_helpers import crop_to_landmarks # from preprocessing.py
from pipeline_helpers import calculate_metrics,  export_results # from data_evaluation.py
from pipeline_helpers import extract_fish_contour, remove_background # from image_operations.py

import numpy as np
import os
import pandas as pd

# Settings
debug_01 = False    # Set to True to enable debug prints and visualisations
AddInfo = False     # Set to True to enable additional info prints of the processing
GetInfo = True    # Set to True to enable detailed info files
GetImages = True   # Set to True to enable images with contour and landmarks drawn
DefinedFile = "all" # Set to specific image name (without extension) e.g. "CC21L003" or "all"

# README - or don't
# variables within this script are often set either in the original image space or in a cropped
# down image space. Hence, it is made explicit by appending _crop or _full to the variable names.

# Paths - may be replaced by a YAML config file later
PROJECT_DIR = Path("C:/Users/korbi/Desktop/A_Master_Thesis/")
FISH_DIR    = PROJECT_DIR / "rawdata"
TPS_FILE    = FISH_DIR / "landmark01.TPS"
OUTPUT_DIR  = PROJECT_DIR / "output/tmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":

    # 1. Get rawdata (images and TPS)
    all_master_results = [] # Accumulator for master CSV
    tps_data = parse_tps(TPS_FILE)
    fish_images = sorted(f for f in os.listdir(FISH_DIR) if f.lower().endswith(".jpg"))
    if len(tps_data) != len(fish_images):
        print(f"WARNING: Number of specimens in TPS ({len(tps_data)}) does not match number of images in storage ({len(fish_images)}).")
    if AddInfo: print(f"Found {len(fish_images)} images and {len(tps_data)} TPS entries.")
    # for testing: reduce to 2 images; 174 total
    fish_images = fish_images[::11]

    # 2. Process each image
    for img_name in fish_images:
        # Step 1: Setup
        img_path = FISH_DIR / img_name
        stem = Path(img_name).stem
        if AddInfo: print(f"\nProcessing {img_name} ...")
        data = tps_data.get(img_name, {})
        # step 2: crop to landmark-box; retrieve geometry data
        img_bgr = load_image(img_path)
        img_h = img_bgr.shape[0]    # get original image height for flip_y() later
        img_cropped, crop_box = crop_to_landmarks(img_bgr, data)
        ox, oy = crop_box[0], crop_box[2] # get point of origin coordinates for offsetting later

        # Step 3: Convert cropped image to bytes and remove background; Use alpha channel as mask; find contours with mask
        img_rgba = remove_background(img_cropped)
        fish_contour_crop, contours_crop = extract_fish_contour(img_rgba)
        
        if fish_contour_crop is None:
            print(f"WARNING: No contours found for {img_name}, skipping.")
            continue

        # Step 4: Find closest contour points to first and last landmark
        landmarks_full = flip_y(data.get("landmarks", []), img_h)
        if len(landmarks_full) < 2:
            print(f"  Not enough landmarks for {img_name}, skipping.")
            continue

        # Get all contour points as (N, 2) array
        contour_pts_crop = fish_contour_crop.reshape(-1, 2).astype(np.float32)
        # Find closest contour point to first and last landmark; Keep coordinates and indices
        closest_first_crop, closest_last_crop, idx_first, idx_last = find_anchor_points(fish_contour_crop, landmarks_full, crop_box)

        # Step 5: Subset contour between the two anchor points (semilandmark_full)
        subset_crop = get_contour_subset(contour_pts_crop, idx_first, idx_last)
        semilandmarks_new_crop = resample_points(subset_crop, num_points=10) # get 10 equidistant points along the contour subset

        # Step 6: 
        # 6.1. Get Ground Truth (GT) semilandmarks from TPS (full image space)
        gt_semi_full = flip_y(data.get("semi_landmarks", [[]])[SEMILANDMARK_CURVE], img_h)
        gt_semi_full = gt_semi_full[::-1] # reverse order to match contour direction (comment out if needed)
        semilandmarks_new_full = semilandmarks_new_crop + np.array([ox, oy], dtype=np.float32) # already in crop space = full space now
        contour_full   = fish_contour_crop.reshape(-1, 2).astype(np.float32) + np.array([ox, oy], dtype=np.float32)
        # 6.2. Calculate metrics
        img_results, img_metrics_df = calculate_metrics(
            img_name, semilandmarks_new_crop, gt_semi_full, contour_full, ox, oy, getinfo = GetInfo
        )
        all_master_results.extend(img_results)

        # Step 7: Detailed Outputs: Collect all data, pass it to CSV export, IMG drawing, debugging
        geometry_results = {
            "img_name": img_name,
            "stem": stem,
            "ox": ox, "oy": oy,
            "crop_box": crop_box,
            "img_cropped": img_cropped,
            "fish_contour_crop": fish_contour_crop,
            "contours_crop": contours_crop,
            "contour_full": contour_full,
            "landmarks_full": landmarks_full,
            "gt_semi_full": gt_semi_full,
            "semilandmarks_new_crop": semilandmarks_new_crop,
            "semilandmarks_new_full": semilandmarks_new_full,
            "subset_crop": subset_crop,
            "idx_first": idx_first, "idx_last": idx_last,
            "img_metrics_df": img_metrics_df
        }
        export_results(geometry_results, OUTPUT_DIR, GetInfo, GetImages, DefinedFile, debug_01)
# ── After loop: combine all distance CSVs into one master file ────────────────
    if all_master_results:
            master_df = pd.DataFrame(all_master_results)
            master_df.to_csv(OUTPUT_DIR / "master_semilandmark_metrics.csv", index=False)
