# remg approach
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_helpers.config import PADDING, CROP_LENGTH, CROP_HEIGHT, SEMILANDMARK_CURVE # from config.py
from pipeline_helpers import parse_tps, load_image # from data_io.py
from pipeline_helpers import flip_y, coords_to_array, array_to_coords, euclidean, find_anchor_points, get_contour_subset, resample_points # from geometry.py
from pipeline_helpers import crop_to_landmarks, normalise_landmarks, denormalise_landmarks, build_dataset # from preprocessing.py
from pipeline_helpers import mean_landmark_error # from data_evaluation.py
from pipeline_helpers.visualisation import draw_information # from visualisation.py

from rembg import remove
from PIL import Image
import cv2
import numpy as np
import io
import os
import pandas as pd

# Settings
debug_01 = True     # Set to True to enable debug prints and visualisations
AddInfo = False     # Set to True to enable additional info prints of the processing
GetInfo = False     # Set to True to enable detailed info files
GetImages = True   # Set to True to enable images with contour and landmarks drawn
DefinedFile = "all" # Set to specific image name (without extension) to enable detailed outputs for that image only, e.g. "CC21L003"

# README - or don't
# variables within this script are often set either in the original image space or in a cropped
# down image space. Hence, it is made explicit by appending _crop or _full to the variable names.

# Paths
PROJECT_DIR = Path("C:/Users/korbi/Desktop/A_Master_Thesis/")
FISH_DIR    = PROJECT_DIR / "rawdata"
TPS_FILE    = FISH_DIR / "landmark01.TPS"
OUTPUT_DIR  = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":

    # 1. Get rawdata (images and TPS)
    tps_data = parse_tps(TPS_FILE)
    fish_images = sorted(f for f in os.listdir(FISH_DIR) if f.lower().endswith(".jpg")    )
    if len(tps_data) != len(fish_images):
        print(f"WARNING: Number of specimens in TPS ({len(tps_data)}) does not match number of images ({len(fish_images)}).")
    # for testing: reduce to 2 images
    fish_images = fish_images[0:2]

    # 2. Process each image
    for img_name in fish_images:
        # Step 1: Setup and load data
        img_path = FISH_DIR / img_name
        stem = Path(img_name).stem
        if AddInfo: print(f"\nProcessing {img_name} ...")
        data = tps_data.get(img_name, {})
        img_bgr = load_image(img_path)
        # step 2: crop to landmark-box; retrieve geometry data
        img_h = img_bgr.shape[0]    # get original image height for flip_y() later
        img_cropped, crop_box = crop_to_landmarks(img_bgr, data)
        cropped_bgr = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV
        ox, oy = crop_box[0], crop_box[2] # get point of origin coordinates for offsetting later

        # Step 3: Convert cropped image to bytes and remove background
        img_cropped_pil = Image.fromarray(img_cropped)  # already RGB from crop_to_landmarks
        buf = io.BytesIO() # create in-memory buffer, so we can feed it to rembg, without an intermediate file on disk
        img_cropped_pil.save(buf, format="PNG")
        output_data = remove(buf.getvalue())
        img_rgba = np.array(Image.open(io.BytesIO(output_data)).convert("RGBA"))

        # Step 4: Use alpha channel as mask; find contours with mask
        alpha = img_rgba[:, :, 3]
        _, binary_mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        #CHAIN_APPROX_NONE ensures every pixel is stored; RETR_EXTERNAL retrieves only the outermost contour
        contours_crop, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours_crop:
            print(f"WARNING:  No contours found for {img_name}, skipping.")
            continue
        # The "Glue" Method: Merge multiple pieces into one solid fish mask
        temp_mask = np.zeros_like(binary_mask)
        for cnt in contours_crop:
            #if cv2.contourArea(cnt) > 50: # Add a threshold to filter out small fragments (optional, adjust as needed)
            cv2.drawContours(temp_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # Re-extract the single master contour from the "glued" mask
        merged_cnts, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        fish_contour_crop = max(merged_cnts, key=cv2.contourArea)

        # Step 5: Save contour points to CSV
        coords_crop = fish_contour_crop.reshape(-1, 2)
        df = pd.DataFrame(coords_crop, columns=["x", "y"])
        csv_path = OUTPUT_DIR / f"{stem}_contour.csv"
        df.to_csv(str(csv_path), index=False)
        if AddInfo: print(f"  Contour points saved: {csv_path}  ({len(df)} points)")

        # Step 6: Find closest contour points to first and last landmark
        landmarks_full = flip_y(data.get("landmarks", []), img_h)
        if len(landmarks_full) < 2:
            print(f"  Not enough landmarks for {img_name}, skipping step 8.")
            continue

        # Get all contour points as (N, 2) array
        contour_pts_crop = fish_contour_crop.reshape(-1, 2).astype(np.float32)

        # Find closest contour point to first and last landmark; Keep coordinates and indices
        closest_first_crop, closest_last_crop, idx_first, idx_last = find_anchor_points(fish_contour_crop, landmarks_full, crop_box)

        if AddInfo: print(f"  Closest contour point to first landmark {landmarks_full[0]}: {closest_first_crop}")
        if AddInfo: print(f"  Closest contour point to last landmark {landmarks_full[-1]}:  {closest_last_crop}")

        # Subset contour between the two anchor points (semilandmark_full)
        subset_crop = get_contour_subset(contour_pts_crop, idx_first, idx_last)
        semilandmarks_new_crop = resample_points(subset_crop, num_points=10) # get 10 equidistant points along the contour subset

        # ── Save anchor points ──
        anchor_df = pd.DataFrame([
            {"label": "closest_to_first_lm",
             "x_cropped": closest_first_crop[0], "y_cropped": closest_first_crop[1],
             "x_full": closest_first_crop[0] + ox, "y_full": closest_first_crop[1] + oy},
            {"label": "closest_to_last_lm",
             "x_cropped": closest_last_crop[0],  "y_cropped": closest_last_crop[1],
             "x_full": closest_last_crop[0] + ox,  "y_full": closest_last_crop[1] + oy},
        ])
        anchor_path = OUTPUT_DIR / f"{stem}_anchors.csv"
        anchor_df.to_csv(str(anchor_path), index=False)
        print(f"  Anchor points saved: {anchor_path}")

        # ── Save semilandmark_full (contour subset) ──
        semi_full_df = pd.DataFrame({
            "x_crop": subset_crop[:, 0].astype(int),
            "y_crop": subset_crop[:, 1].astype(int),
            "x_full":    (subset_crop[:, 0] + ox).astype(int),
            "y_full":    (subset_crop[:, 1] + oy).astype(int),
        })
        semi_full_path = OUTPUT_DIR / f"{stem}_semilandmark_full.csv"
        semi_full_df.to_csv(str(semi_full_path), index=False)
        print(f"  Full contour subset saved: {semi_full_path}  ({len(semi_full_df)} points)")

        # ── Save semilandmarks_new (10 equidistant points) ──
        semi_new_df = pd.DataFrame({
            "label":     [f"semi_{i:02d}" for i in range(10)],
            "x_crop": semilandmarks_new_crop[:, 0].astype(int),
            "y_crop": semilandmarks_new_crop[:, 1].astype(int),
            "x_full":    (semilandmarks_new_crop[:, 0] + ox).astype(int),
            "y_full":    (semilandmarks_new_crop[:, 1] + oy).astype(int),
        })
        semi_new_path = OUTPUT_DIR / f"{stem}_semilandmarks_new.csv"
        semi_new_df.to_csv(str(semi_new_path), index=False)
        print(f"  New semilandmarks saved: {semi_new_path}")

        # Step7: 
        # 7.1. Get Ground Truth (GT) semilandmarks from TPS (full image space)
        gt_semi_full = flip_y(data.get("semi_landmarks", [[]])[SEMILANDMARK_CURVE], img_h)
        gt_semi_full = gt_semi_full[::-1] # reverse order to match contour direction (comment out if needed)

        # 7.2. Convert semilandmarks_new_crop fromcrop space → full image space
        semilandmarks_new_full = semilandmarks_new_crop + np.array([ox, oy], dtype=np.float32) # already in crop space = full space now

        # 7.3. Convert contour from resized-crop space → full image space
        contour_full   = fish_contour_crop.reshape(-1, 2).astype(np.float32) + np.array([ox, oy], dtype=np.float32)

        # 7.4. Compute distances
        results_list = []
        for i, pt_pred in enumerate(semilandmarks_new_full):
            # A. Distance to nearest point on the contour (full image space)
            dist_to_contour = np.min(euclidean(contour_full, pt_pred))

            # B. Distance to GT counterpart (full image space)
            dist_to_counterpart = np.nan
            if i < len(gt_semi_full):
                pt_gt = np.array(gt_semi_full[i], dtype=np.float32)
                dist_to_counterpart = euclidean(pt_pred, pt_gt)

            results_list.append({
                "ImageName":          img_name,
                "Semilandmark_Index": i,
                "Dist_to_Contour":    round(dist_to_contour,      4),
                "Dist_to_Counterpart": round(dist_to_counterpart, 4),
            })

        # 7.7. Save per-image distance CSV
        dist_df   = pd.DataFrame(results_list)
        dist_path = OUTPUT_DIR / f"{stem}_distances.csv"
        dist_df.to_csv(str(dist_path), index=False)
        print(f"  Distances saved: {dist_path}")

        if GetImages and (DefinedFile == "all" or DefinedFile == stem):
            draw_information(
                img_cropped=img_cropped,
                fish_contour_crop=fish_contour_crop,
                landmarks_full=landmarks_full,
                gt_semi_full=gt_semi_full,
                semilandmarks_new_crop=semilandmarks_new_crop,
                ox=ox, oy=oy,
                output_path=OUTPUT_DIR / f"{stem}_final_eval.png"
            )


        if debug_01:
            from debug.debug_01 import run_debug_01
            run_debug_01(
                img_name=img_name,
                semilandmarks_new_full=semilandmarks_new_full,
                gt_semi_full=gt_semi_full,
                contour_full=contour_full,
                crop_box=crop_box,
                contour_pts_crop=contour_pts_crop,
                idx_first=idx_first,
                idx_last=idx_last,
                subset_crop=subset_crop,
                contours_crop=contours_crop,
                output_dir=OUTPUT_DIR,
                stem=stem,
            )
# ── After loop: combine all distance CSVs into one master file ────────────────
all_dist_files = list(OUTPUT_DIR.glob("*_distances.csv"))
if all_dist_files:
    master_df = pd.concat([pd.read_csv(f) for f in all_dist_files], ignore_index=True)
    master_path = OUTPUT_DIR / "master_semilandmark_metrics.csv"
    master_df.to_csv(str(master_path), index=False)
    print(f"\nMaster CSV created at: {master_path}")
