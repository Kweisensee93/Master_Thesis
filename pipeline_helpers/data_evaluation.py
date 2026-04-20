# pipeline_helpers/data_evaluation.py
import pandas as pd
import numpy as np
import cv2
from pipeline_helpers import euclidean

#1. Metrics calculation
#2. Visualisation of results

def calculate_metrics(img_name, semilandmarks_new_crop, gt_semi_full, contour_full, ox, oy, getinfo = False):
    """
    Returns:
        master_row_list: List of dicts (Short version: coords only)
        detailed_df: pandas DataFrame (Extended version: coords + distances)
    """
    master_row_list = []
    detailed_rows = []
    
    # Map new points to full image space
    semilandmarks_new_full = semilandmarks_new_crop + np.array([ox, oy], dtype=np.float32)

    for i, pt_new in enumerate(semilandmarks_new_full):
        # Coordinates (Full Image Space)
        nx, ny = round(float(pt_new[0]), 2), round(float(pt_new[1]), 5)
        ox_val, oy_val = np.nan, np.nan
        
        if i < len(gt_semi_full):
            pt_gt = np.array(gt_semi_full[i], dtype=np.float32)
            ox_val, oy_val = round(float(pt_gt[0]), 5), round(float(pt_gt[1]), 5)

        # 1. Create the Master (Short) Row
        row_basic = {
            "ImageName": img_name, "Index": i,
            "new_x": nx, "new_y": ny, "old_x": ox_val, "old_y": oy_val
        }
        master_row_list.append(row_basic)

        if getinfo:
            # 2. Calculate Distances for the Detailed Version
            dist_new_old = np.nan
            dist_old_cont = np.nan

            if not np.isnan(ox_val):
                pt_gt = np.array([ox_val, oy_val], dtype=np.float32)
                dist_new_old = euclidean(pt_new, pt_gt)
                dist_old_cont = np.min(euclidean(contour_full, pt_gt))

            row_detailed = row_basic.copy()
            row_detailed.update({
                "Dist_New_to_Old": round(float(dist_new_old), 5),
                "Dist_Old_to_Contour": round(float(dist_old_cont), 5)
            })
            detailed_rows.append(row_detailed)

    return master_row_list, pd.DataFrame(detailed_rows)

def draw_information(img_cropped, fish_contour_crop, landmarks_full, 
                     gt_semi_full, semilandmarks_new_crop, ox, oy, 
                     output_path, subset_crop = None):
    """
    Generates a single master evaluation image with all features overlaid.
    """
    # 1. Start with a fresh BGR copy of the crop
    vis_img = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR)

    # 2a. Draw whole contour (Green)
    cv2.drawContours(vis_img, [fish_contour_crop], -1, (0, 255, 0), 2)

    # 2b. Draw Semilandmark Subset Contour (Purple)
    if subset_crop is not None and len(subset_crop) > 1:
        pts = subset_crop.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_img, [pts], isClosed=False, color=(255, 0, 128), thickness=2)

    # 3. Draw Original Landmarks (Red) - Shifted to crop space
    landmarks_crop = np.array(landmarks_full) - np.array([ox, oy])
    for pt in landmarks_crop:
        cv2.circle(vis_img, tuple(pt.astype(int)), 5, (0, 0, 255), -1)

    # 4. Draw Original GT Semilandmarks (Blue) - Shifted to crop space
    if gt_semi_full is not None and len(gt_semi_full) > 0:
        gt_semi_crop = np.array(gt_semi_full) - np.array([ox, oy])
        for pt in gt_semi_crop:
            cv2.circle(vis_img, tuple(pt.astype(int)), 5, (255, 0, 0), -1)

    # 5. Draw New Semilandmarks (Orange)
    for i, pt in enumerate(semilandmarks_new_crop):
        pt_int = tuple(pt.astype(int))
        
        # Anchors (first and last) get a black ring
        if i == 0 or i == len(semilandmarks_new_crop) - 1:
            cv2.circle(vis_img, pt_int, 8, (0, 0, 0), 3) # Black ring
            
        # Orange fill
        cv2.circle(vis_img, pt_int, 5, (0, 165, 255), -1) 

    # 6. Save
    cv2.imwrite(str(output_path), vis_img)

def export_results(res, output_dir, get_info, get_images, defined_file, debug_mode, get_contour = False):
    """
    res: Dictionary containing all geometric and processed data
    """
    stem = res["stem"]
    is_target = (defined_file == "all" or defined_file == stem)

    # A. CSV Outputs
    if get_info and is_target:
        res["img_metrics_df"].to_csv(output_dir / f"{stem}_metrics.csv", index=False)
    if get_contour and is_target:
        pd.DataFrame(res["contour_full"], columns=["x_full", "y_full"]).to_csv(
            output_dir / f"{stem}_full_contour.csv", index=False
        )

    # B. Visualizations
    if get_images and is_target:
        draw_information(
            res["img_cropped"], res["fish_contour_crop"], res["landmarks_full"], 
            res["gt_semi_full"], res["semilandmarks_new_crop"], res["ox"], res["oy"], 
            output_dir / f"{stem}_final_eval.png",
            subset_crop=res["subset_crop"]
        )

    # C. Debugging
    if debug_mode:
        from debug.debug_01 import run_debug_01
        run_debug_01(
            res["img_name"], res["semilandmarks_new_full"], res["gt_semi_full"], 
            res["contour_full"], res["crop_box"], 
            res["fish_contour_crop"].reshape(-1, 2), # contour_pts_crop
            res["idx_first"], res["idx_last"], res["subset_crop"],
            res["contours_crop"], output_dir, stem
        )