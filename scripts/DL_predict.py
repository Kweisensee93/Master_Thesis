import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import cv2
import numpy as np
import pandas as pd
from DL_model import HybridFishNet
from pipeline_helpers import crop_to_landmarks, flip_y, parse_tps, euclidean, denormalise_landmarks_relative

# ── Settings ──────────────────────────────────────────────────────────────────
PROJECT_DIR = Path("C:/Users/korbi/Desktop/A_Master_Thesis/")
FISH_DIR    = PROJECT_DIR / "rawdata"
TPS_FILE    = FISH_DIR / "landmark01.TPS"
MODEL_PATH  = PROJECT_DIR / "output/dl_hybrid/hybrid_model.pth"
OUTPUT_DIR  = PROJECT_DIR / "output/dl_hybrid"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def predict_new(img_path, tps_data, model_path, target_size=(270, 60)):
    """
    Run inference on a single image.
 
    Parameters
    ----------
    img_path    : Path  – path to the .jpg file
    tps_data    : dict  – parsed TPS data (from parse_tps)
    model_path  : str / Path – path to the saved .pth weights
    target_size : (width, height) of the letterbox canvas
 
    Returns
    -------
    np.ndarray of shape (10, 2) – predicted semilandmarks in pixel space
    """
    # 1. Setup Model
    model = HybridFishNet(output_dim=20)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 2. Preprocessing
    img_bgr = cv2.imread(str(img_path))
    img_h, img_w = img_bgr.shape[:2]
    data = tps_data[img_path.name]
    
    # Crop
    img_cropped, crop_box = crop_to_landmarks(img_bgr, data, img_h, img_w)
    
    # Letterbox (to 270x60)
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.float32)
    h, w = img_cropped.shape[:2]
    scale = min(target_size[0]/w, target_size[1]/h)
    resized = cv2.resize(img_cropped, (int(w*scale), int(h*scale)))
    canvas[:resized.shape[0], :resized.shape[1]] = resized.astype(np.float32) / 255.0
    
    # 3. Prepare Anchors (Landmarks 1 & 13)
    lmarks_px = flip_y(data.get("landmarks", []), img_h)
    # Match the normalization logic used in build_dataset_hybrid
    min_x, max_x, min_y, max_y = crop_box
    anchors = [
        ((lmarks_px[0][0] - min_x) / (max_x - min_x), (lmarks_px[0][1] - min_y) / (max_y - min_y)),
        ((lmarks_px[12][0] - min_x) / (max_x - min_x), (lmarks_px[12][1] - min_y) / (max_y - min_y))
    ]
    
    # 4. Run Inference
    img_tensor = torch.FloatTensor(canvas).unsqueeze(0)
    anchor_tensor = torch.FloatTensor(np.array(anchors).flatten()).unsqueeze(0)
    
    with torch.no_grad():
        pred = model(img_tensor, anchor_tensor)
        
    return pred.numpy().reshape(10, 2)

if __name__ == "__main__":
    import os
 
    tps_data    = parse_tps(TPS_FILE)
    fish_images = sorted(f for f in os.listdir(FISH_DIR) if f.lower().endswith(".jpg"))
 
    # Use every 4th image as the validation set (matches DL_learning.py split)
    fish_np  = np.array(fish_images)
    val_np   = fish_np[np.arange(len(fish_np)) % 4 == 3]    # Is currently set to the validation dataset; adapt as needed
 
    rows = []
    for img_name in val_np:
        img_path = FISH_DIR / img_name
        if img_path.name not in tps_data:
            print(f"  Skipping {img_name} – not found in TPS data")
            continue
 
        preds_relative = predict_new(img_path, tps_data, MODEL_PATH)
 
        # Denormalise back to pixel space for evaluation
        img_h      = cv2.imread(str(img_path)).shape[0]
        lmarks_full = flip_y(tps_data[img_name]["landmarks"],            img_h)
        gt_full     = flip_y(tps_data[img_name]["semi_landmarks"][0],    img_h)
 
        pts_pred = denormalise_landmarks_relative(preds_relative, lmarks_full)
 
        for j, (p, g) in enumerate(zip(pts_pred, gt_full)):
            rows.append({
                "img_name":        img_name,
                "index":           j,
                "X_new":           round(p[0], 5),
                "Y_new":           round(p[1], 5),
                "X_old":           round(g[0], 5),
                "Y_old":           round(g[1], 5),
                "dist_Old_to_New": round(euclidean(p, g), 5),
            })
 
    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_DIR / "hybrid_dl_results.csv", index=False)

    dist = results_df["dist_Old_to_New"]
    summary_df = pd.DataFrame({
        "Metric": ["dist_Old_to_New"],
        "min":    [dist.min()],
        "max":    [dist.max()],
        "mean":   [dist.mean()],
        "median": [dist.median()],
    }).round(5)
    summary_df.to_csv(OUTPUT_DIR / "summary_metadata.csv", index=False)
