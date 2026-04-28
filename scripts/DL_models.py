# DL_models.py
import sys
import os
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import cv2
import numpy as np
import pandas as pd

from pipeline_helpers import KNNSemiLandmarkRegressor, SimpleFishMLP, HybridFishNet
from DL_models_learning import main as train_main, build_model, model_path_for, SUPPORTED_MODELS
from pipeline_helpers import (
    crop_to_landmarks, flip_y, parse_tps, euclidean,
    denormalise_landmarks_relative, parse_args, load_config,
)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_type: str, model_path: Path, output_dim: int = 20, k: int = 5):
    """
    Load a trained model from disk.

    KNN models are stored as pickle files (.pkl); PyTorch models as state-dicts
    (.pth). The correct format is inferred from model_type.

    Parameters
    ----------
    model_type : 'knn' | 'mlp' | 'hybrid'
    model_path : path to the saved file
    output_dim : output dimension for PyTorch models
    k          : number of neighbours (KNN only)

    Returns
    -------
    Loaded model, ready for inference.
    """
    model_type = model_type.lower()
    if model_type == "knn":
        with open(model_path, "rb") as f:
            return pickle.load(f)
    else:
        model = build_model(model_type, output_dim=output_dim, k=k)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model


# ── Per-image preprocessing ───────────────────────────────────────────────────

def _preprocess_image(img_path: Path, tps_data: dict, target_size=(270, 60)):
    """
    Shared preprocessing: read → crop → letterbox → normalise anchors.

    Returns
    -------
    canvas      : (H, W, 3) float32 in [0, 1]
    anchors_arr : (4,) float32  – [x1_n, y1_n, x13_n, y13_n]
    crop_box    : (min_x, max_x, min_y, max_y)
    lmarks_px   : list of (x, y) in pixel space (y-flipped)
    img_h       : original image height (needed for denormalisation)
    """
    img_bgr = cv2.imread(str(img_path))
    img_h, img_w = img_bgr.shape[:2]
    data = tps_data[img_path.name]

    # Crop to bounding box of all landmarks
    img_cropped, crop_box = crop_to_landmarks(img_bgr, data, img_h, img_w)

    # Letterbox to target_size
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.float32)
    h, w   = img_cropped.shape[:2]
    scale  = min(target_size[0] / w, target_size[1] / h)
    resized = cv2.resize(img_cropped, (int(w * scale), int(h * scale)))
    canvas[: resized.shape[0], : resized.shape[1]] = resized.astype(np.float32) / 255.0

    # Anchor coords (landmarks 1 & 13) normalised to crop box
    lmarks_px = flip_y(data.get("landmarks", []), img_h)
    min_x, max_x, min_y, max_y = crop_box
    anchors_arr = np.array([
        (lmarks_px[0][0]  - min_x) / (max_x - min_x),
        (lmarks_px[0][1]  - min_y) / (max_y - min_y),
        (lmarks_px[12][0] - min_x) / (max_x - min_x),
        (lmarks_px[12][1] - min_y) / (max_y - min_y),
    ], dtype=np.float32)

    return canvas, anchors_arr, crop_box, lmarks_px, img_h


# ── Per-model inference ───────────────────────────────────────────────────────

def predict_new(
    img_path:    Path,
    tps_data:    dict,
    model,
    model_type:  str,
    target_size: tuple = (270, 60),
) -> np.ndarray:
    """
    Run inference on a single image with any supported model.

    Parameters
    ----------
    img_path   : path to the .jpg file
    tps_data   : parsed TPS data (from parse_tps)
    model      : a loaded model (KNNSemiLandmarkRegressor, SimpleFishMLP,
                 or HybridFishNet)
    model_type : 'knn' | 'mlp' | 'hybrid'
    target_size: (width, height) of the letterbox canvas

    Returns
    -------
    np.ndarray of shape (n_semi, 2) – predicted relative coords
    """
    canvas, anchors_arr, *_ = _preprocess_image(img_path, tps_data, target_size)
    model_type = model_type.lower()

    if model_type == "knn":
        # KNN operates directly on flattened pixel arrays
        pred, _, _ = model.predict(canvas)          # (n_semi*2,)
        return pred.reshape(-1, 2)

    elif model_type == "mlp":
        anchor_t = torch.FloatTensor(anchors_arr).unsqueeze(0)
        with torch.no_grad():
            pred = model(anchor_t)                  # (1, n_semi*2)
        return pred.numpy().reshape(-1, 2)

    elif model_type == "hybrid":
        img_t    = torch.FloatTensor(canvas).unsqueeze(0)
        anchor_t = torch.FloatTensor(anchors_arr).unsqueeze(0)
        with torch.no_grad():
            pred = model(img_t, anchor_t)           # (1, n_semi*2)
        return pred.numpy().reshape(-1, 2)

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {SUPPORTED_MODELS}")


# ── Evaluation loop ───────────────────────────────────────────────────────────

def run_evaluation(
    model,
    model_type: str,
    val_np:     np.ndarray,
    fish_dir:   Path,
    tps_data:   dict,
    output_dir: Path,
):
    """
    Predict semilandmarks for all validation images, compute errors, and save
    CSV results.

    Parameters
    ----------
    model      : loaded model
    model_type : 'knn' | 'mlp' | 'hybrid'
    val_np     : array of validation image filenames
    fish_dir   : directory containing the .jpg files
    tps_data   : parsed TPS data
    output_dir : where the CSVs are written
    """
    rows = []

    for img_name in val_np:
        img_path = fish_dir / img_name
        if img_path.name not in tps_data:
            print(f"  Skipping {img_name} – not found in TPS data")
            continue

        preds_relative = predict_new(img_path, tps_data, model, model_type)

        # Denormalise back to pixel space
        img_h       = cv2.imread(str(img_path)).shape[0]
        lmarks_full = flip_y(tps_data[img_name]["landmarks"],         img_h)
        gt_full     = flip_y(tps_data[img_name]["semi_landmarks"][0], img_h)
        pts_pred    = denormalise_landmarks_relative(preds_relative, lmarks_full)

        for j, (p, g) in enumerate(zip(pts_pred, gt_full)):
            rows.append({
                "img_name":        img_name,
                "model_type":      model_type,
                "index":           j,
                "X_new":           round(p[0], 5),
                "Y_new":           round(p[1], 5),
                "X_old":           round(g[0], 5),
                "Y_old":           round(g[1], 5),
                "dist_Old_to_New": round(euclidean(p, g), 5),
            })

    results_df = pd.DataFrame(rows)
    results_csv = output_dir / f"{model_type}_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"  Results saved → {results_csv}")

    dist = results_df["dist_Old_to_New"]
    summary_df = pd.DataFrame({
        "Metric": ["dist_Old_to_New"],
        "min":    [dist.min()],
        "max":    [dist.max()],
        "mean":   [dist.mean()],
        "median": [dist.median()],
    }).round(5)
    summary_csv = output_dir / f"{model_type}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Summary saved → {summary_csv}")
    print(f"\n  mean error: {dist.mean():.4f} px  |  median: {dist.median():.4f} px")

    return results_df, summary_df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)

    # ── Resolve paths ──────────────────────────────────────────────────────
    PROJECT_DIR = Path(cfg.get("PROJECT_DIR", "C:/Users/korbi/Desktop/A_Master_Thesis/"))
    FISH_DIR    = Path(cfg.get("FISH_DIR",    PROJECT_DIR / "rawdata"))
    TPS_FILE    = Path(cfg.get("TPS_FILE",    FISH_DIR / "landmark01.TPS"))
    OUTPUT_DIR  = Path(cfg.get("OUTPUT_DIR",  PROJECT_DIR / "output/tmp"))
    MODEL_DIR   = Path(cfg.get("DL_MODEL_DIR", PROJECT_DIR / "DL_model/"))
    OUTPUT_DIR.mkdir(parents=False, exist_ok=True)
    MODEL_DIR.mkdir(parents=False,  exist_ok=True)

    # ── Model selection ────────────────────────────────────────────────────
    # Set "model_type" in your YAML/JSON config to one of: knn | mlp | hybrid
    model_type = cfg.get("model_type", "hybrid").lower()
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Config 'model_type' must be one of {SUPPORTED_MODELS}")

    model_path = model_path_for(MODEL_DIR, model_type)

    # ── Auto-train if no model file found ─────────────────────────────────
    if not model_path.exists():
        print(f"No trained {model_type.upper()} model found at {model_path}.")
        print("Starting training via DL_learning.py …\n")
        model_path = train_main(
            model_type  = model_type,
            output_dir  = MODEL_DIR,
            fish_dir    = FISH_DIR,
            tps_file    = TPS_FILE,
            epochs      = cfg.get("epochs",     200),
            batch_size  = cfg.get("batch_size", 16),
            lr          = cfg.get("lr",         0.0005),
            k           = cfg.get("k",          5),
        )
    else:
        print(f"Found trained {model_type.upper()} model at {model_path}.")

    # ── Load model ─────────────────────────────────────────────────────────
    model = load_model(model_type, model_path)
    print(f"Model loaded: {model_type.upper()}")

    # ── Validation split (every 4th image) ────────────────────────────────
    tps_data    = parse_tps(TPS_FILE)
    fish_images = sorted(
        f for f in os.listdir(FISH_DIR)
        if f.lower().endswith(".jpg") and f in tps_data
    )
    fish_np = np.array(fish_images)
    val_np  = fish_np[np.arange(len(fish_np)) % 4 == 3]

    # ── Run evaluation ─────────────────────────────────────────────────────
    print(f"\nRunning evaluation on {len(val_np)} validation images …")
    run_evaluation(model, model_type, val_np, FISH_DIR, tps_data, OUTPUT_DIR)
