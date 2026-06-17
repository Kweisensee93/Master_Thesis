# DL_models.py
# ---------------------------------------------------------------------------
# Unified evaluation driver. Reads a YAML/JSON config, picks a model_type,
# loads the trained model if it exists (else trains it via DL_models_learning),
# runs inference on the validation split, and writes comparable CSVs.
#
#   model_type in config: knn | mlp | cnn | gnn | vit   (hybrid == cnn)
#
# Output CSVs share the same schema across every approach so they can be
# pooled for the thesis comparison:
#   img_name, model_type, index, X_new, Y_new, X_old, Y_old, dist_Old_to_New
# ---------------------------------------------------------------------------

import sys
import os
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import cv2
import numpy as np
import pandas as pd

from pipeline_helpers import KNNSemiLandmarkRegressor  # noqa: F401 (used via pickle)
from fish_dl_models import SimpleFishMLP, HybridFishNet, FishGNN, FishViT  # noqa: F401
from DL_models_learning import (
    main as train_main, build_model, model_path_for,
    SUPPORTED_MODELS, canonical, _IMAGE_MODELS, _ANCHOR_MODELS,
)
from pipeline_helpers import (
    crop_to_landmarks, flip_y, parse_tps, euclidean,
    denormalise_landmarks_relative, parse_args, load_config,
)


# ── Model loading ────────────────────────────────────────────────────────────
def load_model(model_type, model_path, output_dim=20, k=5):
    """KNN -> unpickle; torch models -> build + load_state_dict + eval()."""
    mt = canonical(model_type)
    if mt == "knn":
        with open(model_path, "rb") as f:
            return pickle.load(f)
    model = build_model(mt, output_dim=output_dim, k=k)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


# ── Per-image preprocessing ──────────────────────────────────────────────────
def _preprocess_image(img_path, tps_data, target_size=(270, 60)):
    """read -> crop -> letterbox -> normalise anchors. Mirrors build_dataset_hybrid."""
    img_bgr = cv2.imread(str(img_path))
    img_h, img_w = img_bgr.shape[:2]
    data = tps_data[img_path.name]

    img_cropped, crop_box = crop_to_landmarks(img_bgr, data, img_h, img_w)

    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.float32)
    h, w   = img_cropped.shape[:2]
    scale  = min(target_size[0] / w, target_size[1] / h)
    resized = cv2.resize(img_cropped, (int(w * scale), int(h * scale)))
    canvas[: resized.shape[0], : resized.shape[1]] = resized.astype(np.float32) / 255.0

    lmarks_px = flip_y(data.get("landmarks", []), img_h)
    min_x, max_x, min_y, max_y = crop_box
    anchors_arr = np.array([
        (lmarks_px[0][0]  - min_x) / (max_x - min_x),
        (lmarks_px[0][1]  - min_y) / (max_y - min_y),
        (lmarks_px[12][0] - min_x) / (max_x - min_x),
        (lmarks_px[12][1] - min_y) / (max_y - min_y),
    ], dtype=np.float32)

    return canvas, anchors_arr, crop_box, lmarks_px, img_h


# ── Per-model inference ──────────────────────────────────────────────────────
def predict_new(img_path, tps_data, model, model_type, target_size=(270, 60)):
    """Run inference on one image; returns (n_semi, 2) relative coords."""
    canvas, anchors_arr, *_ = _preprocess_image(img_path, tps_data, target_size)
    mt = canonical(model_type)

    if mt == "knn":
        pred, _, _ = model.predict(canvas)
        return pred.reshape(-1, 2)

    anchor_t = torch.FloatTensor(anchors_arr).unsqueeze(0)
    with torch.no_grad():
        if mt in _ANCHOR_MODELS:                  # mlp, gnn
            pred = model(anchor_t)
        elif mt in _IMAGE_MODELS:                 # cnn, vit
            img_t = torch.FloatTensor(canvas).unsqueeze(0)
            pred = model(img_t, anchor_t)
        else:
            raise ValueError(f"Unhandled model_type '{mt}'")
    return pred.numpy().reshape(-1, 2)


# ── Evaluation loop ──────────────────────────────────────────────────────────
def run_evaluation(model, model_type, val_np, fish_dir, tps_data, output_dir):
    mt = canonical(model_type)
    rows = []

    for img_name in val_np:
        img_path = fish_dir / img_name
        if img_path.name not in tps_data:
            print(f"  Skipping {img_name} – not found in TPS data")
            continue

        preds_relative = predict_new(img_path, tps_data, model, mt)

        img_h       = cv2.imread(str(img_path)).shape[0]
        lmarks_full = flip_y(tps_data[img_name]["landmarks"],         img_h)
        gt_full     = flip_y(tps_data[img_name]["semi_landmarks"][0], img_h)
        pts_pred    = denormalise_landmarks_relative(preds_relative, lmarks_full)

        for j, (p, g) in enumerate(zip(pts_pred, gt_full)):
            rows.append({
                "img_name":        img_name,
                "model_type":      mt,
                "index":           j,
                "X_new":           round(p[0], 5),
                "Y_new":           round(p[1], 5),
                "X_old":           round(g[0], 5),
                "Y_old":           round(g[1], 5),
                "dist_Old_to_New": round(euclidean(p, g), 5),
            })

    results_df  = pd.DataFrame(rows)
    results_csv = output_dir / f"{mt}_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"  Results saved -> {results_csv}")

    dist = results_df["dist_Old_to_New"]
    summary_df = pd.DataFrame({
        "Metric": ["dist_Old_to_New"],
        "min":    [dist.min()],
        "max":    [dist.max()],
        "mean":   [dist.mean()],
        "median": [dist.median()],
    }).round(5)
    summary_csv = output_dir / f"{mt}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Summary saved -> {summary_csv}")
    print(f"\n  mean error: {dist.mean():.4f} px  |  median: {dist.median():.4f} px")

    return results_df, summary_df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)

    PROJECT_DIR = Path(cfg.get("PROJECT_DIR", "C:/Users/korbi/Desktop/A_Master_Thesis/"))
    FISH_DIR    = Path(cfg.get("FISH_DIR",    PROJECT_DIR / "rawdata"))
    TPS_FILE    = Path(cfg.get("TPS_FILE",    FISH_DIR / "landmark01.TPS"))
    OUTPUT_DIR  = Path(cfg.get("OUTPUT_DIR",  PROJECT_DIR / "output/tmp"))
    MODEL_DIR   = Path(cfg.get("DL_MODEL_DIR", PROJECT_DIR / "DL_model/"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True,  exist_ok=True)

    # Set "model_type" in your YAML/JSON config: knn | mlp | cnn | gnn | vit
    model_type = canonical(cfg.get("model_type", "cnn"))
    model_path = model_path_for(MODEL_DIR, model_type)

    # ── Auto-train if no model file found ─────────────────────────────────
    if not model_path.exists():
        print(f"No trained {model_type.upper()} model found at {model_path}.")
        print("Starting training …\n")
        model_path = train_main(
            model_type = model_type,
            output_dir = MODEL_DIR,
            fish_dir   = FISH_DIR,
            tps_file   = TPS_FILE,
            epochs     = cfg.get("epochs",     200),
            batch_size = cfg.get("batch_size", 16),
            lr         = cfg.get("lr",         0.0005),
            k          = cfg.get("k",          5),
        )
    else:
        print(f"Found trained {model_type.upper()} model at {model_path}.")

    model = load_model(model_type, model_path)
    print(f"Model loaded: {model_type.upper()}")

    tps_data    = parse_tps(TPS_FILE)
    fish_images = sorted(
        f for f in os.listdir(FISH_DIR)
        if f.lower().endswith(".jpg") and f in tps_data
    )
    fish_np = np.array(fish_images)
    val_np  = fish_np[np.arange(len(fish_np)) % 4 == 3]

    print(f"\nRunning evaluation on {len(val_np)} validation images …")
    run_evaluation(model, model_type, val_np, FISH_DIR, tps_data, OUTPUT_DIR)