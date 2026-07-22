import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
import numpy as np
import pandas as pd

def parse_args():
    """Same CLI contract as the other scripts: one optional positional config."""
    parser = argparse.ArgumentParser(
        description="Adapt rembg output into the AI/ML CSV schema.")
    parser.add_argument("config", type=Path, nargs="?", default=None,
                        help="Path to config file (.yaml or .json).")
    return parser.parse_args()

def load_config(config_path):
    """Mirror pipeline_helpers.data_io.load_config (.yaml/.json, empty if None)."""
    if config_path is None:
        return {}
    suffix = Path(config_path).suffix.lower()
    with open(config_path, "r") as f:
        if suffix in (".yaml", ".yml"):
            import yaml
            return yaml.safe_load(f)
        elif suffix == ".json":
            import json
            return json.load(f)
        raise ValueError(f"Unsupported config format: '{suffix}'. Use .yaml or .json.")

def euclidean_rows(new_xy, old_xy):
    """Row-wise Euclidean distance, matching pipeline_helpers.geometry.euclidean
    (float32 accumulation) so numbers are identical to the AI/ML driver."""
    a = np.asarray(new_xy, dtype=np.float32)
    b = np.asarray(old_xy, dtype=np.float32)
    diff = a - b
    return np.sqrt((diff ** 2).sum(axis=1))

def compute_validation_split(fish_dir: Path, tps_file: Path) -> set:
    """Reproduce DL_models_run_v01's validation split EXACTLY.
 
    val = sorted(.jpg files present in the TPS) then every 4th (index %% 4 == 3).
    Returns a set of .jpg filenames.
    """
    # Imported lazily so the core transform never pulls in the heavy package.
    from pipeline_helpers import parse_tps
    tps_data = parse_tps(tps_file)
    fish_images = sorted(
        f for f in os.listdir(fish_dir)
        if f.lower().endswith(".jpg") and f in tps_data
    )
    fish_np = np.array(fish_images)
    if len(fish_np) == 0:
        return set()
    return set(fish_np[np.arange(len(fish_np)) % 4 == 3].tolist())

def main():
    args = parse_args()
    cfg = load_config(args.config)
 
    PROJECT_DIR = Path(cfg.get("PROJECT_DIR", "C:/Users/korbi/Desktop/A_Master_Thesis/"))
    OUTPUT_DIR  = Path(cfg.get("OUTPUT_DIR",  PROJECT_DIR / "output/tmp"))
    ADAPTER_OUT = Path(cfg.get("REMBG_ADAPTER_OUTPUT_DIR", OUTPUT_DIR / "summary"))
    ADAPTER_OUT.mkdir(parents=True, exist_ok=True)
 
    method       = str(cfg.get("rembg_method_name", "rembg"))
    master_name  = str(cfg.get("rembg_master_csv", "master_semilandmark_metrics.csv"))
    val_only     = bool(cfg.get("rembg_adapter_val_only", False))
    drop_no_gt   = bool(cfg.get("rembg_adapter_drop_no_gt", True))
 
    master_csv = OUTPUT_DIR / master_name
    if not master_csv.exists():
        raise FileNotFoundError(
            f"Could not find rembg master CSV at '{master_csv}'. "
            f"Run rembg_v08.py first, or set 'rembg_master_csv' / 'OUTPUT_DIR' in the config."
        )
 
    print(f"Reading rembg master CSV : {master_csv}")
    df = pd.read_csv(master_csv)
 
    required = {"ImageName", "Index", "new_x", "new_y", "old_x", "old_y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"'{master_csv}' is missing expected columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )
 
    n_total = len(df)
 
    # ── Optional: restrict to the DL validation split for a like-for-like comparison ──
    if val_only:
        FISH_DIR = Path(cfg.get("FISH_DIR", PROJECT_DIR / "rawdata"))
        TPS_FILE = Path(cfg.get("TPS_FILE", FISH_DIR / "landmark01.TPS"))
        try:
            val_set = compute_validation_split(FISH_DIR, TPS_FILE)
            before = len(df)
            df = df[df["ImageName"].isin(val_set)].copy()
            print(f"  val-only filter        : kept {len(df)}/{before} rows "
                  f"({df['ImageName'].nunique()} images in the validation split)")
        except Exception as e:  # noqa: BLE001 - never fail the whole run on the optional filter
            print(f"  WARNING: could not build validation split ({e}). "
                  f"Proceeding with ALL images.")
 
    # ── Ground-truth handling ─────────────────────────────────────────────────
    # The DL driver only ever emits rows that have a ground-truth semilandmark.
    # rembg can carry rows with no GT (old_x/old_y == NaN). Drop them by default
    # so the distance statistics match DL semantics.
    if drop_no_gt:
        before = len(df)
        df = df.dropna(subset=["old_x", "old_y"]).copy()
        dropped = before - len(df)
        if dropped:
            print(f"  dropped no-GT points   : {dropped}")
 
    if df.empty:
        raise ValueError(
            "No rows left after filtering (val-only / drop-no-GT). "
            "Nothing to write - check the config flags and the master CSV."
        )
 
    # ── Per-point distance: same definition as DL (euclidean(pred, gt)) ────────
    new_xy = df[["new_x", "new_y"]].to_numpy(dtype=np.float64)
    old_xy = df[["old_x", "old_y"]].to_numpy(dtype=np.float64)
    dist = euclidean_rows(new_xy, old_xy)
 
    # ── Build the results frame in the DL column order/naming ──────────────────
    results_df = pd.DataFrame({
        "img_name":        df["ImageName"].to_numpy(),
        "model_type":      method,
        "index":           df["Index"].to_numpy(),
        "X_new":           np.round(new_xy[:, 0], 5),
        "Y_new":           np.round(new_xy[:, 1], 5),
        "X_old":           np.round(old_xy[:, 0], 5),
        "Y_old":           np.round(old_xy[:, 1], 5),
        "dist_Old_to_New": np.round(dist, 5),
    })
 
    results_csv = ADAPTER_OUT / f"{method}_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"  Results saved -> {results_csv}  ({len(results_df)} rows, "
          f"{results_df['img_name'].nunique()} images)")
 
    # ── Summary: identical schema to DL (Metric, min, max, mean, median) ───────
    d = results_df["dist_Old_to_New"]
    summary_df = pd.DataFrame({
        "Metric": ["dist_Old_to_New"],
        "min":    [d.min()],
        "max":    [d.max()],
        "mean":   [d.mean()],
        "median": [d.median()],
    }).round(5)
 
    summary_csv = ADAPTER_OUT / f"{method}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Summary saved -> {summary_csv}")
    print(f"\n  [{method}] processed {n_total} master rows -> "
          f"mean error: {d.mean():.4f} px  |  median: {d.median():.4f} px")
 
 
if __name__ == "__main__":
    main()