# DL_models_learning.py
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from pipeline_helpers import KNNSemiLandmarkRegressor, SimpleFishMLP, HybridFishNet
from pipeline_helpers import parse_tps, build_dataset_hybrid, build_dataset_mlp

# ── Defaults ──────────────────────────────────────────────────────────────────
EPOCHS        = 200
BATCH_SIZE    = 16
LEARNING_RATE = 0.0005
PROJECT_DIR   = Path("C:/Users/korbi/Desktop/A_Master_Thesis/")
FISH_DIR      = PROJECT_DIR / "rawdata"
TPS_FILE      = FISH_DIR   / "landmark01.TPS"

# ── Model factory ─────────────────────────────────────────────────────────────

SUPPORTED_MODELS = ("knn", "mlp", "hybrid")

def build_model(model_type: str, output_dim: int = 20, k: int = 5):
    """
    Instantiate and return the requested model.

    Parameters
    ----------
    model_type : one of 'knn' | 'mlp' | 'hybrid'
    output_dim : number of output values (n_semi * 2)
    k          : neighbours for KNN

    Returns
    -------
    model instance (KNNSemiLandmarkRegressor, SimpleFishMLP, or HybridFishNet)
    """
    model_type = model_type.lower()
    if model_type == "knn":
        return KNNSemiLandmarkRegressor(k=k)
    elif model_type == "mlp":
        return SimpleFishMLP(input_dim=4, output_dim=output_dim)
    elif model_type == "hybrid":
        return HybridFishNet(output_dim=output_dim)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from: {SUPPORTED_MODELS}"
        )


def model_path_for(output_dir: Path, model_type: str) -> Path:
    """Return the canonical save-path for a given model type."""
    ext = ".pkl" if model_type == "knn" else ".pth"
    return output_dir / f"{model_type}_model{ext}"


# ── Training routines ─────────────────────────────────────────────────────────

def _train_knn(model, fish_np, fish_dir, tps_data, mask):
    """Train (fit) the KNN regressor on the training split."""
    train_np = fish_np[~mask]
    # KNN works on raw images; reuse the hybrid builder for images + targets
    x_img, _, y, _ = build_dataset_hybrid(train_np, fish_dir, tps_data)
    model.train(x_img, y.reshape(y.shape[0], -1))
    print(f"  KNN fitted on {len(train_np)} training images.")
    return model, []          # no per-epoch losses


def _train_torch(
    model,
    model_type: str,
    fish_np,
    fish_dir,
    tps_data,
    mask,
    epochs: int,
    batch_size: int,
    lr: float,
):
    """
    Generic PyTorch training loop shared by MLP and HybridFishNet.

    The MLP receives only anchor coordinates; HybridFishNet receives both
    the cropped image and the anchors.  The dataset builder is selected
    accordingly.
    """
    train_np = fish_np[~mask]

    # ── Build tensors ──────────────────────────────────────────────────────
    if model_type == "mlp":
        x_anch, y, _ = build_dataset_mlp(train_np, fish_dir, tps_data)
        X_t = torch.FloatTensor(x_anch)
        Y_t = torch.FloatTensor(y.reshape(y.shape[0], -1))
        dataset = TensorDataset(X_t, Y_t)

        def forward_fn(batch):
            b_anch, b_y = batch
            return model(b_anch), b_y

    else:  # hybrid
        x_img, x_anch, y, _ = build_dataset_hybrid(train_np, fish_dir, tps_data)
        X_img_t  = torch.FloatTensor(x_img)
        X_anch_t = torch.FloatTensor(x_anch)
        Y_t      = torch.FloatTensor(y.reshape(y.shape[0], -1))
        dataset  = TensorDataset(X_img_t, X_anch_t, Y_t)

        def forward_fn(batch):
            b_img, b_anch, b_y = batch
            return model(b_img, b_anch), b_y

    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses    = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            pred, target = forward_fn(batch)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:>3}/{epochs}  Loss: {losses[-1]:.6f}")

    return model, losses


# ── Public entry point ────────────────────────────────────────────────────────

def main(
    model_type: str   = "hybrid",
    output_dir: Path  = None,
    fish_dir:   Path  = None,
    tps_file:   Path  = None,
    epochs:     int   = EPOCHS,
    batch_size: int   = BATCH_SIZE,
    lr:         float = LEARNING_RATE,
    k:          int   = 5,
    output_dim: int   = 20,
) -> Path:
    """
    Train the requested model and save it.

    Parameters
    ----------
    model_type  : 'knn' | 'mlp' | 'hybrid'
    output_dir  : directory where the model file and plots are saved
    fish_dir    : directory containing .jpg images
    tps_file    : path to the .TPS landmark file
    epochs      : training epochs (ignored for KNN)
    batch_size  : mini-batch size (ignored for KNN)
    lr          : Adam learning rate (ignored for KNN)
    k           : number of neighbours for KNN (ignored otherwise)
    output_dim  : number of predicted coordinates (n_semi * 2)

    Returns
    -------
    Path to the saved model file.
    """
    model_type = model_type.lower()
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"model_type must be one of {SUPPORTED_MODELS}, got '{model_type}'")

    # ── Resolve paths ──────────────────────────────────────────────────────
    fish_dir   = Path(fish_dir  or FISH_DIR)
    tps_file   = Path(tps_file  or TPS_FILE)
    output_dir = Path(output_dir or PROJECT_DIR / f"output/dl_{model_type}")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path  = model_path_for(output_dir, model_type)

    # ── Data split ─────────────────────────────────────────────────────────
    tps_data    = parse_tps(tps_file)
    fish_images = sorted(f for f in os.listdir(fish_dir) if f.lower().endswith(".jpg"))
    fish_np     = np.array(fish_images)
    mask        = np.arange(len(fish_np)) % 4 == 3   # every 4th → validation

    print(f"\n[Training] model_type={model_type}  |  "
          f"train={fish_np[~mask].size}  val={fish_np[mask].size}")

    # ── Build & train ──────────────────────────────────────────────────────
    model = build_model(model_type, output_dim=output_dim, k=k)

    if model_type == "knn":
        model, losses = _train_knn(model, fish_np, fish_dir, tps_data, mask)
    else:
        model, losses = _train_torch(
            model, model_type, fish_np, fish_dir, tps_data, mask,
            epochs, batch_size, lr,
        )

    # ── Save ───────────────────────────────────────────────────────────────
    if model_type == "knn":
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  KNN model saved → {save_path}")
    else:
        torch.save(model.state_dict(), save_path)
        print(f"  PyTorch model saved → {save_path}")

    # ── Loss plot (PyTorch models only) ───────────────────────────────────
    if losses:
        plt.figure()
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Avg MSE Loss")
        plt.title(f"Training Loss – {model_type.upper()}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_type}_loss_plot.png")
        plt.close()

    return save_path


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a semilandmark prediction model.")
    parser.add_argument("--model",      default="hybrid",  choices=SUPPORTED_MODELS)
    parser.add_argument("--epochs",     default=EPOCHS,    type=int)
    parser.add_argument("--batch_size", default=BATCH_SIZE, type=int)
    parser.add_argument("--lr",         default=LEARNING_RATE, type=float)
    parser.add_argument("--k",          default=5,         type=int)
    args = parser.parse_args()

    main(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        k=args.k,
    )
