# DL_models_learning.py
# ---------------------------------------------------------------------------
# Training backbone for every model in the comparison.
#   knn | mlp | cnn | gnn | vit       ("hybrid" is accepted as an alias for cnn)
#
# Model architectures come from fish_dl_models.py; KNN comes from
# pipeline_helpers. Dataset builders are reused as-is:
#     mlp, gnn  -> build_dataset_mlp     (anchors + targets)
#     cnn, vit  -> build_dataset_hybrid  (image + anchors + targets)
#     knn       -> build_dataset_hybrid  (images used directly)
# ---------------------------------------------------------------------------

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

from pipeline_helpers import KNNSemiLandmarkRegressor
from pipeline_helpers import parse_tps, build_dataset_hybrid, build_dataset_mlp
from pipeline_helpers import SimpleFishMLP, HybridFishNet, FishGNN, FishViT

# ── Defaults ───────────────────────────────────────────────────────────────
EPOCHS        = 200
BATCH_SIZE    = 16
LEARNING_RATE = 0.0005
PROJECT_DIR   = Path("C:/Users/korbi/Desktop/A_Master_Thesis/")
FISH_DIR      = PROJECT_DIR / "rawdata"
TPS_FILE      = FISH_DIR   / "landmark01.TPS"

# Canonical names. "hybrid" maps to "cnn" for backward compatibility.
SUPPORTED_MODELS = ("knn", "mlp", "cnn", "gnn", "vit")
_ALIASES = {"hybrid": "cnn"}

# Which models consume the image vs anchors only.
_IMAGE_MODELS  = {"cnn", "vit"}
_ANCHOR_MODELS = {"mlp", "gnn"}


def canonical(model_type: str) -> str:
    """Normalise a model_type string (lowercase + resolve aliases)."""
    mt = model_type.lower()
    mt = _ALIASES.get(mt, mt)
    if mt not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from {SUPPORTED_MODELS} "
                         f"(or aliases {list(_ALIASES)}).")
    return mt


# ── Model factory ───────────────────────────────────────────────────────────
def build_model(model_type: str, output_dim: int = 20, k: int = 5, **arch_kwargs):
    """Instantiate the requested model. arch_kwargs are forwarded to gnn/vit."""
    mt = canonical(model_type)
    if mt == "knn":
        return KNNSemiLandmarkRegressor(k=k)
    if mt == "mlp":
        return SimpleFishMLP(input_dim=4, output_dim=output_dim)
    if mt == "cnn":
        return HybridFishNet(output_dim=output_dim)
    if mt == "gnn":
        return FishGNN(output_dim=output_dim, **arch_kwargs)
    if mt == "vit":
        return FishViT(output_dim=output_dim, **arch_kwargs)
    raise ValueError(f"Unhandled model_type '{mt}'")  # unreachable


def model_path_for(output_dir: Path, model_type: str) -> Path:
    """Canonical save-path: KNN -> .pkl, torch models -> .pth."""
    mt  = canonical(model_type)
    ext = ".pkl" if mt == "knn" else ".pth"
    return output_dir / f"{mt}_model{ext}"


# ── Training routines ────────────────────────────────────────────────────────
def _train_knn(model, fish_np, fish_dir, tps_data, mask):
    train_np = fish_np[~mask]
    x_img, _, y, _ = build_dataset_hybrid(train_np, fish_dir, tps_data)
    model.train(x_img, y.reshape(y.shape[0], -1))
    print(f"  KNN fitted on {len(train_np)} training images.")
    return model, []


def _train_torch(model, model_type, fish_np, fish_dir, tps_data, mask,
                 epochs, batch_size, lr):
    """Shared PyTorch loop for mlp / cnn / gnn / vit."""
    mt = canonical(model_type)
    train_np = fish_np[~mask]

    if mt in _ANCHOR_MODELS:                      # mlp, gnn -> anchors only
        x_anch, y, _ = build_dataset_mlp(train_np, fish_dir, tps_data)
        X_t = torch.FloatTensor(x_anch)
        Y_t = torch.FloatTensor(y.reshape(y.shape[0], -1))
        dataset = TensorDataset(X_t, Y_t)

        def forward_fn(batch):
            b_anch, b_y = batch
            return model(b_anch), b_y

    elif mt in _IMAGE_MODELS:                     # cnn, vit -> image + anchors
        x_img, x_anch, y, _ = build_dataset_hybrid(train_np, fish_dir, tps_data)
        X_img_t  = torch.FloatTensor(x_img)
        X_anch_t = torch.FloatTensor(x_anch)
        Y_t      = torch.FloatTensor(y.reshape(y.shape[0], -1))
        dataset  = TensorDataset(X_img_t, X_anch_t, Y_t)

        def forward_fn(batch):
            b_img, b_anch, b_y = batch
            return model(b_img, b_anch), b_y
    else:
        raise ValueError(f"'{mt}' is not a torch model")

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
            print(f"  Epoch {epoch + 1:>3}/{epochs}  Loss: {losses[-1]:.6f}")

    return model, losses


# ── Public entry point ───────────────────────────────────────────────────────
def main(model_type="cnn", output_dir=None, fish_dir=None, tps_file=None,
         epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
         k=5, output_dim=20, **arch_kwargs) -> Path:
    """Train the requested model, save it, and return the save path."""
    mt = canonical(model_type)

    fish_dir   = Path(fish_dir or FISH_DIR)
    tps_file   = Path(tps_file or TPS_FILE)
    output_dir = Path(output_dir or PROJECT_DIR / f"output/dl_{mt}")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path  = model_path_for(output_dir, mt)

    tps_data    = parse_tps(tps_file)
    fish_images = sorted(f for f in os.listdir(fish_dir) if f.lower().endswith(".jpg") and f in tps_data) #relieve! the and argument fixes the shifts since manual TPS curation 
    fish_np     = np.array(fish_images)
    mask        = np.arange(len(fish_np)) % 4 == 3            # every 4th -> validation

    print(f"\n[Training] model_type={mt}  |  train={fish_np[~mask].size}  val={fish_np[mask].size}")

    model = build_model(mt, output_dim=output_dim, k=k, **arch_kwargs)

    if mt == "knn":
        model, losses = _train_knn(model, fish_np, fish_dir, tps_data, mask)
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  KNN model saved -> {save_path}")
    else:
        model, losses = _train_torch(model, mt, fish_np, fish_dir, tps_data,
                                      mask, epochs, batch_size, lr)
        torch.save(model.state_dict(), save_path)
        print(f"  PyTorch model saved -> {save_path}")

    if losses:
        plt.figure()
        plt.plot(losses)
        plt.xlabel("Epoch"); plt.ylabel("Avg MSE Loss")
        plt.title(f"Training Loss – {mt.upper()}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{mt}_loss_plot.png")
        plt.close()

    return save_path


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a semilandmark-prediction model.")
    parser.add_argument("--model",      default="cnn", choices=list(SUPPORTED_MODELS) + list(_ALIASES))
    parser.add_argument("--epochs",     default=EPOCHS,        type=int)
    parser.add_argument("--batch_size", default=BATCH_SIZE,    type=int)
    parser.add_argument("--lr",         default=LEARNING_RATE, type=float)
    parser.add_argument("--k",          default=5,             type=int)
    args = parser.parse_args()
    main(model_type=args.model, epochs=args.epochs,
         batch_size=args.batch_size, lr=args.lr, k=args.k)
