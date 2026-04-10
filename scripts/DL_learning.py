import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from DL_model import HybridFishNet
from pipeline_helpers import parse_tps, build_dataset_hybrid

# ── Settings ──────────────────────────────────────────────────────────────────
EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 0.0005 # Slightly lower for hybrid models to stabilize
PROJECT_DIR = Path("C:/Users/korbi/Desktop/A_Master_Thesis/")
FISH_DIR    = PROJECT_DIR / "rawdata"
TPS_FILE    = FISH_DIR / "landmark01.TPS"
OUTPUT_DIR  = PROJECT_DIR / "output/dl_hybrid"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
# Load data
    tps_data = parse_tps(TPS_FILE)
    fish_images = sorted(f for f in os.listdir(FISH_DIR) if f.lower().endswith(".jpg"))

    # 1. Split & Load Data
    fish_np = np.array(fish_images)
    mask = np.arange(len(fish_np)) % 4 == 3
    train_np, val_np = fish_np[~mask], fish_np[mask]

    x_train_img, x_train_anch, y_train, _ = build_dataset_hybrid(train_np, FISH_DIR, tps_data)
    x_val_img, x_val_anch, y_val, val_names_ok = build_dataset_hybrid(val_np, FISH_DIR, tps_data)
    # convert to tensors
    X_train_img_t = torch.FloatTensor(x_train_img)
    X_train_anch_t = torch.FloatTensor(x_train_anch)
    Y_train_t = torch.FloatTensor(y_train.reshape(y_train.shape[0], -1))

    X_val_img_t = torch.FloatTensor(x_val_img)
    X_val_anch_t = torch.FloatTensor(x_val_anch)

    model = HybridFishNet(output_dim=20)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    losses = []

    dataset = TensorDataset(X_train_img_t, X_train_anch_t, Y_train_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0

        for b_img, b_anc, b_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(b_img, b_anc), b_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
        if (epoch+1) % 10 == 0: print(f"Epoch {epoch+1} Loss: {losses[-1]:.6f}")

    torch.save(model.state_dict(), OUTPUT_DIR / "hybrid_model.pth")
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Avg MSE Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_plot.png")