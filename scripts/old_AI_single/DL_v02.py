import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import cv2
from pathlib import Path

# Assuming these are in your pipeline_helpers/preprocessing.py now
from pipeline_helpers import (
    parse_tps, build_dataset_hybrid, flip_y, euclidean,
    denormalise_landmarks_relative
)

# ── Settings ──────────────────────────────────────────────────────────────────
EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 0.0005 # Slightly lower for hybrid models to stabilize
PROJECT_DIR = Path("C:/Users/korbi/Desktop/A_Master_Thesis/")
FISH_DIR    = PROJECT_DIR / "rawdata"
TPS_FILE    = FISH_DIR / "landmark01.TPS"
OUTPUT_DIR  = PROJECT_DIR / "output/dl_hybrid"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model Architecture ────────────────────────────────────────────────────────
class HybridFishNet(nn.Module):
    def __init__(self, output_dim, input_shape=(3, 60, 270)):
        super(HybridFishNet, self).__init__()
        
        # 1. Image Branch
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # -> 30x135
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # -> 15x67
            nn.Flatten()
        )
        
        # 2. Auto-calculate the flattening size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            cnn_out = self.cnn(dummy_input)
            self.cnn_flat_size = cnn_out.shape[1]
            
        # 3. Landmark Branch (Anchors: Points 1 & 13)
        self.anchor_mlp = nn.Sequential(
            nn.Linear(4, 16), 
            nn.ReLU()
        )

        # 4. Combined Head
        self.combined = nn.Sequential(
            nn.Linear(self.cnn_flat_size + 16, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim) 
        )

    def forward(self, img, anchors):
        img_features = self.cnn(img.permute(0, 3, 1, 2))
        anchor_features = self.anchor_mlp(anchors)
        x = torch.cat((img_features, anchor_features), dim=1)
        return self.combined(x)

# ── Execution ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tps_data = parse_tps(TPS_FILE)
    fish_images = sorted(f for f in os.listdir(FISH_DIR) if f.lower().endswith(".jpg"))

    # 1. Split & Load Data
    fish_np = np.array(fish_images)
    mask = np.arange(len(fish_np)) % 4 == 3
    train_np, val_np = fish_np[~mask], fish_np[mask]

    # Use the new hybrid dataset builder
    # Returning: images, anchors (LM1&13), labels (rel_coords), names
    x_train_img, x_train_anch, y_train, _ = build_dataset_hybrid(train_np, FISH_DIR, tps_data)
    x_val_img, x_val_anch, y_val, val_names_ok = build_dataset_hybrid(val_np, FISH_DIR, tps_data)

    # Convert to Tensors
    X_train_img_t = torch.FloatTensor(x_train_img)
    X_train_anch_t = torch.FloatTensor(x_train_anch)
    Y_train_t = torch.FloatTensor(y_train.reshape(y_train.shape[0], -1))

    X_val_img_t = torch.FloatTensor(x_val_img)
    X_val_anch_t = torch.FloatTensor(x_val_anch)

    # 2. Training Loop with Two Inputs
    # We pack both inputs into the dataset
    dataset = TensorDataset(X_train_img_t, X_train_anch_t, Y_train_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Output dim is 20 (10 pairs of x,y semilandmarks)
    model = HybridFishNet(output_dim=20)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(f"Training Hybrid Model for {EPOCHS} epochs...")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch_img, batch_anch, batch_y in loader:
            optimizer.zero_grad()
            # Pass both inputs to the model
            pred = model(batch_img, batch_anch)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.6f}")

    # 3. Evaluation
    model.eval()
    with torch.no_grad():
        preds_raw = model(X_val_img_t, X_val_anch_t).numpy()
    
    preds_relative = preds_raw.reshape(-1, 10, 2)

    # 4. Export Results
    rows_dl = []
    # Note: Need val_lmarks_full for denormalisation
    # If build_dataset_hybrid doesn't return it, you may need to grab it from tps_data directly
    for i, img_name in enumerate(val_names_ok):
        img_h = cv2.imread(str(FISH_DIR / img_name)).shape[0]
        gt_full = flip_y(tps_data[img_name]["semi_landmarks"][0], img_h)
        lmarks_full = flip_y(tps_data[img_name]["landmarks"], img_h)
        
        pts_pred = denormalise_landmarks_relative(preds_relative[i], lmarks_full)

        for j, (p, g) in enumerate(zip(pts_pred, gt_full)):
            rows_dl.append({
                "img_name": img_name, "index": j,
                "X_new": round(p[0], 3), "Y_new": round(p[1], 3),
                "X_old": round(g[0], 3), "Y_old": round(g[1], 3),
                "dist_Old_to_New": round(euclidean(p, g), 3)
            })

    pd.DataFrame(rows_dl).to_csv(OUTPUT_DIR / "hybrid_dl_results.csv", index=False)
    print(f"Results saved to {OUTPUT_DIR}")