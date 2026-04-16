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

from pipeline_helpers import (
    parse_tps, build_dataset, flip_y, euclidean,
    denormalise_landmarks_relative
)

# ── Settings ──────────────────────────────────────────────────────────────────
EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
PROJECT_DIR = Path("C:/Users/korbi/Desktop/A_Master_Thesis/")
FISH_DIR    = PROJECT_DIR / "rawdata"
TPS_FILE    = FISH_DIR / "landmark01.TPS"
OUTPUT_DIR  = PROJECT_DIR / "output/dl_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model Architecture ────────────────────────────────────────────────────────
class SemiLandmarkNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SemiLandmarkNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_SIZE),
            nn.BatchNorm1d(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE * 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ── Execution ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tps_data = parse_tps(TPS_FILE)
    fish_images = sorted(f for f in os.listdir(FISH_DIR) if f.lower().endswith(".jpg"))

    # 1. Split & Load Data
    fish_np = np.array(fish_images)
    mask = np.arange(len(fish_np)) % 4 == 3
    train_np, val_np = fish_np[~mask], fish_np[mask]

    # Build using RelativeToLM coordinates
    x_train, _, y_train, _, _, _ = build_dataset(train_np, FISH_DIR, tps_data)
    x_val, _, y_val, val_boxes, val_lmarks_full, val_names_ok = build_dataset(val_np, FISH_DIR, tps_data)

    # Convert to Tensors
    X_train_t = torch.FloatTensor(x_train)
    Y_train_t = torch.FloatTensor(y_train.reshape(y_train.shape[0], -1)) # Flatten Y
    X_val_t = torch.FloatTensor(x_val)

    # 2. Training Loop
    dataset = TensorDataset(X_train_t, Y_train_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SemiLandmarkNet(X_train_t.shape[1], Y_train_t.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(f"Training Model for {EPOCHS} epochs...")
    model.train()
    for epoch in range(EPOCHS):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.6f}")

    # 3. Evaluation
    model.eval()
    with torch.no_grad():
        preds_raw = model(X_val_t).numpy()
    
    # Reshape back to (Samples, 10, 2)
    preds_relative = preds_raw.reshape(-1, 10, 2)

    rows_dl = []
    for i, img_name in enumerate(val_names_ok):
        img_h = cv2.imread(str(FISH_DIR / img_name)).shape[0]
        gt_full = flip_y(tps_data[img_name]["semi_landmarks"][0], img_h)
        
        # Denormalise DL prediction back to full image pixels
        pts_pred = denormalise_landmarks_relative(preds_relative[i], val_lmarks_full[i])

        for j, (p, g) in enumerate(zip(pts_pred, gt_full)):
            rows_dl.append({
                "img_name": img_name, "index": j,
                "X_new": round(p[0], 3), "Y_new": round(p[1], 3),
                "X_old": round(g[0], 3), "Y_old": round(g[1], 3),
                "dist_Old_to_New": round(euclidean(p, g), 3)
            })

    pd.DataFrame(rows_dl).to_csv(OUTPUT_DIR / "dl_results_relative.csv", index=False)
    print(f"Results saved to {OUTPUT_DIR}")