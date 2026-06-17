#models.py
# The Thesis developed more towards a comparison of different approaches / models
# Therefore, the models are stored within this file

# So far
# 1. KNN regressor
# 2. FNN / MLP (SimpleFishMLP)
# 3. Hybrid CNN + MLP (HybridFishNet)
# 4. GCN (Graph Convolutional Network) (FishGNN)
# 5. ViT (Vision Transformer) (FishViT)
# Options:
# U-Net (stacked hourglass); GCN (Graph Convolutional Network); ViT (Vision Transformer)

import torch
import torch.nn as nn
import numpy as np

class KNNSemiLandmarkRegressor:
    """
    K-nearest-neighbour regressor for semi-landmark prediction (default k=5).

    Extends the 1-NN approach from the cats-vs-dogs notebook to K neighbours:
    - Training  : memorise all (image, semi_landmark_coords) pairs  ← identical
    - Prediction: find the K closest training images by mean absolute pixel
                  distance, then return the MEAN of their semi-landmark
                  coordinate vectors as the prediction.

    Why averaging instead of majority vote?
        Semi-landmark coordinates are continuous values, not class labels.
        Averaging the K neighbours smooths out outliers – if one of the K
        nearest images had a slightly unusual curve placement, the other four
        pull the prediction back toward a sensible position.  With k=1 a
        single atypical training specimen would dominate the prediction entirely.

    Because semi-landmark coordinates are stored in the LM1→LM13 relative
    coordinate system (pose-invariant), averaging neighbours produces
    meaningful results even when fish appear at different positions/scales/
    orientations across images.
    For readability, this custom regressor is implemented. Later on we may use:
    from sklearn.neighbors import KNeighborsRegressor
    """

    def __init__(self, k: int = 5):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k             = k
        self.train_images: np.ndarray | None = None   # (N, H, W, C) float32
        self.train_semi:   np.ndarray | None = None   # (N, n_semi*2) float32

    def train(
        self,
        images: np.ndarray,
        semi_landmark_coords: np.ndarray,
    ) -> None:
        """
        Store training images and their normalised semi-landmark coordinates.

        Parameters
        ----------
        images               : (N, H, W, C) float32 array of cropped images
        semi_landmark_coords : (N, n_semi*2) float32 array – each row is the
                               flat relative [t0,d0,t1,d1,…] vector for one
                               specimen, produced by normalise_landmarks_relative()
        """
        if len(images) < self.k:
            raise ValueError(
                f"Training set ({len(images)} images) is smaller than k={self.k}."
            )
        self.train_images = images
        self.train_semi   = semi_landmark_coords

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict semi-landmark coordinates for a single query image.

        Ranks all training images by mean absolute pixel distance, selects the
        K closest, and returns the unweighted mean of their semi-landmark
        coordinate vectors.

        Parameters
        ----------
        image : (H, W, C) float32 array – one cropped query image

        Returns
        -------
        predicted_semi  : (n_semi*2,) float32 – mean relative coords of K neighbours
        k_distances     : (k,) float32        – MAD distances to each neighbour
        k_indices       : (k,) int            – indices of the K neighbours in
                                                the training set, nearest first
        """
        # Vectorised MAD across all training images
        distances = np.mean(np.abs(self.train_images - image), axis=(1, 2, 3))

        # Indices of the k smallest distances, sorted nearest-first
        k_indices   = np.argsort(distances)[: self.k]
        k_distances = distances[k_indices]

        # Average the semi-landmark vectors of the K neighbours
        # Shape: (k, n_semi*2) → mean over axis 0 → (n_semi*2,)
        predicted_semi = self.train_semi[k_indices].mean(axis=0)

        return predicted_semi, k_distances, k_indices
    
class SimpleFishMLP(nn.Module):
    # To be used as a simple FEED FORWARD DL approach:
    # "Feed-Forward Neural Network (FFNN) or Multilayer Perceptron (MLP)"
    def __init__(self, input_dim=4, output_dim=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),  nn.ReLU(),
            nn.Linear(64, 128),        nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, anchors):
        return self.net(anchors)
    
class HybridFishNet(nn.Module):
    # A hybrid model that processes both the image and the landmark anchors to predict semilandmarks.
    # This is a more complex CNN based approach.
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
        
        # 2. Auto-calculate flattening size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            cnn_out = self.cnn(dummy_input)
            self.cnn_flat_size = cnn_out.shape[1]
            
        # 3. Landmark Branch (Anchors: Points 1 & 13)
        self.anchor_mlp = nn.Sequential(nn.Linear(4, 16), nn.ReLU())

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

class FishGNN(nn.Module):
    """
    Graph neural network over a chain of `n_semi` nodes.

    The graph is a 1-D path (semilandmark i connected to i-1 and i+1). Each
    node is initialised at its linear-interpolation position between the two
    anchors, and message passing along the chain refines those positions.
    A dense, symmetrically-normalised adjacency (Kipf & Welling GCN) is used,
    so no torch_geometric dependency is required. (Swap in a real GCNConv if
    you prefer; the interface is identical.)

    Input : anchors (B, 4)        – same input as the MLP, by design.
    Output: (B, output_dim)
    """
    def __init__(self, output_dim: int = 20, hidden: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_nodes = output_dim // 2

        # Chain adjacency + self-loops, symmetrically normalised: D^-1/2 A D^-1/2
        A = torch.eye(self.n_nodes)
        for i in range(self.n_nodes - 1):
            A[i, i + 1] = 1.0
            A[i + 1, i] = 1.0
        deg_inv_sqrt = A.sum(1).pow(-0.5)
        A_norm = deg_inv_sqrt.unsqueeze(1) * A * deg_inv_sqrt.unsqueeze(0)
        self.register_buffer("A_norm", A_norm)

        # Node feature = [t, interp_x, interp_y, a1x, a1y, a13x, a13y] -> 7 dims
        in_feat = 7
        self.gcn = nn.ModuleList()
        prev = in_feat
        for _ in range(n_layers):
            self.gcn.append(nn.Linear(prev, hidden))
            prev = hidden
        self.head = nn.Linear(hidden, 2)

    def _node_features(self, anchors):
        B = anchors.shape[0]
        a1, a13 = anchors[:, :2], anchors[:, 2:]
        t = torch.linspace(0, 1, self.n_nodes, device=anchors.device)        # (N,)
        interp = (a1.unsqueeze(1) * (1 - t).view(1, -1, 1)
                  + a13.unsqueeze(1) * t.view(1, -1, 1))                      # (B,N,2)
        t_feat  = t.view(1, -1, 1).expand(B, -1, -1)                          # (B,N,1)
        a1_rep  = a1.unsqueeze(1).expand(-1, self.n_nodes, -1)               # (B,N,2)
        a13_rep = a13.unsqueeze(1).expand(-1, self.n_nodes, -1)              # (B,N,2)
        return torch.cat([t_feat, interp, a1_rep, a13_rep], dim=-1)          # (B,N,7)

    def forward(self, anchors):
        H = self._node_features(anchors)
        for lin in self.gcn:
            H = torch.einsum("ij,bjf->bif", self.A_norm, H)   # aggregate neighbours
            H = torch.relu(lin(H))                            # transform
        out = self.head(H)                                    # (B, N, 2)
        return out.reshape(out.shape[0], -1)                  # (B, output_dim)


# ── Vision transformer (+ anchor token) ────────────────────────────────────
class FishViT(nn.Module):
    """
    Small ViT on the letterboxed crop, with the anchors injected as an extra
    learnable token (analogous to a CLS token). The anchor-token output feeds
    the regression head.

    Input : img (B, H, W, C), anchors (B, 4)
    Output: (B, output_dim)

    patch=15 divides both 60 and 270 -> 4 x 18 = 72 patches (+1 anchor token).
    """
    def __init__(self, output_dim: int = 20, input_shape=(3, 60, 270),
                 patch: int = 15, dim: int = 64, depth: int = 4,
                 heads: int = 4, mlp_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        C, H, W = input_shape
        assert H % patch == 0 and W % patch == 0, "patch must divide H and W"

        self.patch_embed = nn.Conv2d(C, dim, kernel_size=patch, stride=patch)
        n_patches = (H // patch) * (W // patch)

        self.anchor_proj = nn.Linear(4, dim)
        self.pos_emb     = nn.Parameter(torch.randn(1, n_patches + 1, dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm    = nn.LayerNorm(dim)
        self.head    = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(),
            nn.Linear(mlp_dim, output_dim),
        )

    def forward(self, img, anchors):
        x = self.patch_embed(img.permute(0, 3, 1, 2))     # (B, dim, h, w)
        x = x.flatten(2).transpose(1, 2)                  # (B, n_patches, dim)
        anchor_tok = self.anchor_proj(anchors).unsqueeze(1)
        x = torch.cat([anchor_tok, x], dim=1) + self.pos_emb
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))              # anchor-token -> head

