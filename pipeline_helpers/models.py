# The Thesis developed more towards a comparison of different approaches / models
# Therefore, the models are stored within this file

# So far
# 1. KNN regressor
# 2. FNN / MLP (SimpleFishMLP)
# 3. Hybrid CNN + MLP (HybridFishNet)
# Options:
# U-Net (stacked hourglass); GCN (Graph Convolutional Network); ViT (Vision Transformer)

import torch
import torch.nn as nn

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