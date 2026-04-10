# pipeline_helpers/preprocessing.py
from .config import PADDING, CROP_LENGTH, CROP_HEIGHT, SEMILANDMARK_CURVE
from .geometry import flip_y, coords_to_array
from .data_io import load_image
from .normalisation import normalise_landmarks_relative

import numpy as np
import cv2
from pathlib import Path

def find_dynamic_crop_box(
    all_coords: list[tuple],
    img_h: int,
    img_w: int,
    padding_frac: float = 0.10,
) -> tuple[int, int, int, int]:
    """
    Derive a crop box dynamically from landmark extents + percentage padding.

    Parameters
    ----------
    all_coords   : list of (x, y) tuples – all landmarks in full image space
    img_h        : original image height  (for boundary clamping)
    img_w        : original image width   (for boundary clamping)
    padding_frac : fractional padding added to each side (default 0.10 = 10%)

    Returns
    -------
    (min_x, max_x, min_y, max_y) clamped to image boundaries
    """
    xs = [x for x, _ in all_coords]
    ys = [y for _, y in all_coords]

    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)

    pad_x = int(span_x * padding_frac)
    pad_y = int(span_y * padding_frac)

    min_x = max(0,     min(xs) - pad_x)
    max_x = min(img_w, max(xs) + pad_x)
    min_y = max(0,     min(ys) - pad_y)
    max_y = min(img_h, max(ys) + pad_y)

    return min_x, max_x, min_y, max_y

def crop_to_landmarks(
    img_bgr: np.ndarray,
    data: dict,
    padding: int       = PADDING,
    length: int        = CROP_LENGTH,
    height: int        = CROP_HEIGHT,
    semilandmark_curve: int = SEMILANDMARK_CURVE,
) -> tuple[np.ndarray, tuple]:
    """
    Crop a fixed-size window anchored to the leftmost landmark, then resize.

    Returns
    -------
    img_cropped_resized : np.ndarray  (H, W, 3) uint8 – RGB, model-ready
    crop_box            : (min_x, max_x, min_y, max_y) in original image coords
    """
    img_h, img_w = img_bgr.shape[:2]

    landmarks      = flip_y(data.get("landmarks",      []), img_h)
    semi_landmarks = flip_y(data.get("semi_landmarks", [[]])[semilandmark_curve], img_h)
    all_coords     = landmarks + semi_landmarks

    if not all_coords:
        raise ValueError("No landmarks available – cannot crop.")

    xs = [x for x, _ in all_coords]
    ys = [y for _, y in all_coords]

    min_x = max(0, min(xs) - 2*padding)
    min_y = max(0, min(ys) - padding)
    max_x = min_x + length
    max_y = min_y + height

    if max_x > img_w:
        raise ValueError(f"Crop window exceeds image width: {max_x} > {img_w}")
    if max_y > img_h:
        raise ValueError(f"Crop window exceeds image height: {max_y} > {img_h}")

    img_cropped = img_bgr[min_y:max_y, min_x:max_x]
    img_rgb     = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)

    return img_rgb, (min_x, max_x, min_y, max_y)

# ── Landmark coordinate normalisation ────────────────────────────────────────
def normalise_landmarks(
    coords: list[tuple],
    crop_box: tuple,
) -> np.ndarray:
    """
    Express landmark coordinates relative to the crop window and then scale
    them to [0, 1] using the resized image dimensions.

    This makes the coordinates directly comparable across specimens regardless
    of where in the original image the crop landed.

    Returns
    -------
    np.ndarray  flat float32 array  [x0, y0, x1, y1, …]  values in [0, 1]
    """
    min_x, max_x, min_y, max_y = crop_box
    crop_w = max_x - min_x
    crop_h = max_y - min_y

    normalised = []
    for x, y in coords:
        nx = (x - min_x) / crop_w
        ny = (y - min_y) / crop_h
        normalised.append((nx, ny))

    return coords_to_array(normalised)


def denormalise_landmarks(
    arr: np.ndarray,
    crop_box: tuple,
) -> list[tuple]:
    """
    Reverse of normalise_landmarks: map [0, 1] values back to original pixel
    coordinates in the full (unresized) image.
    """
    min_x, max_x, min_y, max_y = crop_box
    crop_w = max_x - min_x
    crop_h = max_y - min_y

    coords = arr.reshape(-1, 2)
    return [
        (int(round(nx * crop_w + min_x)), int(round(ny * crop_h + min_y)))
        for nx, ny in coords
    ]

# ── Dataset builder ───────────────────────────────────────────────────────────
def build_dataset(
    image_names: list | np.ndarray,
    fish_dir: Path,
    tps_data: dict,
    include_semi: bool = True,
    length: int        = CROP_LENGTH,
    height: int        = CROP_HEIGHT
):
    images_list, bare_list, rel_list, crop_boxes, lmark_list, names = [], [], [], [], [], []

    for name in image_names:
        img_path = fish_dir / name
        if not img_path.exists(): continue
        data = tps_data.get(name, {})
        try:
            img_bgr = load_image(img_path)
            img_h = img_bgr.shape[0]
            img_cropped, crop_box = crop_to_landmarks(img_bgr, data, length=length, height=height)

            images_list.append(img_cropped.astype(np.float32))
            crop_boxes.append(crop_box)
            names.append(name)
            
            lmarks_px = flip_y(data.get("landmarks", []), img_h)
            lmark_list.append(lmarks_px)

            if include_semi:
                semi_px = flip_y(data.get("semi_landmarks", [[]])[SEMILANDMARK_CURVE], img_h)
                # Version A: Normalised [0,1] relative to the crop box
                bare_norm = normalise_landmarks(semi_px, crop_box)
                bare_list.append(bare_norm)
                # Version B: Normalised relative to LM1->LM13 basis
                rel_norm = normalise_landmarks_relative(semi_px, lmarks_px)
                rel_list.append(rel_norm)
            else:
                bare_list.append(np.zeros(1))
                rel_list.append(np.zeros(1))
        except Exception as e:
            print(f"Error processing {name}: {e}")

    return (np.stack(images_list), np.stack(bare_list), np.stack(rel_list), 
            crop_boxes, lmark_list, names)


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
