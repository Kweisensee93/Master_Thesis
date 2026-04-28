# pipeline_helpers/preprocessing.py
from .config import SEMILANDMARK_CURVE
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

    min_x = max(0,     min(xs) - 2*pad_x)
    max_x = min(img_w, max(xs) + pad_x)
    min_y = max(0,     min(ys) - 2*pad_y)
    max_y = min(img_h, max(ys) + 3*pad_y)

    return min_x, max_x, min_y, max_y

def crop_to_landmarks(
    img_bgr: np.ndarray,
    data: dict,
    img_h: int,
    img_w: int,
    semilandmark_curve: int = SEMILANDMARK_CURVE,
    padding_frac: float = 0.10,
) -> tuple[np.ndarray, tuple]:
    """
    Crop a dynamically-sized window around all landmarks + percentage padding.

    Returns
    -------
    img_cropped : np.ndarray  (H, W, 3) uint8 – RGB
    crop_box    : (min_x, max_x, min_y, max_y) in original image coords
    """
    landmarks      = flip_y(data.get("landmarks",      []), img_h)
    #semi_landmarks = flip_y(data.get("semi_landmarks", [[]])[semilandmark_curve], img_h)
    # Fails if no curves exist, so we handle it more robust:
    # First we get all curves, then we check if the requested curve index exists before trying to flip it.
    # If it doesn't exist, we just use an empty list for semi_landmarks.
    # If this doesn't impress you, I have way more disappointments at the ready
    all_curves = data.get("semi_landmarks", [])
    if all_curves and len(all_curves) > semilandmark_curve:
        semi_landmarks = flip_y(all_curves[semilandmark_curve], img_h)
    else:
        semi_landmarks = [] # Fallback if no curves exist

    all_coords     = landmarks + semi_landmarks

    if not all_coords:
        raise ValueError("No landmarks available – cannot crop.")

    min_x, max_x, min_y, max_y = find_dynamic_crop_box(all_coords, img_h, img_w, padding_frac)

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
    include_semi: bool = True
):
    images_list, bare_list, rel_list, crop_boxes, lmark_list, names = [], [], [], [], [], []

    for name in image_names:
        img_path = fish_dir / name
        if not img_path.exists(): continue
        data = tps_data.get(name, {})
        try:
            img_bgr, img_h, img_w = load_image(img_path)
            img_cropped, crop_box = crop_to_landmarks(img_bgr, data, img_h, img_w)

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

# For DL approach, we need a resized dataset:
def build_dataset_resized(
    image_names: list | np.ndarray,
    fish_dir: Path,
    tps_data: dict,
    target_size: tuple = (128, 128),
    include_semi: bool = True
):
    """
    Builds a dataset where all images are resized to target_size to allow stacking.
    Returns images, bare_landmarks, relative_landmarks, and metadata.
    """
    images_list, bare_list, rel_list, crop_boxes, lmark_list, names = [], [], [], [], [], []

    for name in image_names:
        img_path = fish_dir / name
        if not img_path.exists(): continue
        data = tps_data.get(name, {})
        
        try:
            # 1. Load and Crop
            img_bgr, img_h, img_w = load_image(img_path)
            # Fix: Ensure positional arguments match: (img_bgr, data, img_h, img_w)
            img_cropped, crop_box = crop_to_landmarks(img_bgr, data, img_h, img_w)

            # 2. Resize Image to standard dimensions
            img_resized = cv2.resize(img_cropped, target_size)
            # Normalize pixel values to [0, 1] for Neural Network stability
            images_list.append(img_resized.astype(np.float32) / 255.0)
            
            crop_boxes.append(crop_box)
            names.append(name)
            
            # 3. Handle Landmarks
            lmarks_px = flip_y(data.get("landmarks", []), img_h)
            lmark_list.append(lmarks_px)

            if include_semi:
                semi_px = flip_y(data.get("semi_landmarks", [[]])[SEMILANDMARK_CURVE], img_h)
                
                # Version A: Normalised [0,1] relative to the crop box
                bare_norm = normalise_landmarks(semi_px, crop_box)
                bare_list.append(bare_norm)
                
                # Version B: Normalised relative to LM1->LM13 basis (Ideal for DL)
                from .normalisation import normalise_landmarks_relative
                rel_norm = normalise_landmarks_relative(semi_px, lmarks_px)
                rel_list.append(rel_norm)
            else:
                bare_list.append(np.zeros(1))
                rel_list.append(np.zeros(1))
                
        except Exception as e:
            print(f"Error processing {name}: {e}")

    # np.stack now works because all images in images_list are (128, 128, 3)
    return (np.stack(images_list), np.stack(bare_list), np.stack(rel_list), 
            crop_boxes, lmark_list, names)

def build_dataset_hybrid(image_names, fish_dir, tps_data, target_size=(270, 60)):
    images_list, anchor_list, rel_list, names = [], [], [], []

    for name in image_names:
        img_path = fish_dir / name
        data = tps_data.get(name, {})
        img_bgr, img_h, img_w = load_image(img_path)
        img_cropped, crop_box = crop_to_landmarks(img_bgr, data, img_h, img_w)

        # Calculate scale to fit inside target_size while maintaining aspect ratio
        h, w = img_cropped.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h) # width/height scaling
        nw, nh = int(w * scale), int(h * scale)
        
        img_resized = cv2.resize(img_cropped, (nw, nh))
        
        # Create blank canvas and paste image in center (Letterboxing)
        canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.float32)
        y_offset = (target_size[1] - nh) // 2
        x_offset = (target_size[0] - nw) // 2
        canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = img_resized.astype(np.float32) / 255.0
        
        images_list.append(canvas)
        #images_list.append(canvas.astype(np.float32) / 255.0)

        # 2. Get ONLY Landmarks 1 and 13 (Indices 0 and 12)
        lmarks_px = flip_y(data.get("landmarks", []), img_h)
        # We normalize these relative to the crop box so the model knows where they are in the image
        anchors = [( (lmarks_px[0][0]-crop_box[0])/(crop_box[1]-crop_box[0]), 
                     (lmarks_px[0][1]-crop_box[2])/(crop_box[3]-crop_box[2]) ),
                   ( (lmarks_px[12][0]-crop_box[0])/(crop_box[1]-crop_box[0]), 
                     (lmarks_px[12][1]-crop_box[2])/(crop_box[3]-crop_box[2]) )]
        anchor_list.append(np.array(anchors).flatten())

        # 3. Target: Relative Semilandmarks
        semi_px = flip_y(data.get("semi_landmarks", [[]])[0], img_h)
        rel_norm = normalise_landmarks_relative(semi_px, lmarks_px)
        rel_list.append(rel_norm)
        names.append(name)

    return np.stack(images_list), np.stack(anchor_list), np.stack(rel_list), names