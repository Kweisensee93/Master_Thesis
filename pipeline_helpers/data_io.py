# pipeline_helpers/data_io.py
from pathlib import Path
import numpy as np
import cv2

#1  parse_tps:       Parse TPS files
#2  load_image:      Get image

def parse_tps(tps_path: Path) -> dict:
    """
    Parse an entire TPS file and return a dict keyed by IMAGE name.
    It needs to follow the structure of the TPS file as follows (if order changes, ADAPTION NEEDED):
    1) LM=          number_of_landmarks
    2) (x, y) coordinates of landmarks
    3) CURVES=      number of curves (optional, if there are curves)
    4) POINTS=      number_of_points_in_curves
    5) (x, y) coordinates of semi-landmarks from CURVES blocks
    6) IMAGE=       image_name (used as key in the returned dict)
    7) ID=          specimen ID (optional)
    8) SCALE=       scale factor (optional)

    Each value is a dict with:
        'landmarks'      : list of (x, y) – fixed landmarks  (LM= block)
        'semi_landmarks' : nested list with list of (x, y) – semi-landmarks from CURVES blocks per POINTS
        'id'             : specimen ID string
        'scale'          : scale factor (float)
    """
    specimens = {}
    current: dict = {}
    lm_count, lm_read = 0, 0
    reading_lm = False
    curve_points_count, curve_points_read = 0, 0
    reading_curve = False

    with open(tps_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            ### We follow the TPS file structure
            #1) LM
            if line.upper().startswith("LM="):
                # initialize new entry
                current = {"landmarks": [], "semi_landmarks": [], "id": None, "scale": None}
                lm_count = int(line.split("=", 1)[1]) #extract landmark count
                lm_read = 0 #initialize read counter for landmarks
                reading_lm = True #start reading landmarks
                reading_curve = False #reset curve reading state (better safe than sorry)
                continue
            #2) (x, y) coordinates of landmarks
            if reading_lm and lm_read < lm_count: #ensure we only read landmarks, and count through them
                try:
                    x, y = map(float, line.split())
                    # add rounding as a safeguard for placing landmarks
                    current["landmarks"].append((int(round(x)), int(round(y)))) # feed landmarks
                    lm_read += 1
                except ValueError:
                    pass
                continue
            #3) CURVES
            if line.upper().startswith("CURVES="):
                reading_lm = reading_curve = False      #reset reading states
                continue

            if line.upper().startswith("POINTS="):
                curve_points_count= int(line.split("=", 1)[1])
                curve_points_read = 0
                reading_curve = True                     #start reading curves
                current["semi_landmarks"].append([])     #start a new sublist for this curve
                continue

            if reading_curve and curve_points_read < curve_points_count:
                try:
                    x, y = map(float, line.split())
                    # add rounding as a safeguard
                    current["semi_landmarks"][-1].append((int(round(x)), int(round(y)))) #append to last nested list
                    curve_points_read += 1
                except ValueError:
                    pass
                if curve_points_read >= curve_points_count:
                    reading_curve = False
                continue

            if line.upper().startswith("IMAGE="):
                reading_lm = reading_curve = False
                img_name = line.split("=", 1)[1].strip()
                current["image"] = img_name
                specimens[img_name] = current
                current = {}
                continue

            if line.upper().startswith("ID="):
                current["id"] = line.split("=", 1)[1].strip()
                continue

            if line.upper().startswith("SCALE="):
                try:
                    current["scale"] = float(line.split("=", 1)[1])
                except ValueError:
                    pass
                continue

    return specimens

def load_image(img_path: Path | str) -> tuple[np.ndarray, int, int]:
    """
    Load an image from disk.

    Returns
    -------
    np.ndarray  shape (H, W, 3), dtype uint8, BGR channel order
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    return img_bgr, img_bgr.shape[0], img_bgr.shape[1]
