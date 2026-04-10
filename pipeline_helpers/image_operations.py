# pipeline_helpers/Image_Operations.py
import numpy as np
import cv2
from PIL import Image
import io
from rembg import remove

def extract_fish_contour(img_rgba: np.ndarray) -> np.ndarray:
    """
    Processes RGBA image to find the largest master contour.
    Uses the 'Glue' method to merge fragmented segments.
    """
    # Use alpha channel as mask --> create binary mask based on alpha value
    alpha = img_rgba[:, :, 3]
    _, binary_mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)

    # Find initial contours = retrieve the outermost contours from the binary mask
    contours_crop, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours_crop:
        return None, None

    # The "Glue" Method: Fill all detected shapes into one solid mask
    temp_mask = np.zeros_like(binary_mask)
    for cnt in contours_crop:
        cv2.drawContours(temp_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Re-extract the single master contour from the "glued" mask = find contour again after glueing
    merged_cnts, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not merged_cnts:
        return None, None
    
    # Remove any small contours that may remain (noise, artfacts)
    fish_contour = max(merged_cnts, key=cv2.contourArea)
    # Fish countour = downstream; contours_crop = debugging
    return fish_contour, contours_crop

def remove_background(img_rgb: np.ndarray) -> np.ndarray:
    """
    Takes an RGB image (numpy array), runs rembg, 
    and returns an RGBA numpy array.
    """
    img_pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    
    # Process with rembg
    output_data = remove(buf.getvalue())
    
    # Return as RGBA array
    return np.array(Image.open(io.BytesIO(output_data)).convert("RGBA"))