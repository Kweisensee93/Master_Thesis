"""
Image input/output utilities.

Handles loading, saving, and cropping of images.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
from PIL import Image

from .landmark_utils import get_landmark_bounds, draw_landmarks_on_image


def load_image(image_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load an image and return multiple color space versions.
    
    Parameters:
        image_path: Path to the image file
    
    Returns:
        Tuple of (original BGR, RGB, grayscale) numpy arrays
        
    Raises:
        FileNotFoundError: If image cannot be loaded
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    
    # OpenCV uses BGR instead of RGB O_o
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img, img_rgb, img_gray


def crop_to_landmarks(
    img: np.ndarray,
    landmarks: List[Tuple[int, int]],
    padding: int = 100,
    bottom_padding: int = 50
) -> np.ndarray:
    """
    Crop image to the bounding box of landmarks with padding.
    
    Parameters:
        img: Input image (numpy array)
        landmarks: List of (x, y) coordinate tuples
        padding: Padding to add around landmarks
        bottom_padding: Extra padding for bottom
    
    Returns:
        Cropped image
    """
    if not landmarks:
        print("Warning: No landmarks found, skipping crop")
        return img
    
    min_x, max_x, min_y, max_y = get_landmark_bounds(
        landmarks, 
        padding, 
        bottom_padding
    )
    
    # Ensure bounds are within image dimensions
    height, width = img.shape[:2]
    min_x = max(0, min_x)
    max_x = min(width, max_x)
    min_y = max(0, min_y)
    max_y = min(height, max_y)
    
    return img[min_y:max_y, min_x:max_x]


def prepare_images_with_landmarks(
    image_path: Path,
    landmarks: Optional[List[Tuple[int, int]]] = None,
    crop: bool = True,
    landmark_radius: int = 8,
    landmark_color: Tuple[int, int, int] = (255, 0, 0)
) -> Dict[str, np.ndarray]:
    """
    Load image and prepare versions with landmarks and cropping.
    
    Parameters:
        image_path: Path to the image file
        landmarks: List of (x, y) coordinate tuples (optional)
        crop: Whether to crop to landmark bounds
        landmark_radius: Radius for landmark visualization
        landmark_color: RGB color for landmarks
    
    Returns:
        Dictionary with keys: 'rgb', 'gray', 'with_landmarks'
    """
    # Load image
    _, img_rgb, img_gray = load_image(image_path)
    
    # Handle landmarks
    if landmarks is None:
        landmarks = []
        print(f"Warning: No landmarks provided")
    
    # Draw landmarks on RGB version
    img_pil = Image.fromarray(img_rgb)
    img_with_landmarks_pil = draw_landmarks_on_image(
        img_pil, 
        landmarks, 
        landmark_radius, 
        landmark_color
    )
    img_with_landmarks = np.array(img_with_landmarks_pil)
    
    # Crop if requested
    if crop and landmarks:
        img_rgb = crop_to_landmarks(img_rgb, landmarks)
        img_gray = crop_to_landmarks(img_gray, landmarks)
        img_with_landmarks = crop_to_landmarks(img_with_landmarks, landmarks)
    
    return {
        'rgb': img_rgb,
        'gray': img_gray,
        'with_landmarks': img_with_landmarks
    }


def save_images(
    output_dir: Path,
    images: Dict[str, np.ndarray],
    overwrite: bool = True
) -> None:
    """
    Save multiple images to disk.
    
    Parameters:
        output_dir: Directory to save images
        images: Dictionary mapping filenames to image arrays
        overwrite: Whether to overwrite existing files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, img in images.items():
        output_path = output_dir / filename
        
        if not overwrite and output_path.exists():
            print(f"Skipping {filename} (already exists)")
            continue
        
        # Determine if image is grayscale or color
        if len(img.shape) == 2:
            # Grayscale
            Image.fromarray(img, mode="L").save(output_path)
        else:
            # RGB
            Image.fromarray(img).save(output_path)
        
        print(f"Saved: {output_path}")