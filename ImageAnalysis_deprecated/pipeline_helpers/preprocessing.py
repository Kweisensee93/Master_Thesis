"""
Image preprocessing utilities.

Includes CLAHE, various blur filters, and other preprocessing operations.
"""

from typing import Dict, Any
import cv2
import numpy as np


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Parameters:
        img: Input grayscale image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
    
    Returns:
        CLAHE-enhanced image
    """
    if len(img.shape) != 2:
        raise ValueError("CLAHE requires grayscale image")
    
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size)
    )
    return clahe.apply(img)


def apply_blur(
    img: np.ndarray,
    blur_type: str,
    ksize: int,
    **kwargs
) -> np.ndarray:
    """
    Apply various types of blur filters.
    
    Parameters:
        img: Input image
        blur_type: Type of blur ('Averaging', 'Gaussian', 'Median', 'Bilateral')
        ksize: Kernel size
        **kwargs: Additional parameters for specific blur types
    
    Returns:
        Blurred image
        
    Raises:
        ValueError: If invalid blur type or kernel size
    """

    # This should be imported from YAML file - no redundancies!
    valid_types = ['Averaging', 'Gaussian', 'Median', 'Bilateral']
    
    if blur_type not in valid_types:
        raise ValueError(
            f"Invalid blur type: {blur_type}. "
            f"Valid options are: {', '.join(valid_types)}"
        )
    
    # Validate kernel size for specific blur types
    if blur_type in {'Gaussian', 'Median'} and ksize % 2 == 0:
        raise ValueError(
            f"{blur_type} blur requires odd ksize, got {ksize}"
        )
    
    if blur_type == "Averaging":
        return cv2.blur(img, (ksize, ksize))
    
    elif blur_type == "Gaussian":
        sigma = kwargs.get('sigma', 0)
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)
    
    elif blur_type == "Median":
        return cv2.medianBlur(img, ksize)
    
    elif blur_type == "Bilateral":
        sigma_color = kwargs.get('sigma_color', 75)
        sigma_space = kwargs.get('sigma_space', 75)
        return cv2.bilateralFilter(img, ksize, sigma_color, sigma_space)
    
    else:
        raise RuntimeError("Unreachable blur type")


def preprocess_pipeline(
    img_gray: np.ndarray,
    clahe_params: Dict[str, Any],
    blur_params: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    Apply full preprocessing pipeline: CLAHE + Blur.
    
    Parameters:
        img_gray: Input grayscale image
        clahe_params: Dictionary with CLAHE parameters
        blur_params: Dictionary with blur parameters
    
    Returns:
        Dictionary with intermediate results:
        - 'clahe': Image after CLAHE
        - 'blurred': Image after CLAHE and blur
    """
    # Apply CLAHE
    img_clahe = apply_clahe(
        img_gray,
        clip_limit=clahe_params.get('clipLimit', 2.0),
        tile_grid_size=clahe_params.get('tileGridSize', 8)
    )
    
    # Apply blur
    img_blurred = apply_blur(
        img_clahe,
        blur_type=blur_params['type'],
        ksize=blur_params['ksize']
    )
    
    return {
        'clahe': img_clahe,
        'blurred': img_blurred
    }