"""
Utilities for reading and manipulating landmark data.

Supports TPS file format commonly used in morphometrics.
"""

from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw
import numpy as np


def read_landmarks_from_tps(
    tps_path: Path, 
    image_name: str
) -> Optional[List[Tuple[int, int]]]:
    """
    Read landmarks from a TPS file for a specific image.
    
    Parameters:
        tps_path: Path to the TPS file
        image_name: Name of the image (e.g., 'CC21L003.JPG')
    
    Returns:
        List of (x, y) coordinate tuples, or None if image not found
    """
    landmarks = []
    reading_landmarks = False
    lm_count = 0
    lm_read = 0


    # With this setup, we read each landmarks until we reach the correct image and then keep the
    # landmarks. This may or may not slow down computation with larger .TPS files ?
    # Possible fix in the future: We read .TPS up to the right "IMAGE=" and then go up a defined
    # number of lines. BUT then the lines within the .TPS need to be fixed. OR I need to check if
    # there is a function that reads the lines "backwards" --> It works for now, maybe update later
    
    with open(tps_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Check if we're starting a new specimen
            if line.startswith('LM='):
                lm_count = int(line.split('=')[1])
                reading_landmarks = True
                lm_read = 0
                landmarks = []
                
            # Read landmark coordinates ; works with Python, because we start at 0
            elif reading_landmarks and lm_read < lm_count:
                try:
                    x, y = map(float, line.split())
                    landmarks.append((int(x), int(y)))
                    lm_read += 1
                except ValueError:
                    pass
                    
            # Check if this is the image we're looking for
            elif line.startswith('IMAGE='):
                current_image = line.split('=')[1]
                if current_image == image_name:
                    return landmarks
                else:
                    # Reset for next specimen
                    reading_landmarks = False
                    landmarks = []
                    
    return None  # Image not found in TPS file


def draw_landmarks_on_image(
    img: Image.Image,
    landmarks: List[Tuple[int, int]],
    radius: int = 8,
    color: Tuple[int, int, int] = (255, 0, 0)
) -> Image.Image:
    """
    Draw landmarks on an image.
    
    Parameters:
        img: PIL Image object
        landmarks: List of (x, y) coordinate tuples
        radius: Radius of the landmark circles
        color: RGB color tuple for the landmarks
    
    Returns:
        PIL Image with landmarks drawn
    """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for x, y in landmarks:
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            outline=color,
            fill=color,
            width=2
        )
    
    return img_copy


# This may be removed, once I get the cropping from Ben :)
def get_landmark_bounds(
    landmarks: List[Tuple[int, int]],
    padding: int = 100,
    bottom_padding: int = 50
) -> Tuple[int, int, int, int]:
    """
    Calculate bounding box around landmarks with padding.
    
    Parameters:
        landmarks: List of (x, y) coordinate tuples
        padding: Padding to add around landmarks (default: 100)
        bottom_padding: Extra padding for bottom (default: 50)
    
    Returns:
        Tuple of (min_x, max_x, min_y, max_y)
    """
    if not landmarks:
        raise ValueError("No landmarks provided")
    
    x_coords = [x for x, y in landmarks]
    y_coords = [y for x, y in landmarks]
    
    min_x = min(x_coords) - padding
    max_x = max(x_coords) + padding
    min_y = min(y_coords) - padding
    max_y = max(y_coords) + padding + bottom_padding
    
    return min_x, max_x, min_y, max_y