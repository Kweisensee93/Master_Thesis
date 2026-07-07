# First part of pipeline detection outlines
# Steps: Import, read landmarks, CLAHE, Blur
from pathlib import Path
import yaml
from PIL import Image, ImageDraw
import cv2 
import numpy as np
#import matplotlib.pyplot as plt

# Set variables
# Use YAML file like a pro 8-)
CONFIG_PATH = Path(
    "C:/Users/korbi/Desktop/A_Master_Thesis/Pipeline/Scripts/Outline_Parameters.yaml"
)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

## retrive variables from YAML
# Image
IMAGE_NAME = config["image"]["name"]
IMAGE_DIR = Path(config["image"]["path"])
IMAGE_FILE = IMAGE_DIR / IMAGE_NAME
# Output
OUTPUT_DIR = Path(config["output"]["path"])
# Landmark / TPS
TPS_NAME = config["tps"]["name"]
TPS_DIR = Path(config["tps"]["path"])
TPS_FILE = TPS_DIR / TPS_NAME
# Parameters
CLAHE = config["parameters_01"]["CLAHE"]
FILTER_BLUR = config["parameters_01"]["Blur"]

# R shiny input should be safe, but YAML can be edited manually
if FILTER_BLUR["type"] not in FILTER_BLUR["valid"]:
    raise ValueError(f"Invalid blur type: {FILTER_BLUR['type']}. Valid options are: {', '.join(FILTER_BLUR['valid'])}")

# Helper function to retrieve landmarks
def read_landmarks_from_tps(tps_path: Path, image_name: str):
    """
    Read landmarks from a TPS file for a specific image.
    
    Parameters:
    -----------
    tps_path : str
        Path to the TPS file
    image_name : str
        Name of the image (e.g., 'CC21L003.JPG')
    
    Returns:
    --------
    list of tuples
        List of (x, y) coordinate tuples, or None if image not found
    """
    landmarks = []
    in_correct_specimen = False
    reading_landmarks = False
    lm_count = 0
    lm_read = 0
    
    with open(tps_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Check if we're starting a new specimen
            if line.startswith('LM='):
                lm_count = int(line.split('=')[1])
                reading_landmarks = True
                lm_read = 0
                landmarks = []
                
            # Read landmark coordinates
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
                    in_correct_specimen = True
                    return landmarks
                else:
                    # Reset for next specimen
                    reading_landmarks = False
                    landmarks = []
                    
    return None  # Image not found in TPS file

# Load image
img = cv2.imread(IMAGE_FILE)
if img is None:
    raise FileNotFoundError(f"Failed to load image: {IMAGE_FILE}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Reduce to Gray for better edge detection

# Add landmarks
img_PIL = Image.open(IMAGE_FILE)
# landmarks are misplaced (maybe origin is bottom-left and in PIL it is top-left)
width, height = img_PIL.size
draw = ImageDraw.Draw(img_PIL)
landmarks = read_landmarks_from_tps(TPS_FILE, IMAGE_NAME)
if landmarks is None:
    print(f"Warning: No landmarks found for {IMAGE_NAME}")
    landmarks = []  # Use empty list as fallback

# Highlight each landmark with a circle
radius = 8
color = (255, 0, 0)  # Red

for x, y in landmarks:
    y_img = height - y  # Adjust y-coordinate for PIL image
    # Draw a circle around each pixel
#    draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
#                 outline=color, fill=color, width=2)
    draw.ellipse(
        [x-radius, y_img-radius, x+radius, y_img+radius],
        outline=color,
        fill=color,
        width=2
    )

# Convert PIL image to numpy array for matplotlib
# Convert to BGR for OpenCV compatibility - who/why/what for uses BGR?!?
#img_with_landmarks = cv2.cvtColor(np.array(img_PIL),cv2.COLOR_RGB2BGR)
# I went with PIL save in the end. But lets keep the conversion as comment - just in case
img_with_landmarks = np.array(img_PIL)

# Crop image down
if landmarks:
    # Extract x and y coordinates
    x_coords = [x for x, y in landmarks]
    y_coords = [y for x, y in landmarks]
    y_coords = [height - y for y in y_coords]  # Adjust y-coordinates for image array
    
    padding = 100
    min_x = max(0, min(x_coords) - padding)
    max_x = min(img_gray.shape[1], max(x_coords) + padding)
    min_y = max(0, min(y_coords) - padding)
    max_y = min(img_gray.shape[0], max(y_coords) + padding)
    
    # Crop all images
    img_gray = img_gray[min_y:max_y, min_x:max_x]
    img_rgb_cropped = img_rgb[min_y:max_y, min_x:max_x]
    img_with_landmarks = img_with_landmarks[min_y:max_y, min_x:max_x]
else:
    print("Warning: No landmarks found, skipping crop")
    img_rgb_cropped = img_rgb

# Step 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=CLAHE["clipLimit"],
                        tileGridSize=(CLAHE["tileGridSize"],CLAHE["tileGridSize"]))
img_clahe = clahe.apply(img_gray)

# Step 2: Blur
blur_ksize = FILTER_BLUR["ksize"]
if FILTER_BLUR["type"] in {"Gaussian", "Median"} and blur_ksize % 2 == 0:
    raise ValueError(
        f"{FILTER_BLUR['type']} blur requires odd ksize, got {blur_ksize}"
    )

if FILTER_BLUR["type"] == "Averaging":
    img_blurred = cv2.blur(img_clahe, (blur_ksize, blur_ksize))
elif FILTER_BLUR["type"] == "Gaussian":
    img_blurred = cv2.GaussianBlur(img_clahe, (blur_ksize, blur_ksize), 0)
elif FILTER_BLUR["type"] == "Median":
    img_blurred = cv2.medianBlur(img_clahe, blur_ksize)
elif FILTER_BLUR["type"] == "Bilateral":
    img_blurred = cv2.bilateralFilter(img_clahe, blur_ksize, 75, 75)
else: # should never happen due to earlier check, but just in case
    raise RuntimeError("Unreachable blur type")

# Save outputs for visualization
# With this setup, we save the intermediate images to the output directory
# Note that we overwrite the files if they already exist - this is intentional to save space
#cv2.imwrite(OUTPUT_DIR / "01_raw.png", img_rgb_cropped)
#cv2.imwrite(OUTPUT_DIR / "02_landmarks.png", img_with_landmarks)
#cv2.imwrite(OUTPUT_DIR / "03_clahe.png", img_clahe)
#cv2.imwrite(OUTPUT_DIR / "04_blur.png", img_blurred)

# Replace cv2.imwrite with PIL saving to avoid BGR/RGB confusion
Image.fromarray(img_rgb_cropped).save(OUTPUT_DIR / "01_raw.png")
Image.fromarray(img_with_landmarks).save(OUTPUT_DIR / "02_landmarks.png")
Image.fromarray(img_clahe, mode="L").save(OUTPUT_DIR / "03_clahe.png")
Image.fromarray(img_blurred, mode="L").save(OUTPUT_DIR / "04_blur.png")
