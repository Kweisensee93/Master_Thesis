# Comparison of edge detection filters with CLAHE preprocessing for stickleback

# Import
from PIL import Image, ImageDraw
import cv2 
import numpy as np
import matplotlib.pyplot as plt

# Set variables
IMAGE_NAME = "CC21L032.JPG"
IMAGE_PATH = "C:/Users/korbi/Desktop/A_Master_Thesis/rawdata/" + IMAGE_NAME
OUT_DIR = "C:/Users/korbi/Desktop/A_Master_Thesis/output/"
TPS_PATH = "C:/Users/Korbi/Desktop/A_Master_Thesis/landmark01.TPS"

## For fine-tuning:
LOWER = 40                              # 0
UPPER = 255                             # 255

CANNY_THRESHOLD1 = 100                  # 50
CANNY_THRESHOLD2 = 200                  # 150

SOBEL_KERNEL = 3                        # 5
SOBEL_THRESHOLD_LOWER = LOWER + 0
SOBEL_THRESHOLD_UPPER = UPPER - 0

LAPLACIAN_KERNEL = 10                   # 3     Smaller = Sharper
LAPLACIAN_THRESHOLD_LOWER = LOWER + 0
LAPLACIAN_THRESHOLD_UPPER = UPPER - 0

PREWITT_THRESHOLD_LOWER = LOWER + 0
PREWITT_THRESHOLD_UPPER = UPPER - 0

CLOSING_KERNEL = 10                     # 5

# Combination method
WEIGHTED_CANNY = 0.4            # Weight for Canny in weighted average (0.0-1.0)
WEIGHTED_SOBEL = 0.3            # Weight for Sobel in weighted average (0.0-1.0)
WEIGHTED_PREWITT = 0.3          # Weight for Prewitt in weighted average (0.0-1.0)
WEIGHTED_THRESHOLD = 127        # Threshold for weighted combination (0-255)

# Helper function to retrieve landmarks
def read_landmarks_from_tps(tps_path, image_name):
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
img = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Reduce to Gray for better edge detection

# Add landmarks
img_PIL = Image.open(IMAGE_PATH)
draw = ImageDraw.Draw(img_PIL)
landmarks = read_landmarks_from_tps(TPS_PATH, IMAGE_NAME)
if landmarks is None:
    print(f"Warning: No landmarks found for {IMAGE_NAME}")
    landmarks = []  # Use empty list as fallback

# Highlight each landmark with a circle
radius = 10
color = (255, 0, 0)  # Red

for x, y in landmarks:
    # Draw a circle around each pixel
    draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                 outline=color, fill=color, width=2)

# Convert PIL image to numpy array for matplotlib
img_with_landmarks = np.array(img_PIL)

# Crop image down
if landmarks:
    # Extract x and y coordinates
    x_coords = [x for x, y in landmarks]
    y_coords = [y for x, y in landmarks]
    
    padding = 100
    min_x = max(0, min(x_coords) - padding)
    max_x = min(img_gray.shape[1], max(x_coords) + padding)
    min_y = max(0, min(y_coords) - padding)
    max_y = min(img_gray.shape[0], max(y_coords) + padding + 50)  # usually cut off at bottom
    
    # Crop all images
    img_gray = img_gray[min_y:max_y, min_x:max_x]
    img_rgb_cropped = img_rgb[min_y:max_y, min_x:max_x]
    img_with_landmarks = img_with_landmarks[min_y:max_y, min_x:max_x]
else:
    print("Warning: No landmarks found, skipping crop")
    img_rgb_cropped = img_rgb

# =====================
# PREPROCESSING PIPELINE
# =====================

# Step 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img_gray)

# Step 2: Gaussian Blur
img_gaussian = cv2.GaussianBlur(img_clahe, (5, 5), 0)

# =====================
# EDGE DETECTION METHODS
# =====================

# Canny Edge Detection
edges_canny = cv2.Canny(img_gaussian,
                        threshold1=CANNY_THRESHOLD1,
                        threshold2=CANNY_THRESHOLD2,
                        apertureSize=3)

# Sobel Filter
sobelx = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=SOBEL_KERNEL)
sobely = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1, ksize=SOBEL_KERNEL)
sobel_magnitude = cv2.magnitude(sobelx, sobely)
edges_sobel = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

_, edges_sobel = cv2.threshold(edges_sobel, SOBEL_THRESHOLD_LOWER, SOBEL_THRESHOLD_UPPER, cv2.THRESH_BINARY)

# Laplacian Filter
laplacian = cv2.Laplacian(img_gaussian, cv2.CV_64F)
edges_laplacian = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

_, edges_laplacian = cv2.threshold(edges_laplacian,
                                   LAPLACIAN_THRESHOLD_LOWER,
                                   LAPLACIAN_THRESHOLD_UPPER,
                                   cv2.THRESH_BINARY)

# Prewitt Filter
kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
px = cv2.filter2D(img_gaussian, -1, kernelx)
py = cv2.filter2D(img_gaussian, -1, kernely)
prewitt_magnitude = cv2.magnitude(px.astype(np.float32), py.astype(np.float32))
edges_prewitt = cv2.normalize(prewitt_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

_, edges_prewitt = cv2.threshold(edges_prewitt, 40, 255, cv2.THRESH_BINARY)

# =====================
# COMBINATION METHODS
# =====================

# 1. OR Combination (Union) - combines all edges
combined_or = cv2.bitwise_or(edges_canny, edges_sobel)
combined_or = cv2.bitwise_or(combined_or, edges_prewitt)

# 2. AND Combination (Intersection) - only edges detected by all methods
combined_and = cv2.bitwise_and(edges_canny, edges_sobel)
combined_and = cv2.bitwise_and(combined_and, edges_prewitt)

# 3. Weighted Average - gives different importance to each method
combined_weighted = (WEIGHTED_CANNY * edges_canny.astype(float) + 
                     WEIGHTED_SOBEL * edges_sobel.astype(float) + 
                     WEIGHTED_PREWITT * edges_prewitt.astype(float))
combined_weighted = np.clip(combined_weighted, 0, 255).astype(np.uint8)
_, combined_weighted = cv2.threshold(combined_weighted, WEIGHTED_THRESHOLD, 255, cv2.THRESH_BINARY)

# 4. Maximum - brightest pixel wins
combined_max = np.maximum(edges_canny, edges_sobel)
combined_max = np.maximum(combined_max, edges_prewitt)

# =====================
# MORPHOLOGICAL CLOSING
# =====================

# Create elliptical kernel for closing (better for organic shapes)
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSING_KERNEL, CLOSING_KERNEL))

# Apply closing to each edge detection result
edges_canny_closed = cv2.morphologyEx(edges_canny, cv2.MORPH_CLOSE, kernel_close)
edges_sobel_closed = cv2.morphologyEx(edges_sobel, cv2.MORPH_CLOSE, kernel_close)
edges_laplacian_closed = cv2.morphologyEx(edges_laplacian, cv2.MORPH_CLOSE, kernel_close)
edges_prewitt_closed = cv2.morphologyEx(edges_prewitt, cv2.MORPH_CLOSE, kernel_close)

# Apply closing to combinations
combined_or_closed = cv2.morphologyEx(combined_or, cv2.MORPH_CLOSE, kernel_close)
combined_and_closed = cv2.morphologyEx(combined_and, cv2.MORPH_CLOSE, kernel_close)
combined_weighted_closed = cv2.morphologyEx(combined_weighted, cv2.MORPH_CLOSE, kernel_close)
combined_max_closed = cv2.morphologyEx(combined_max, cv2.MORPH_CLOSE, kernel_close)


# =====================
# VISUALIZATION
# =====================

fig = plt.figure(figsize=(20, 15))

titles = [
    "Original Image",
    "Image with Landmarks",
    "CLAHE Enhanced",
    "Gaussian Blurred",
    "Canny (Bare)",
    "Sobel (Bare)",
    "Laplacian (Bare)",
    "Prewitt (Bare)",
    "Canny (Closed)",
    "Sobel (Closed)",
    "Laplacian (Closed)",
    "Prewitt (Closed)",
    "Combined OR (Union)",
    "Combined AND (Intersection)",
    "Combined Weighted",
    "Combined MAX",
    "Combined OR (Closed)",
    "Combined AND (Closed)",
    "Combined Weighted (Closed)",
    "Combined MAX (Closed)"
]

images = [
    img_rgb_cropped,
    img_with_landmarks,
    img_clahe,
    img_gaussian,
    edges_canny,
    edges_sobel,
    edges_laplacian,
    edges_prewitt,
    edges_canny_closed,
    edges_sobel_closed,
    edges_laplacian_closed,
    edges_prewitt_closed,
    combined_or,
    combined_and,
    combined_weighted,
    combined_max,
    combined_or_closed,
    combined_and_closed,
    combined_weighted_closed,
    combined_max_closed
]

for i, (title, im) in enumerate(zip(titles, images)):
    plt.subplot(5, 4, i+1)
    plt.title(title, fontsize=9, fontweight='bold')
    if i < 2:  # Original and landmarks (RGB images)
        plt.imshow(im)
    else:
        plt.imshow(im, cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.savefig(OUT_DIR + "pipeline_comparison_" + IMAGE_NAME, dpi=300, bbox_inches="tight")
plt.show()

print("\n=== Processing Complete ===")
print(f"Output saved to: {OUT_DIR}comparison_{IMAGE_NAME}")