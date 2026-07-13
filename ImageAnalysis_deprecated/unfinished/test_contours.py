import cv2
import numpy as np

IMAGE_PATH = "C:/Users/korbi/Desktop/A_Master_Thesis/Photo_1/CC21L003.JPG"
OUT_DIR = "C:/Users/korbi/Desktop/A_Master_Thesis/Photo_1/"

# Load image
img = cv2.imread(IMAGE_PATH)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Landmarks (x, y) in image coordinates (top-left origin)
landmarks = [
    (3161, 1825), (3108, 1775), (3017, 1828), (2878, 1829),
    (2586, 1783), (2578, 1710), (1949, 1673), (1550, 1773),
    (1453, 1760), (1450, 1856), (1551, 1838), (2111, 1919),
    (2541, 1949)
]

# --- CROP IMAGE BASED ON LANDMARKS ---
padding = 100

x_coords = [x for x, y in landmarks]
y_coords = [y for x, y in landmarks]

min_x = max(0, min(x_coords) - padding)
max_x = min(img.shape[1], max(x_coords) + padding)
min_y = max(0, min(y_coords) - padding)
max_y = min(img.shape[0], max(y_coords) + padding)

img_cropped = img[min_y:max_y, min_x:max_x]
gray_cropped = img_gray[min_y:max_y, min_x:max_x]

# --- THRESHOLD FOR CONTOURS ---
_, binary = cv2.threshold(
    gray_cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# --- FIND CONTOURS ---
contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# --- DRAW CONTOURS ---
contour_img = img_cropped.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

# --- DRAW LANDMARKS (adjusted to crop) ---
for x, y in landmarks:
    cv2.circle(
        contour_img,
        (x - min_x, y - min_y),
        radius=7,
        color=(0, 0, 255),
        thickness=-1
    )

# Save result
cv2.imwrite(OUT_DIR + "cropped_contours.jpg", contour_img)

print(f"Found {len(contours)} contours")
