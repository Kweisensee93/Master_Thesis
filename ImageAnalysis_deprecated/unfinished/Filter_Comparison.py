# Comparison of several filters for line detection for stickleback

# Import
from PIL import Image, ImageDraw
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import structure_tensor

# Load image
IMAGE_PATH = "C:/Users/korbi/Desktop/A_Master_Thesis/Photo_1/CC21L003.JPG"
OUT_DIR = "C:/Users/korbi/Desktop/A_Master_Thesis/Photo_1/"

img = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Add landmarks:
img_PIL = Image.open(IMAGE_PATH)
draw = ImageDraw.Draw(img_PIL)
landmarks = [
    (3161, 1825),
    (3108, 1775),
    (3017, 1828),
    (2878, 1829),
    (2586, 1783),
    (2578, 1710),
    (1949, 1673),
    (1550, 1773),
    (1453, 1760),
    (1450, 1856),
    (1551, 1838),
    (2111, 1919),
    (2541, 1949)
]
# Highlight each landmark with a circle
radius = 10
color = (255, 0, 0)  # Red

for x, y in landmarks:
    # Draw a circle around each pixel
    draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                 outline=color, fill=color, width=2)
# Convert PIL image to numpy array for matplotlib
img_with_landmarks = np.array(img_PIL)

#crop image down
# cropped = img[min_y:max_y, min_x:max_x]
img_gray = img_gray[1600:2200, 1300:3200]


###Canny Edge Detection###
edges = cv2.Canny(img_gray, threshold1=50, threshold2=150)

###Sobel Filter###
sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
sobel_magnitude = cv2.magnitude(sobelx, sobely)
# Normalize for visualization
sobel_vis = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


###Laplacian Filter###
blur = cv2.GaussianBlur(img_gray, (7,7), 0)
laplacian = cv2.Laplacian(blur, cv2.CV_64F)
# Normalize for better visualization
laplacian_vis = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


###Prewitt Filter###
kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
px = cv2.filter2D(img_gray, -1, kernelx)
py = cv2.filter2D(img_gray, -1, kernely)
prewitt_mag = cv2.magnitude(
    px.astype(np.float32),
    py.astype(np.float32))
prewitt_mag = cv2.normalize(
    prewitt_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)



###Hough Transform for Line Detection###
# #lines_std = cv2.HoughLines(edges, 1, np.pi/180, 150)
# lines_prob = cv2.HoughLinesP(
#     edges,
#     rho=1,
#     theta=np.pi/180,
#     threshold=100,
#     minLineLength=50,
#     maxLineGap=10
# )
# #hough_img = img.copy()
# hough_img = img_rgb.copy()
# if lines_prob is not None:
#     for x1, y1, x2, y2 in lines_prob[:, 0]:
#         cv2.line(hough_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

#We leave out hough since it gives discrete lines (maybe we backtack later)

###Morphological Operations###
##kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,1))
#lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7,1))
kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1,7))

lines_h = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)
lines_v = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)

lines_morph = cv2.bitwise_or(lines_h, lines_v)

# Alternative: Use closing to fill gaps in lines
kernel_h_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
kernel_v_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
lines_h_enhanced = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h_close)
lines_v_enhanced = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_v_close)
lines_morph_enhanced = cv2.bitwise_or(lines_h_enhanced, lines_v_enhanced)

###Structure Tensor###
Axx, Axy, Ayy = structure_tensor(img_gray, sigma=2)
# Compute eigenvalues to find coherence and orientation
coherence = np.sqrt((Axx - Ayy)**2 + 4*Axy**2) / (Axx + Ayy + 1e-10)

# Visualization
fig = plt.figure(figsize=(16, 10))

titles = [
    "Original Image",
    "Image with Landmarks",
    "Canny Edges",
    "Sobel Magnitude",
    "Laplacian",
    "Prewitt Magnitude",
    #"Hough Lines",
    "Morphological Lines (Open)",
    "Morphological Lines (Close)",
    "Structure Tensor Coherence"
]

images = [
    img_rgb,
    img_with_landmarks,
    edges,
    sobel_vis,
    laplacian_vis,
    prewitt_mag,
    #hough_img,
    lines_morph,
    lines_morph_enhanced,
    coherence
]

cmaps = ['viridis', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray']
# other options for cmaps could be 'gray', 'viridis', 'hot'

for i, (title, im, cmap) in enumerate(zip(titles, images, cmaps)):
    plt.subplot(3, 3, i+1)
    plt.title(title, fontsize=10, fontweight='bold')
    if i == 0 or i == 5:  # Original and Hough (RGB images)
        plt.imshow(im)
    else:
        plt.imshow(im, cmap=cmap)
    plt.axis('off')

plt.tight_layout()
plt.savefig(OUT_DIR + "comparison_filters.png", dpi=300, bbox_inches="tight")
plt.show()

