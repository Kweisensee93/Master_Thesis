# pipeline_helpers/geometry.py
# Helpers for handling TPS coordinates and translation between coordinates and arrays
import numpy as np

#-----Helpers------------------
#1  flip_y:             Convert between TPS bottom-left origin and image top-left origin
#2  coords_to_array:    Convert list of (x, y) tuples to flat array
#3  array_to_coords:    Convert flat array back to list of (x, y) tuples
#4  euclidean:          Compute Euclidean distance between points
#5 to_crop_space:    Shift coordinates to crop space by subtracting crop box origin
#-----Finding Semilandmark-----
#4  find_anchor_points: Find contour points closest to first and last landmark (anchors for semilandmark curve)
#5  get_contour_subset: Extract contour segment between the two anchor points
#6  resample_points:    Resample the contour segment to get equidistant semilandmark points

def flip_y(coords: list[tuple], img_height: int) -> list[tuple]:
    """
    TPS bottom-left origin → image top-left origin.
    Needs the image height / y-axis length to flip the y-coordinates correctly.
    """
    return [(x, img_height - y) for x, y in coords]

def coords_to_array(coords: list[tuple]) -> np.ndarray:
    """Convert a list of (x, y) tuples to a flat float32 array [x0,y0,x1,y1,…]."""
    return np.array(coords, dtype=np.float32).flatten()

def array_to_coords(arr: np.ndarray) -> list[tuple]:
    """Convert a flat array [x0,y0,x1,y1,…] back to a list of (x, y) tuples."""
    arr = arr.reshape(-1, 2)
    return [(int(round(x)), int(round(y))) for x, y in arr]

def euclidean(a: np.ndarray, b: np.ndarray) -> float | np.ndarray:
    """
    Compute Euclidean distance between points.
    Behaviour depends on input shape:
    - Two single points (1D arrays): returns a scalar float
    - Array of points vs single point (2D vs 1D): returns 1D array of distances
    Parameters
    ----------
    a : np.ndarray  single point (2,) or array of points (N, 2)
    b : np.ndarray  single point (2,)
    Returns
    -------
    float or np.ndarray
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    diff = a - b # Handles x and y subtraction in one step
    return float(np.sqrt((diff ** 2).sum())) if a.ndim == 1 else np.sqrt((diff ** 2).sum(axis=1))

def to_crop_space(
    coords_full: list[tuple],
    crop_box: tuple,
) -> list[tuple]:
    """
    Shift full-image pixel coordinates into crop-local space.

    Parameters
    ----------
    coords_full : list of (x, y) in full-image pixel space (already y-flipped)
    crop_box    : (min_x, max_x, min_y, max_y)
    """
    ox, oy = crop_box[0], crop_box[2]
    return [(x - ox, y - oy) for x, y in coords_full]

#
def find_anchor_points(contour: np.ndarray, landmarks: list[tuple], crop_box: tuple) -> tuple[int, int, tuple, tuple]:
    """
    Finds coordinates and inices of the contour points closest to landmark 1 and 13, which are at the
    same time first and last landmark as well as first and last semilandmark.
    Parameters:
        contour: Numpy array (N, 2) of the fish outline (in crop space).
        landmarks: List of (x, y) tuples (in full image space).
        crop_box: (min_x, max_x, min_y, max_y) used to shift landmarks.
    Returns:
        pt_first: The actual (x, y) coordinate of the first anchor.
        pt_last: The actual (x, y) coordinate of the last anchor.
        idx_first: Index in the contour array for the start.
        idx_last: Index in the contour array for the end.
    """
    # 1. Shift landmarks into the same coordinate space as the contour (crop space)
    ox, oy = crop_box[0], crop_box[2]   # origin x,y of the crop box, which is the offset
    first_lm_crop = np.array([landmarks[0][0] - ox, landmarks[0][1] - oy])
    last_lm_crop = np.array([landmarks[-1][0] - ox, landmarks[-1][1] - oy])

    # 2. Reshape contour for distance calculation
    contour_pts = contour.reshape(-1, 2).astype(np.float32)

    # 3. Find closest points using Euclidean distance
    dists_first = euclidean(contour_pts, first_lm_crop)
    dists_last = euclidean(contour_pts, last_lm_crop)

    idx_first = int(np.argmin(dists_first))
    idx_last = int(np.argmin(dists_last))
    
    pt_first = tuple(contour_pts[idx_first].astype(int))
    pt_last = tuple(contour_pts[idx_last].astype(int))

    return pt_first, pt_last, idx_first, idx_last

def get_contour_subset(contour: np.ndarray, idx_start: int, idx_end: int) -> np.ndarray:
    """
    Extracts the segment of the contour between two indices, handling array wrap-around.
    """
    contour_pts = contour.reshape(-1, 2).astype(np.float32)
    
    if idx_start <= idx_end:
        subset = contour_pts[idx_start : idx_end + 1]
    else:
        # Case where the contour indices wrap around the end of the array
        subset = np.concatenate([contour_pts[idx_start:], contour_pts[: idx_end + 1]])
        
    return subset

def resample_points(points: np.ndarray, num_points: int = 10) -> np.ndarray:
    """
    Resamples a path of points into 'num_points' equidistant segments.
    """
    # Compute cumulative arc length
    diffs = np.diff(points, axis=0) #distance between consecutive points, but for x and y seperately
    seg_lengths = np.sqrt((diffs**2).sum(axis=1)) #euclidean distance between consecutive points, by a^2 + b^2 = c^2
    arc_length = np.concatenate([[0], np.cumsum(seg_lengths)]) # cumulative length along the line per point
    total_length = arc_length[-1] # cumulative length of the last point = total length of the line
    # equidistant points along the line, from 0 to total_length, with num_points in total
    target_distances = np.linspace(0, total_length, num_points) 
    
    resampled = []
    for d in target_distances:
        #look for the point closest to the mathematically ideal point at distance d along the line
        idx = np.searchsorted(arc_length, d) #find the index where d would fit in the arc_length array, which gives us the segment of the line we are on
        idx = np.clip(idx, 1, len(arc_length) - 1) #ensure idx is within bounds (we need at least 2 points to interpolate)
        
        # Linear interpolation
        t = (d - arc_length[idx - 1]) / (arc_length[idx] - arc_length[idx - 1] + 1e-8) #proportion along the segment we are, with a small epsilon to avoid division by zero
        pt = (1 - t) * points[idx - 1] + t * points[idx] #interpolated point at distance d along the line
        resampled.append(pt) #add the resampled point to the list
        
    return np.array(resampled)
