# pipeline_helpers/normalisation.py
# LM1→LM13 relative coordinate system.
# Pose-, scale-, and rotation-invariant normalisation for semi-landmarks.

import numpy as np
from .geometry import coords_to_array

def get_lm_basis(
    landmarks: list[tuple],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute origin, unit axes, and scale from LM1 (index 0) and LM13 (index 12).

    Coordinate system
    -----------------
    Origin  : LM1
    X-axis  : unit vector LM1 → LM13
    Y-axis  : unit vector perpendicular (90° CCW)
    Scale   : pixel distance LM1 → LM13

    LM1 maps to (0, 0) and LM13 maps to (1, 0) in normalised space.
    (t, d) values represent fractional progress along / deviation from the axis.

    Returns
    -------
    origin, u_along, u_perp, scale
    """
    lm1  = np.array(landmarks[0],  dtype=np.float64)
    lm13 = np.array(landmarks[12], dtype=np.float64)

    vec   = lm13 - lm1
    scale = float(np.linalg.norm(vec))
    if scale < 1e-9:
        raise ValueError("LM1 and LM13 are coincident – cannot define axis.")

    u_along = vec / scale
    u_perp  = np.array([-u_along[1], u_along[0]])  # 90° CCW

    return lm1, u_along, u_perp, scale


def normalise_landmarks_relative(
    semi_landmarks_px: list[tuple],
    landmarks_px: list[tuple],
) -> np.ndarray:
    """
    Express semi-landmarks in the LM1→LM13 coordinate system.

    Each (x, y) becomes (t, d):
        t = projection onto LM1→LM13 axis  / LM1–LM13 distance
        d = projection onto perpendicular  / LM1–LM13 distance

    Returns
    -------
    np.ndarray  flat float32  [t0, d0, t1, d1, …]
    """
    origin, u_along, u_perp, scale = get_lm_basis(landmarks_px)

    normalised = []
    for x, y in semi_landmarks_px:
        v = np.array([x, y], dtype=np.float64) - origin
        t = float(np.dot(v, u_along)) / scale
        d = float(np.dot(v, u_perp))  / scale
        normalised.append((t, d))

    return coords_to_array(normalised)


def denormalise_landmarks_relative(
    arr: np.ndarray,
    landmarks_px: list[tuple],
) -> list[tuple]:
    """
    Inverse of normalise_landmarks_relative.
    Maps (t, d) back to full image pixel space using the LM1→LM13 basis.

    Parameters
    ----------
    arr          : flat float32  [t0, d0, t1, d1, …]
    landmarks_px : y-flipped fixed landmarks, needs >= 13 entries

    Returns
    -------
    list of (x, y) int pixel coordinates
    """
    origin, u_along, u_perp, scale = get_lm_basis(landmarks_px)

    result = []
    for t, d in arr.reshape(-1, 2):
        px = origin + (t * scale) * u_along + (d * scale) * u_perp
        result.append((int(round(px[0])), int(round(px[1]))))
    return result