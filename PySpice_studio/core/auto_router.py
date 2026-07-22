"""
core/auto_router.py — Shared Helper Utilities
Provides pin-to-wire snapping and wire segment extraction for the
image-based wire-tracing pipeline.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List, Set, Dict


def snap_pin_to_wire(
    pin_x: int,
    pin_y: int,
    wire_mask: np.ndarray,
    radius: int = 15,
) -> Optional[Tuple[int, int]]:
    """
    Search a square neighbourhood of *radius* pixels around (pin_x, pin_y)
    on the binary wire_mask for the nearest white (255) pixel.

    Returns the snapped (x, y) coordinate or None if no white pixel is
    found within the search window.

    The search is optimised: we first check the pin itself, then spiral
    outward in increasing Manhattan-distance shells so we can early-exit
    as soon as the closest pixel is found.
    """
    h, w = wire_mask.shape[:2]

    # Clamp search window to image bounds
    y_lo = max(0, pin_y - radius)
    y_hi = min(h, pin_y + radius + 1)
    x_lo = max(0, pin_x - radius)
    x_hi = min(w, pin_x + radius + 1)

    # Quick check: if the pin is already on a wire pixel, return it directly
    if 0 <= pin_y < h and 0 <= pin_x < w and wire_mask[pin_y, pin_x] > 0:
        return (pin_x, pin_y)

    # Extract the local ROI and find all white pixel coordinates
    roi = wire_mask[y_lo:y_hi, x_lo:x_hi]
    ys, xs = np.where(roi > 0)

    if len(xs) == 0:
        return None

    # Convert back to absolute image coordinates
    abs_xs = xs + x_lo
    abs_ys = ys + y_lo

    # Compute squared Euclidean distances to the pin
    dx = abs_xs.astype(np.int32) - pin_x
    dy = abs_ys.astype(np.int32) - pin_y
    dists_sq = dx * dx + dy * dy

    best_idx = int(np.argmin(dists_sq))

    # Verify the best candidate is within the circular radius
    if dists_sq[best_idx] > radius * radius:
        return None

    return (int(abs_xs[best_idx]), int(abs_ys[best_idx]))


def extract_wire_segments(
    cluster_pixels: Set[Tuple[int, int]],
) -> List[Dict[str, int]]:
    """
    Given a set of pixel coordinates belonging to a single DFS cluster,
    produce a simplified polyline representation suitable for the JSON
    ``connections`` response.

    Strategy:
        1. Collect all cluster pixels into a contour-compatible array.
        2. Compute the convex hull to outline the wire path.
        3. Apply Douglas-Peucker approximation (epsilon = 2.0) to reduce
           the point count while preserving key direction changes.
        4. If the cluster is too small (< 3 pixels), return the raw
           bounding points directly.

    Returns a list of {\"x\": int, \"y\": int} dicts.
    """
    if not cluster_pixels:
        return []

    pts = np.array(list(cluster_pixels), dtype=np.int32)

    if len(pts) < 3:
        return [{"x": int(p[0]), "y": int(p[1])} for p in pts]

    # Reshape for OpenCV contour functions: (N, 1, 2) with (x, y) ordering
    contour = pts.reshape(-1, 1, 2)

    # Convex hull gives us the outer envelope of the wire cluster
    try:
        hull = cv2.convexHull(contour)
    except cv2.error:
        return [{"x": int(p[0]), "y": int(p[1])} for p in pts[:10]]

    # Douglas-Peucker simplification
    epsilon = 2.0
    approx = cv2.approxPolyDP(hull, epsilon, closed=False)

    points = []
    for pt in approx:
        points.append({"x": int(pt[0][0]), "y": int(pt[0][1])})

    return points


def find_skeleton_endpoints(
    cluster_pixels: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """
    Given a set of wire-path pixels, find the two endpoints of the wire
    segment (the pixels that are farthest apart).

    This is useful for straight or near-straight wires where we want
    to represent the wire as a simple two-point segment.

    Returns a list of up to 2 (x, y) tuples.
    """
    if len(cluster_pixels) < 2:
        return list(cluster_pixels)

    pts = np.array(list(cluster_pixels), dtype=np.int32)

    # Find the point farthest from the centroid → first endpoint
    centroid = pts.mean(axis=0)
    dists = np.sum((pts - centroid) ** 2, axis=1)
    idx_a = int(np.argmax(dists))
    pt_a = pts[idx_a]

    # Find the point farthest from pt_a → second endpoint
    dists_from_a = np.sum((pts - pt_a) ** 2, axis=1)
    idx_b = int(np.argmax(dists_from_a))
    pt_b = pts[idx_b]

    return [(int(pt_a[0]), int(pt_a[1])), (int(pt_b[0]), int(pt_b[1]))]
