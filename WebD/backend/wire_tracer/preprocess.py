"""
wire_tracer/preprocess.py — Dual-Path Image Binarization
=========================================================
Converts a BGR circuit image to a clean binary wire mask using a
dual-path strategy validated on hand-drawn circuit images:

    Path A — Adaptive Gaussian Threshold
        Handles non-uniform illumination (shadows, pencil fading).

    Path B — Canny Edge Detection
        Catches faint strokes the adaptive threshold may miss.

Both paths are OR-merged, then repaired with a morphological close
to seal pen-skip gaps.
"""
import logging
import cv2
import numpy as np

logger = logging.getLogger("wire_tracer")


def preprocess_image(
    image_bgr: np.ndarray,
    blur_ksize: int = 5,
    adaptive_block_size: int = 15,
    adaptive_c: int = 8,
    morph_close_ksize: int = 7,
    morph_close_iterations: int = 1,
    min_blob_area: int = 20,
    # Canny edge detection (Path B)
    canny_low: int = 50,
    canny_high: int = 150,
    canny_blur_ksize: int = 9,
) -> np.ndarray:
    """
    Convert a BGR circuit image to a clean binary wire mask.

    Dual-path binarization pipeline:
        1. Convert to grayscale
        2. PATH A: Adaptive Gaussian threshold (handles uneven lighting)
        3. PATH B: Canny edge detection (catches faint pen strokes)
        4. OR-merge both paths
        5. Morphological CLOSE (bridges small gaps in hand-drawn lines)
        6. Small blob removal

    Returns:
        binary: np.ndarray, shape (H, W), dtype uint8, values in {0, 255}
    """
    if not isinstance(image_bgr, np.ndarray):
        raise ValueError("Input image must be a numpy array.")

    if image_bgr.size == 0:
        raise ValueError("Input image is empty.")

    # Check if image is already grayscale
    if len(image_bgr.shape) == 3 and image_bgr.shape[2] == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    elif len(image_bgr.shape) == 2:
        gray = image_bgr.copy()
    else:
        raise ValueError(f"Unsupported image shape: {image_bgr.shape}. Expected (H, W, 3) or (H, W).")

    h, w = gray.shape
    if h == 0 or w == 0:
        raise ValueError("Image has zero height or width.")

    if adaptive_block_size % 2 == 0:
        logger.warning(f"adaptive_block_size {adaptive_block_size} is even. Incrementing by 1.")
        adaptive_block_size += 1

    # Guard against image being smaller than block size
    if w < adaptive_block_size or h < adaptive_block_size:
        new_size = min(w, h)
        if new_size % 2 == 0:
            new_size -= 1
        if new_size < 3:
            raise ValueError(f"Image too small ({w}x{h}) for adaptive thresholding.")
        logger.warning(f"Image smaller than adaptive_block_size {adaptive_block_size}. Reducing to {new_size}.")
        adaptive_block_size = new_size

    # Check for uniform images (solid color)
    if np.all(gray == gray[0, 0]):
        return np.zeros_like(gray)

    # =========================================
    # PATH A — Adaptive Gaussian Threshold
    #   THRESH_BINARY_INV: dark ink → white px
    # =========================================
    _blur_a = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    _blur_a = max(1, _blur_a)

    blurred_a = cv2.GaussianBlur(gray, (_blur_a, _blur_a), 0) if _blur_a > 1 else gray

    adapt_map = cv2.adaptiveThreshold(
        blurred_a,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_block_size,
        adaptive_c
    )

    # =========================================
    # PATH B — Canny Edge Detection
    #   Heavier blur suppresses paper texture.
    # =========================================
    if canny_low > 0 and canny_high > 0:
        _blur_b = canny_blur_ksize if canny_blur_ksize % 2 == 1 else canny_blur_ksize + 1
        _blur_b = max(1, _blur_b)

        blurred_b = cv2.GaussianBlur(gray, (_blur_b, _blur_b), 0)
        canny_map = cv2.Canny(blurred_b, canny_low, canny_high)
        merged = cv2.bitwise_or(adapt_map, canny_map)
    else:
        merged = adapt_map

    if morph_close_ksize > 0 and morph_close_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_ksize, morph_close_ksize))
        binary = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iterations)
    else:
        binary = merged

    # Small blob removal
    if min_blob_area > 1:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA]
        small_labels = np.where((areas < min_blob_area) & (np.arange(num_labels) > 0))[0]

        if len(small_labels) > 0:
            mask_to_remove = np.isin(labels, small_labels)
            binary[mask_to_remove] = 0

    return binary
