import logging
import cv2
import numpy as np

logger = logging.getLogger("wire_tracer")

def preprocess_image(
    image_bgr: np.ndarray,
    blur_ksize: int = 3,
    adaptive_block_size: int = 15,
    adaptive_c: int = 4,
    morph_close_ksize: int = 3,
    morph_close_iterations: int = 1,
    min_blob_area: int = 20,
) -> np.ndarray:
    """
    Convert a BGR circuit image to a clean binary wire mask.

    Pipeline:
        1. Convert to grayscale
        2. Gaussian blur (reduces sensor noise without destroying line edges)
        3. Adaptive threshold (handles uneven lighting across the paper)
        4. Morphological CLOSE (dilate then erode: bridges small gaps in hand-drawn lines)
        5. Small blob removal

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
        # Instead of failing, just adjust block size to be smaller than the image, and odd
        new_size = min(w, h)
        if new_size % 2 == 0:
            new_size -= 1
        if new_size < 3:
            raise ValueError(f"Image too small ({w}x{h}) for adaptive thresholding.")
        logger.warning(f"Image smaller than adaptive_block_size {adaptive_block_size}. Reducing to {new_size}.")
        adaptive_block_size = new_size

    # Check for all-black or all-white images which might crash thresholding if not handled, 
    # but adaptiveThreshold handles them gracefully usually. However, if image is solid, 
    # adaptive threshold might just return noise. Let's let OpenCV handle it, or check:
    if np.all(gray == gray[0, 0]):
        # Uniform image. If it's pure white, wires should be empty.
        # In inverted binary, dark pens are 255, white paper is 0.
        # If it's a uniform image, we should probably just return all 0s.
        return np.zeros_like(gray)

    # 1. Blur
    if blur_ksize > 0:
        # ensure odd ksize
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    else:
        blurred = gray

    # 2. Adaptive Threshold (BINARY_INV)
    # Dark lines become 255 (foreground), light background becomes 0.
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_block_size,
        adaptive_c
    )

    # 3. Morphological Close
    if morph_close_ksize > 0 and morph_close_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_ksize, morph_close_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iterations)

    # 4. Small blob removal
    if min_blob_area > 1:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        # stats: [x, y, w, h, area]
        
        # We start from label 1 because 0 is background
        # Vectorized removal:
        # Find which labels have area < min_blob_area
        # area is at index 4
        areas = stats[:, cv2.CC_STAT_AREA]
        
        # Mask of labels to keep. Label 0 is kept implicitly because we only zero out.
        # Wait, if we keep label 0, its area is usually huge.
        small_labels = np.where((areas < min_blob_area) & (np.arange(num_labels) > 0))[0]
        
        if len(small_labels) > 0:
            # Zero out the small blobs
            # Using np.isin is efficient for this
            mask_to_remove = np.isin(labels, small_labels)
            binary[mask_to_remove] = 0

    return binary
