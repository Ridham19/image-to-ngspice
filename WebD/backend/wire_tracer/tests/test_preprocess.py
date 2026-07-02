import pytest
import numpy as np
import cv2
from wire_tracer.preprocess import preprocess_image

def test_grayscale_image_unchanged():
    # If already grayscale, pipeline should not crash.
    # We create a 50x50 grayscale image
    img = np.ones((50, 50), dtype=np.uint8) * 128
    result = preprocess_image(img, blur_ksize=3, adaptive_block_size=11, morph_close_ksize=0, min_blob_area=0)
    assert result.shape == (50, 50)
    assert result.dtype == np.uint8

def test_uniform_white_image_returns_empty_mask():
    # A pure white image should return all zeros after BINARY_INV thresholding.
    img = np.ones((50, 50, 3), dtype=np.uint8) * 255
    result = preprocess_image(img, blur_ksize=3, adaptive_block_size=11)
    assert np.all(result == 0)

def test_small_gap_bridged_by_close():
    # Draw two short lines with a 2px gap. After preprocessing with morph close, they should be connected.
    # Create white image
    img = np.ones((50, 50, 3), dtype=np.uint8) * 255
    # Draw black boxes (wire)
    cv2.rectangle(img, (10, 24), (20, 26), (0, 0, 0), -1)
    # Line 2: gap is 21, 22
    cv2.rectangle(img, (23, 24), (33, 26), (0, 0, 0), -1)
    
    # Process without morph close: should have gap
    result_no_close = preprocess_image(img, blur_ksize=0, adaptive_block_size=11, morph_close_ksize=0, min_blob_area=0)
    num_labels_no_close, _ = cv2.connectedComponents(result_no_close)
    # 0=bg, 1=line1, 2=line2 -> 3 labels total
    assert num_labels_no_close >= 3, "Without closing, should have >= 2 connected components for the wire."
    
    # Process with morph close (ksize=5 to easily bridge 2px gap)
    result_with_close = preprocess_image(img, blur_ksize=0, adaptive_block_size=11, morph_close_ksize=5, min_blob_area=0)
    num_labels_with_close, _ = cv2.connectedComponents(result_with_close)
    # 0=bg, 1=line (bridged) -> 2 labels total
    assert num_labels_with_close == 2, "With closing, the gap should be bridged into 1 connected component."

def test_adaptive_block_size_must_be_odd():
    # Passing an even block size should auto-correct and not raise.
    img = np.ones((50, 50, 3), dtype=np.uint8) * 200
    # Add some noise so it's not perfectly uniform, avoiding the uniform early exit
    img[20:30, 20:30] = 0
    # Pass 10 (even)
    try:
        result = preprocess_image(img, adaptive_block_size=10)
        assert result is not None
    except Exception as e:
        pytest.fail(f"preprocess_image raised an exception for even block size: {e}")
