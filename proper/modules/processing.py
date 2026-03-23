import cv2
import numpy as np

def preprocess_image(image_path):
    original = cv2.imread(image_path)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # 1. Adaptive Thresholding (Automatically adjusts to shadows and bad lighting)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    # 2. Morphological Healing: Bridge tiny gaps in handwritten ink
    kernel = np.ones((3, 3), np.uint8)
    binary_healed = cv2.dilate(binary, kernel, iterations=1)
    
    return original, gray, binary_healed

def separate_layers(gray, binary):
    # Pass the fully healed binary mask straight through as the wire mask
    return None, binary, None