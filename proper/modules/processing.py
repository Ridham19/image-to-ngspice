import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None: raise ValueError("Image not found")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    return img, gray, thresh

def separate_layers(gray_img, binary_thresh):
    """
    Splits the image into 'Components' and 'Wires' using morphology.
    """
    # 1. Clean Text/Noise
    contours, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned_binary = binary_thresh.copy()
    ai_input_gray = gray_img.copy() 
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 400: # Filter small noise
            cv2.drawContours(cleaned_binary, [cnt], -1, 0, thickness=cv2.FILLED)
            cv2.drawContours(ai_input_gray, [cnt], -1, 255, thickness=cv2.FILLED)

    # 2. Extract Wires
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    wires_h = cv2.morphologyEx(cleaned_binary, cv2.MORPH_OPEN, h_kernel)
    wires_v = cv2.morphologyEx(cleaned_binary, cv2.MORPH_OPEN, v_kernel)
    
    wire_mask = cv2.bitwise_or(wires_h, wires_v)
    
    # 3. Extract Components (Binary - Wires)
    component_mask = cv2.bitwise_and(cleaned_binary, cv2.bitwise_not(wire_mask))
    
    # Heal fragmented components
    kernel = np.ones((3,3), np.uint8)
    component_mask = cv2.dilate(component_mask, kernel, iterations=2)
    
    return component_mask, wire_mask, ai_input_gray

def get_component_contours(component_mask):
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 15 and h > 15: valid_contours.append((x, y, w, h))
    return valid_contours