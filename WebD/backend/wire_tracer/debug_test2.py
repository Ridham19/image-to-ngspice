import cv2
import numpy as np

img = np.ones((50, 50, 3), dtype=np.uint8) * 255
cv2.line(img, (10, 25), (20, 25), (0, 0, 0), 1)
cv2.line(img, (23, 25), (33, 25), (0, 0, 0), 1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

print("binary:", (binary > 0).sum())
print("closed:", (closed > 0).sum())
