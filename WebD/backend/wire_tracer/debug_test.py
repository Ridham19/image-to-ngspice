import cv2
import numpy as np
from preprocess import preprocess_image

img = np.ones((50, 50, 3), dtype=np.uint8) * 255
cv2.line(img, (10, 25), (20, 25), (0, 0, 0), 1)
cv2.line(img, (23, 25), (33, 25), (0, 0, 0), 1)

result = preprocess_image(img, blur_ksize=0, adaptive_block_size=11, morph_close_ksize=5, min_blob_area=0)
cv2.imwrite("debug_res.png", result)
print(cv2.connectedComponents(result)[0])
