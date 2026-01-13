import os
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm

# --- Configuration ---
SOURCE_DIR = "dataset_clean_v1"
TARGET_DIR = "dataset_noisy_v1"
IMAGES_PER_CLASS = 100

def apply_augmentations(img):
    """Applies random noise, color shifts, and slight blurs."""
    # 1. Random Color/Brightness Shift
    # alpha: contrast [0.8, 1.2], beta: brightness [-20, 20]
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-20, 20)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # 2. Add Random Gaussian Noise
    row, col = img.shape
    mean = 0
    sigma = random.uniform(5, 15) # Random intensity of noise
    gauss = np.random.normal(mean, sigma, (row, col))
    img = np.clip(img + gauss, 0, 255).astype('uint8')

    # 3. Random Slight Blur (Simulating focus issues)
    if random.choice([True, False]):
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # 4. Random Rotation (Slight)
    angle = random.uniform(-5, 5)
    M = cv2.getRotationMatrix2D((col//2, row//2), angle, 1)
    img = cv2.warpAffine(img, M, (col, row), borderValue=255) # Fill with white

    return img

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: {SOURCE_DIR} not found!")
        return

    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    os.makedirs(TARGET_DIR)

    classes = [f for f in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, f))]
    print(f"Augmenting {len(classes)} classes...")

    for cls in tqdm(classes):
        src_cls_path = os.path.join(SOURCE_DIR, cls)
        dst_cls_path = os.path.join(TARGET_DIR, cls)
        os.makedirs(dst_cls_path, exist_ok=True)

        # Get the 4 clean images we generated previously
        clean_images = [f for f in os.listdir(src_cls_path) if f.endswith('.png')]
        
        if not clean_images:
            continue

        for i in range(IMAGES_PER_CLASS):
            # Pick one of the 4 clean images as a base
            base_img_name = random.choice(clean_images)
            img = cv2.imread(os.path.join(src_cls_path, base_img_name), cv2.IMREAD_GRAYSCALE)
            
            # Apply random noise and shifts
            augmented_img = apply_augmentations(img)
            
            # Save
            save_path = os.path.join(dst_cls_path, f"{cls}_aug_{i}.png")
            cv2.imwrite(save_path, augmented_img)

    print(f"\nDone! New dataset created in '{TARGET_DIR}' with 100 images per folder.")

if __name__ == "__main__":
    main()