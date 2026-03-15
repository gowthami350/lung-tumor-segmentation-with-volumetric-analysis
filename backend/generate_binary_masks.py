import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

RAW_ROOT = "data/raw/train/CT"
CLASSES = ["ADC", "LCC", "SCC"]
NUM_SAMPLES = 5


def generate_mask(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)

    _, mask = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean = np.zeros_like(mask)
        clean[labels == largest] = 255
        mask = clean

    return mask


# Collect all images
all_images = []
for cls in CLASSES:
    cls_dir = os.path.join(RAW_ROOT, cls)
    for f in os.listdir(cls_dir):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            all_images.append(os.path.join(cls_dir, f))

samples = random.sample(all_images, NUM_SAMPLES)

for idx, img_path in enumerate(samples, 1):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = generate_mask(img)

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay[mask > 0] = [255, 0, 0]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("CT Image")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Generated Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.suptitle(f"Sample {idx}")
    plt.show()
