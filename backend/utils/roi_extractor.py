import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm

# ----------------------------------
# CONFIG
# ----------------------------------
DATA_ROOT = "data/raw"
OUTPUT_ROOT = "data/processed/classification"
MODEL_PATH = "models/segmentation/weights/best_unetpp_segmentation.pth"

IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = ["ADC", "LCC", "SCC"]

os.makedirs(OUTPUT_ROOT, exist_ok=True)
for cls in CLASSES:
    os.makedirs(os.path.join(OUTPUT_ROOT, cls), exist_ok=True)

# ----------------------------------
# LOAD SEGMENTATION MODEL
# ----------------------------------
model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights=None,
    in_channels=1,
    classes=1
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ----------------------------------
# HELPER: CT WINDOWING
# ----------------------------------
def ct_windowing(img):
    img = np.clip(img, 40 - 200, 40 + 200)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)

# ----------------------------------
# HELPER: GET BOUNDING BOX
# ----------------------------------
def get_bbox(mask):
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    return x_min, y_min, x_max, y_max

# ----------------------------------
# ROI EXTRACTION PIPELINE
# ----------------------------------
def extract_rois(split="train"):
    print(f"\n🔍 Extracting ROIs from {split} set")

    for cls in CLASSES:
        ct_dir = os.path.join(DATA_ROOT, split, "CT", cls)
        save_dir = os.path.join(OUTPUT_ROOT, cls)

        if not os.path.isdir(ct_dir):
            continue

        for fname in tqdm(os.listdir(ct_dir), desc=f"{cls}"):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(ct_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # CT windowing
            img = ct_windowing(img)

            # Prepare tensor
            img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img_tensor = (
                torch.from_numpy(img_resized)
                .unsqueeze(0)
                .unsqueeze(0)
                .float() / 255.0
            ).to(DEVICE)

            # Predict mask
            with torch.no_grad():
                pred = model(img_tensor)
                pred = torch.sigmoid(pred)
                pred_mask = (pred > 0.5).float()

            pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)

            # Resize mask back to original size
            pred_mask = cv2.resize(
                pred_mask,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            bbox = get_bbox(pred_mask)
            if bbox is None:
                continue

            x_min, y_min, x_max, y_max = bbox

            # Crop ROI
            roi = img[y_min:y_max, x_min:x_max]

            # Safety check
            if roi.size == 0:
                continue

            # Resize ROI for classifier
            roi = cv2.resize(roi, (224, 224))

            save_path = os.path.join(save_dir, fname)
            cv2.imwrite(save_path, roi)

    print("\n✅ ROI extraction completed")

# ----------------------------------
# MAIN
# ----------------------------------
if __name__ == "__main__":
    extract_rois(split="train")
    extract_rois(split="test")
