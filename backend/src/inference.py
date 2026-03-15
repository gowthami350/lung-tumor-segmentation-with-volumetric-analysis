# # backend/src/inference.py

# import cv2
# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# from model import UNet

# # ============================================================
# # CONFIG
# # ============================================================

# IMAGE_SIZE = 256
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CHECKPOINT_PATH = "checkpoints/best_model.pth"

# # ============================================================
# # INFERENCE
# # ============================================================

# def predict(image_path):
#     model = UNet().to(DEVICE)
#     checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.eval()

#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
#     img_norm = img_resized / 255.0

#     tensor = torch.tensor(img_norm).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         pred = model(tensor)[0, 0].cpu().numpy()

#     mask = (pred > 0.5).astype(np.uint8)

#     return img_resized, mask

# def visualize(image, mask):
#     overlay = image.copy()
#     overlay[mask == 1] = 255

#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 3, 1)
#     plt.title("CT Image")
#     plt.imshow(image, cmap="gray")

#     plt.subplot(1, 3, 2)
#     plt.title("Predicted Mask")
#     plt.imshow(mask, cmap="gray")

#     plt.subplot(1, 3, 3)
#     plt.title("Overlay")
#     plt.imshow(overlay, cmap="gray")

#     plt.tight_layout()
#     plt.show()

# # ============================================================
# # RUN
# # ============================================================

# if __name__ == "__main__":
#     img_path = "sample_ct.jpg"  # change this
#     image, mask = predict(img_path)
#     visualize(image, mask)


import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import UNet

# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/best_model.pth"

NUM_CLASSES = 4  # 0=BG, 1=ADC, 2=LCC, 3=SCC

# Color map for visualization
COLOR_MAP = {
    0: (0, 0, 0),       # Background - black
    1: (255, 0, 0),     # ADC - red
    2: (0, 255, 0),     # LCC - green
    3: (0, 0, 255),     # SCC - blue
}

# ============================================================
# INFERENCE
# ============================================================

def predict(image_path):
    model = UNet(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_norm = img_resized.astype(np.float32) / 255.0

    tensor = (
        torch.from_numpy(img_norm)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        output = model(tensor)              # (1,4,H,W)
        pred_mask = torch.argmax(output, dim=1)[0].cpu().numpy()

    return img_resized, pred_mask

# ============================================================
# VISUALIZATION
# ============================================================

def colorize_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in COLOR_MAP.items():
        color_mask[mask == cls] = color

    return color_mask

def visualize(image, mask):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    color_mask = colorize_mask(mask)

    overlay = cv2.addWeighted(image_rgb, 0.6, color_mask, 0.4, 0)

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.title("CT Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask (Classes)")
    plt.imshow(color_mask)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    img_path = "sample_ct.jpg"  # change this
    image, mask = predict(img_path)
    visualize(image, mask)
