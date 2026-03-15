# # backend/src/evaluate.py

# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# from dataset import LungCancerBinaryDataset
# from model import UNet

# # ============================================================
# # CONFIG
# # ============================================================

# DATA_ROOT = "data/raw"
# IMAGE_SIZE = 256
# BATCH_SIZE = 8
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CHECKPOINT_PATH = "checkpoints/best_model.pth"

# # ============================================================
# # METRICS
# # ============================================================

# def dice_score(pred, target, smooth=1e-6):
#     pred = (pred > 0.5).float()
#     pred = pred.view(-1)
#     target = target.view(-1)
#     intersection = (pred * target).sum()
#     return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# def accuracy_score(pred, target):
#     pred = (pred > 0.5).float()
#     correct = (pred == target).float().sum()
#     return correct / torch.numel(pred)

# def dice_per_class(pred, target, num_classes=4, smooth=1e-6):
#     dice_scores = {}

#     for cls in range(1, num_classes):  # skip background
#         pred_cls = (pred == cls).float()
#         target_cls = (target == cls).float()

#         intersection = (pred_cls * target_cls).sum()
#         dice = (2 * intersection + smooth) / (
#             pred_cls.sum() + target_cls.sum() + smooth
#         )

#         dice_scores[cls] = dice.item()

#     return dice_scores


# # ============================================================
# # EVALUATION
# # ============================================================

# def evaluate():
#     print("🔍 Evaluating model on TEST set")

#     test_dataset = LungCancerBinaryDataset(
#         data_root=DATA_ROOT,
#         split="test",
#         image_size=IMAGE_SIZE
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=2
#     )

#     model = UNet().to(DEVICE)
#     checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.eval()

#     total_dice = 0.0
#     total_acc = 0.0

#     with torch.no_grad():
#         for images, masks in tqdm(test_loader):
#             images = images.to(DEVICE)
#             masks = masks.to(DEVICE)

#             preds = model(images)

#             total_dice += dice_score(preds, masks).item()
#             total_acc += accuracy_score(preds, masks).item()

#     avg_dice = total_dice / len(test_loader)
#     avg_acc = total_acc / len(test_loader)

#     print("✅ Evaluation completed")
#     print(f"📊 Test Dice Score: {avg_dice:.4f}")
#     print(f"📊 Test Accuracy:   {avg_acc:.4f}")

# if __name__ == "__main__":
#     evaluate()


import torch
from torch.utils.data import DataLoader
from dataset import LungCancerDataset
from model import UNet   # change if your model name is different

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def dice_score(pred, target, eps=1e-7):
    """
    Binary Dice Score
    pred, target: [B, H, W]
    """
    pred = pred.float()
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    return (2 * intersection + eps) / (union + eps)


def main():
    data_root = "data/raw"   # 🔁 change if needed
    batch_size = 4

    dataset = LungCancerDataset(
        data_root=data_root,
        split="test",  # Changed from "val" to "test"
        image_size=256
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = UNet(num_classes=4)   # background + 3 tumor classes
    
    # Try different checkpoint paths
    checkpoint_paths = [
        "checkpoints/best_model_optimized.pth",
        "checkpoints/best_model.pth",
        "checkpoints/last_model_optimized.pth",
        "checkpoints/best_fast_model.pth"
    ]
    
    checkpoint_loaded = False
    for path in checkpoint_paths:
        try:
            checkpoint = torch.load(path, map_location=DEVICE)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"✅ Loaded checkpoint: {path}")
            checkpoint_loaded = True
            break
        except FileNotFoundError:
            continue
    
    if not checkpoint_loaded:
        print("❌ No checkpoint found. Please train the model first.")
        return
    model.to(DEVICE)
    model.eval()

    dice_scores = []

    with torch.no_grad():
        for img, mask in loader:
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)

            logits = model(img)
            pred = torch.argmax(logits, dim=1)

            # 🔥 MULTI-CLASS → BINARY
            pred_bin = (pred > 0).long()
            mask_bin = (mask > 0).long()

            dice = dice_score(pred_bin, mask_bin)
            dice_scores.append(dice.item())

    mean_dice = sum(dice_scores) / len(dice_scores)
    print(f"✅ Combined Dice Score (Tumor vs Background): {mean_dice:.4f}")


if __name__ == "__main__":
    main()


# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# from dataset import LungCancerDataset
# from model import UNet
# from utils import dice_per_class

# # ============================================================
# # CONFIG
# # ============================================================

# DATA_ROOT = "data/raw"
# IMAGE_SIZE = 256
# BATCH_SIZE = 8
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CHECKPOINT_PATH = "checkpoints/best_model_optimized.pth"
# NUM_CLASSES = 4  # 0=BG, 1=ADC, 2=LCC, 3=SCC

# # ============================================================
# # TRANSFORMS
# # ============================================================

# val_transform = A.Compose([
#     A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
#     A.Normalize(
#         mean=[0.0],
#         std=[1.0],
#         max_pixel_value=255.0,
#     ),
#     ToTensorV2(),
# ])

# # ============================================================
# # EVALUATION
# # ============================================================

# def evaluate():
#     print("🔍 Evaluating model on TEST set")

#     test_dataset = LungCancerDataset(
#         data_root=DATA_ROOT,
#         split="test",
#         image_size=IMAGE_SIZE,
#         transform=val_transform
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=0
#     )

#     model = UNet(num_classes=NUM_CLASSES).to(DEVICE)
#     try:
#         checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
#         model.load_state_dict(checkpoint["model_state_dict"])
#     except FileNotFoundError:
#         print(f"❌ Checkpoint not found at {CHECKPOINT_PATH}. Please train the model first.")
#         return

#     model.eval()

#     dice_scores = [0, 0, 0, 0]

#     with torch.no_grad():
#         for images, masks in tqdm(test_loader):
#             images = images.to(DEVICE)
#             masks = masks.to(DEVICE)

#             outputs = model(images)
#             preds = torch.argmax(outputs, dim=1)

#             batch_dice_scores = dice_per_class(preds, masks, NUM_CLASSES)
#             for i in range(NUM_CLASSES):
#                 dice_scores[i] += batch_dice_scores[i]

#     num_batches = len(test_loader)
#     dice_scores = [s / num_batches for s in dice_scores]

#     print("✅ Evaluation completed\n")

#     print("📊 Per‑Class Dice Scores:")
#     print(f"   ADC (Class 1): {dice_scores[1]:.4f}")
#     print(f"   LCC (Class 2): {dice_scores[2]:.4f}")
#     print(f"   SCC (Class 3): {dice_scores[3]:.4f}")

#     mean_dice = sum(dice_scores[1:]) / 3
    
#     print(f"\n📊 Mean Dice (tumor classes): {mean_dice:.4f}")

# # ============================================================
# # ENTRY POINT
# # ============================================================

# if __name__ == "__main__":
#     evaluate()
