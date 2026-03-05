# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import segmentation_models_pytorch as smp
# from dataset_segmentation import LungSegmentationDataset
# # from transforms import SegmentationTransform

# # -------------------------------
# # CONFIG
# # -------------------------------
# DATA_ROOT = "data/raw"
# SAVE_DIR = "models/segmentation/weights"
# IMAGE_SIZE = 256
# BATCH_SIZE = 8
# EPOCHS = "max"
# LR = 1e-4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# os.makedirs(SAVE_DIR, exist_ok=True)

# # -------------------------------
# # DICE METRIC
# # -------------------------------
# def dice_score(pred, target, smooth=1e-6):
#     pred = torch.sigmoid(pred)
#     pred = (pred > 0.5).float()

#     intersection = (pred * target).sum(dim=(1,2,3))
#     union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))

#     dice = (2. * intersection + smooth) / (union + smooth)
#     return dice.mean()

# # -------------------------------
# # DATASETS & LOADERS
# # -------------------------------
# train_dataset = LungSegmentationDataset(
#     data_root=DATA_ROOT,
#     split="train",
#     image_size=IMAGE_SIZE,
#     transform=None
# )

# val_dataset = LungSegmentationDataset(
#     data_root=DATA_ROOT,
#     split="test",
#     image_size=IMAGE_SIZE,
#     transform=None
# )

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=0,
#     pin_memory=True
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=0,
#     pin_memory=True
# )

# if __name__ == '__main__':
#     print(f"✅ Train samples: {len(train_dataset)}")
#     print(f"✅ Val samples:   {len(val_dataset)}")

#     # -------------------------------
#     # MODEL
#     # -------------------------------
#     model = smp.UnetPlusPlus(
#         encoder_name="efficientnet-b4",
#         encoder_weights="imagenet",
#         in_channels=1,
#         classes=1
#     ).to(DEVICE)

#     # -------------------------------
#     # LOSS & OPTIMIZER
#     # -------------------------------
#     dice_loss = smp.losses.DiceLoss(mode="binary")
#     bce_loss = nn.BCEWithLogitsLoss()

#     def combined_loss(pred, target):
#         return dice_loss(pred, target) + bce_loss(pred, target)

#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#     # -------------------------------
#     # TRAINING LOOP
#     # -------------------------------
#     best_dice = 0.0

#     for epoch in range(EPOCHS):
#         model.train()
#         train_loss = 0.0
#         train_dice = 0.0

#         loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

#         for images, masks in loop:
#             images = images.to(DEVICE)
#             masks = masks.to(DEVICE)

#             optimizer.zero_grad()
#             outputs = model(images)

#             loss = combined_loss(outputs, masks)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             train_dice += dice_score(outputs, masks).item()

#             loop.set_postfix(
#                 loss=loss.item(),
#                 dice=train_dice / (loop.n + 1)
#             )

#         train_loss /= len(train_loader)
#         train_dice /= len(train_loader)

#         # -------------------------------
#         # VALIDATION
#         # -------------------------------
#         model.eval()
#         val_loss = 0.0
#         val_dice = 0.0

#         with torch.no_grad():
#             for images, masks in val_loader:
#                 images = images.to(DEVICE)
#                 masks = masks.to(DEVICE)

#                 outputs = model(images)
#                 loss = combined_loss(outputs, masks)

#                 val_loss += loss.item()
#                 val_dice += dice_score(outputs, masks).item()

#         val_loss /= len(val_loader)
#         val_dice /= len(val_loader)

#         print(f"\nEpoch {epoch+1}")
#         print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
#         print(f"Val   Loss: {val_loss:.4f} | Val   Dice: {val_dice:.4f}")

#         # -------------------------------
#         # SAVE BEST MODEL
#         # -------------------------------
#         if val_dice > best_dice:
#             best_dice = val_dice
#             save_path = os.path.join(SAVE_DIR, "best_unetpp_segmentation.pth")
#             torch.save(model.state_dict(), save_path)
#             print(f"🔥 Best model saved! Dice: {best_dice:.4f}")

#     print("✅ Training completed.")



# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import segmentation_models_pytorch as smp

# from dataset_segmentation import LungSegmentationDataset

# # -------------------------------
# # CONFIG (client-safe)
# # -------------------------------
# DATA_ROOT = "data/raw"
# SAVE_DIR = "models/segmentation/weights"

# IMAGE_SIZE = 256
# BATCH_SIZE = 8
# MAX_EPOCHS = 120              # upper bound
# PATIENCE = 10                 # early stopping patience
# LR = 1e-4

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", DEVICE)
# print("GPU:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU")

# os.makedirs(SAVE_DIR, exist_ok=True)

# # -------------------------------
# # DICE METRIC (ONLY METRIC)
# # -------------------------------
# def dice_score(pred, target, smooth=1e-6):
#     pred = torch.sigmoid(pred)
#     pred = (pred > 0.5).float()

#     intersection = (pred * target).sum(dim=(1, 2, 3))
#     union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

#     dice = (2.0 * intersection + smooth) / (union + smooth)
#     return dice.mean()

# # -------------------------------
# # DATA
# # -------------------------------
# train_dataset = LungSegmentationDataset(
#     data_root=DATA_ROOT,
#     split="train",
#     image_size=IMAGE_SIZE,
#     transform=None
# )

# val_dataset = LungSegmentationDataset(
#     data_root=DATA_ROOT,
#     split="test",
#     image_size=IMAGE_SIZE,
#     transform=None
# )

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=0
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=0
# )

# # -------------------------------
# # MAIN
# # -------------------------------
# if __name__ == "__main__":

#     print(f"✅ Train samples: {len(train_dataset)}")
#     print(f"✅ Val samples:   {len(val_dataset)}")

#     # -------------------------------
#     # MODEL
#     # -------------------------------
#     model = smp.UnetPlusPlus(
#         encoder_name="efficientnet-b4",
#         encoder_weights="imagenet",
#         in_channels=1,
#         classes=1
#     ).to(DEVICE)

#     # -------------------------------
#     # LOSS (internal only)
#     # -------------------------------
#     loss_fn = smp.losses.TverskyLoss(
#         mode="binary",
#         alpha=0.7,
#         beta=0.3
#     )

#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode="max",
#         factor=0.5,
#         patience=5,
#     )

#     # -------------------------------
#     # TRAINING LOOP (DICE-FOCUSED)
#     # -------------------------------
#     best_dice = 0.0
#     patience_counter = 0

#     for epoch in range(1, MAX_EPOCHS + 1):
#         model.train()
#         train_dice = 0.0

#         loop = tqdm(train_loader, desc=f"Epoch [{epoch}]")

#         for images, masks in loop:
#             images = images.to(DEVICE)
#             masks = masks.to(DEVICE)

#             optimizer.zero_grad()
#             outputs = model(images)

#             loss = loss_fn(outputs, masks)
#             loss.backward()
#             optimizer.step()

#             batch_dice = dice_score(outputs, masks).item()
#             train_dice += batch_dice

#             loop.set_postfix(dice=batch_dice)

#         train_dice /= len(train_loader)

#         # -------------------------------
#         # VALIDATION (ONLY DICE)
#         # -------------------------------
#         model.eval()
#         val_dice = 0.0

#         with torch.no_grad():
#             for images, masks in val_loader:
#                 images = images.to(DEVICE)
#                 masks = masks.to(DEVICE)

#                 outputs = model(images)
#                 val_dice += dice_score(outputs, masks).item()

#         val_dice /= len(val_loader)

#         print(f"\nEpoch {epoch}")
#         print(f"Train Dice: {train_dice:.4f}")
#         print(f"Val   Dice: {val_dice:.4f}")

#         scheduler.step(val_dice)

#         # -------------------------------
#         # CHECKPOINT + EARLY STOP
#         # -------------------------------
#         if val_dice > best_dice:
#             best_dice = val_dice
#             patience_counter = 0

#             torch.save(
#                 model.state_dict(),
#                 os.path.join(SAVE_DIR, "best_unetpp_segmentation.pth")
#             )

#             print(f"🔥 Best Dice improved to {best_dice:.4f}")

#         else:
#             patience_counter += 1
#             print(f"⏳ No Dice improvement ({patience_counter}/{PATIENCE})")

#         if patience_counter >= PATIENCE:
#             print("\n🛑 Early stopping triggered")
#             break

#     print(f"\n✅ Training completed. Best Val Dice: {best_dice:.4f}")



import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import segmentation_models_pytorch as smp

from dataset_segmentation import LungSegmentationDataset
from transforms import SegmentationTransform

# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = "data/raw"
SAVE_DIR = "models/segmentation/weights"

IMAGE_SIZE = 256
CROP_SIZE = 192
BATCH_SIZE = 8
MAX_EPOCHS = 120
PATIENCE = 10
LR = 1e-4

RESUME = False   # 🔁 set False for fresh training

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

LAST_CKPT = os.path.join(SAVE_DIR, "last_checkpoint.pth")
BEST_CKPT = os.path.join(SAVE_DIR, "best_unetpp_maxdice.pth")

print("Using device:", DEVICE)

# =========================================================
# DICE (NO THRESHOLD)
# =========================================================
def dice_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2.0 * intersection + smooth) / (union + smooth)

# =========================================================
# BOUNDARY LOSS
# =========================================================
def boundary_loss(pred, target):
    pred = torch.sigmoid(pred)

    sobel_x = torch.tensor(
        [[[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]],
        device=pred.device,
        dtype=torch.float32
    ).unsqueeze(1)

    sobel_y = sobel_x.transpose(2, 3)

    pred_edge_x = F.conv2d(pred, sobel_x, padding=1)
    pred_edge_y = F.conv2d(pred, sobel_y, padding=1)

    tgt_edge_x = F.conv2d(target, sobel_x, padding=1)
    tgt_edge_y = F.conv2d(target, sobel_y, padding=1)

    pred_edges = torch.sqrt(pred_edge_x**2 + pred_edge_y**2)
    tgt_edges = torch.sqrt(tgt_edge_x**2 + tgt_edge_y**2)

    return F.l1_loss(pred_edges, tgt_edges)

# =========================================================
# DATASETS
# =========================================================
train_transform = SegmentationTransform(image_size=IMAGE_SIZE)

train_dataset = LungSegmentationDataset(
    data_root=DATA_ROOT,
    split="train",
    image_size=IMAGE_SIZE,
    crop_size=CROP_SIZE,
    tumor_crop_prob=0.8,
    transform=train_transform
)

val_dataset = LungSegmentationDataset(
    data_root=DATA_ROOT,
    split="test",
    image_size=IMAGE_SIZE,
    crop_size=IMAGE_SIZE,
    tumor_crop_prob=0.0,
    transform=None
)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples:   {len(val_dataset)}")

# =========================================================
# WEIGHTED SAMPLER
# =========================================================
weights = []
for _, _, has_tumor in train_dataset:
    weights.append(5.0 if has_tumor else 1.0)

sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True
)

# =========================================================
# MODEL
# =========================================================
model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1
).to(DEVICE)

# =========================================================
# LOSS
# =========================================================
dice_loss = smp.losses.DiceLoss(mode="binary")
bce_loss = nn.BCEWithLogitsLoss()

def loss_fn(pred, target):
    return (
        dice_loss(pred, target)
        + bce_loss(pred, target)
        + 0.1 * boundary_loss(pred, target)
    )

# =========================================================
# OPTIMIZER & SCHEDULER
# =========================================================
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    patience=5,
    factor=0.5
)

# =========================================================
# RESUME LOGIC
# =========================================================
start_epoch = 1
best_dice = 0.0
patience_counter = 0

if RESUME and os.path.exists(LAST_CKPT):
    ckpt = torch.load(LAST_CKPT, map_location=DEVICE)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])

    start_epoch = ckpt["epoch"] + 1
    best_dice = ckpt["best_dice"]
    patience_counter = ckpt.get("patience_counter", 0)

    print(f"🔄 Resumed from epoch {start_epoch}")
    print(f"🔄 Best Dice so far: {best_dice:.4f}")

# =========================================================
# TRAINING LOOP
# =========================================================
for epoch in range(start_epoch, MAX_EPOCHS + 1):

    # -------------------- TRAIN --------------------
    model.train()
    train_dice = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}")

    for imgs, masks, _ in loop:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        preds = model(imgs)

        loss = loss_fn(preds, masks)
        loss.backward()
        optimizer.step()

        train_dice += dice_score(preds, masks).item()

    train_dice /= len(train_loader)

    # -------------------- VALIDATION (TTA) --------------------
    model.eval()
    inter, uni = 0.0, 0.0

    with torch.no_grad():
        for imgs, masks, _ in val_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            p1 = torch.sigmoid(model(imgs))
            p2 = torch.sigmoid(
                torch.flip(model(torch.flip(imgs, dims=[3])), dims=[3])
            )

            preds = (p1 + p2) / 2.0

            inter += (preds * masks).sum().item()
            uni += preds.sum().item() + masks.sum().item()

    val_dice = (2 * inter + 1e-6) / (uni + 1e-6)

    print("\n------------------------------------")
    print(f"Epoch {epoch}")
    print(f"Train Dice: {train_dice:.4f}")
    print(f"Val   Dice: {val_dice:.4f}")
    print("------------------------------------")

    scheduler.step(val_dice)

    # -------------------- SAVE LAST CHECKPOINT --------------------
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_dice": best_dice,
            "patience_counter": patience_counter
        },
        LAST_CKPT
    )

    # -------------------- SAVE BEST --------------------
    if val_dice > best_dice:
        best_dice = val_dice
        patience_counter = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_dice": best_dice
            },
            BEST_CKPT
        )

        print(f"🔥 New Best Dice: {best_dice:.4f}")

    else:
        patience_counter += 1
        print(f"⏳ No improvement ({patience_counter}/{PATIENCE})")

    if patience_counter >= PATIENCE:
        print("🛑 Early stopping triggered")
        break

print(f"\n✅ TRAINING DONE. BEST VAL DICE: {best_dice:.4f}")
