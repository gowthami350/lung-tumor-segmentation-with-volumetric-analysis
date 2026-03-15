import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from dataset import LungCancerDataset
from model import UNet
from utils import CombinedLoss, dice_per_class, save_checkpoint

# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = "data/raw"
IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 100  # Increased epochs
LR = 1e-4     # Lower initial LR for stability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BEST_CKPT = os.path.join(CHECKPOINT_DIR, "best_model_optimized.pth")
LAST_CKPT = os.path.join(CHECKPOINT_DIR, "last_model_optimized.pth")

# EARLY STOPPING
PATIENCE = 15
MIN_DELTA = 1e-4

NUM_CLASSES = 4  # 0=BG, 1=ADC, 2=LCC, 3=SCC

# ============================================================
# SEED
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ============================================================
# TRANSFORMS
# ============================================================

train_transform = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
    A.Rotate(limit=35, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.05),
    A.Normalize(
        mean=[0.0],
        std=[1.0],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
    A.Normalize(
        mean=[0.0],
        std=[1.0],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])

# ============================================================
# TRAINING LOOP
# ============================================================

def train():
    print(f"🚀 Training started on device: {DEVICE}")

    # Load Full Dataset
    full_dataset = LungCancerDataset(
        data_root=DATA_ROOT,
        split="train",
        image_size=IMAGE_SIZE,
        transform=None # We will apply transforms in subsets
    )

    # Split Train/Val
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # Assign Transforms manually (Hack for Subset)
    train_subset.dataset.transform = train_transform
    # Note: This assigns transform to the underlying dataset. 
    # Since we need different transforms for train and val, we should ideally instantiate two datasets.
    # Let's do that properly.
    
    train_dataset = LungCancerDataset(data_root=DATA_ROOT, split="train", image_size=IMAGE_SIZE, transform=train_transform)
    val_dataset = LungCancerDataset(data_root=DATA_ROOT, split="train", image_size=IMAGE_SIZE, transform=val_transform)
    
    # We need to ensure we split them consistently. 
    # Using random_split on the indices is better.
    indices = torch.randperm(len(train_dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    print(f"📊 Data Split: Train={len(train_subset)}, Val={len(val_subset)}")

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0, # Windows usually prefers 0 or fewer workers
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = UNet(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=5,
        factor=0.5,
        # verbose=True
    )
    # Class Weights (Background, ADC, LCC, SCC)
    class_weights = torch.tensor([0.1, 1.5, 1.5, 1.5]).to(DEVICE) 
    criterion = CombinedLoss(weight=class_weights, dice_weight=0.5)

    best_val_dice = 0.0
    start_epoch = 1
    epochs_without_improve = 0

    # ========================================================
    # RESUME TRAINING
    # ========================================================

    if os.path.exists(LAST_CKPT):
        print("🔄 Resuming training")
        checkpoint = torch.load(LAST_CKPT, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        if "best_dice" in checkpoint:
             best_val_dice = checkpoint["best_dice"]
        print(f"➡️ Resuming from epoch {start_epoch}, Best Val Dice: {best_val_dice:.4f}")

    # ========================================================
    # TRAIN LOOP
    # ========================================================

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_dice_scores = [0, 0, 0, 0]  # BG, ADC, LCC, SCC
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}] [Train]")
        for images, masks in loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            # Calculate training dice scores
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                batch_dice_scores = dice_per_class(preds, masks, NUM_CLASSES)
                for i in range(NUM_CLASSES):
                    train_dice_scores[i] += batch_dice_scores[i]
            
            # Update progress bar with loss and dice scores
            current_dice = [s / (loop.n + 1) for s in train_dice_scores]
            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                BG=f"{current_dice[0]:.3f}",
                ADC=f"{current_dice[1]:.3f}", 
                LCC=f"{current_dice[2]:.3f}",
                SCC=f"{current_dice[3]:.3f}"
            )

        train_loss /= len(train_loader)
        train_dice_scores = [s / len(train_loader) for s in train_dice_scores]

        # ====================================================
        # VALIDATION LOOP
        # ====================================================
        model.eval()
        val_loss = 0.0
        dice_scores = [0, 0, 0, 0] # BG, ADC, LCC, SCC

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                
                batch_dice_scores = dice_per_class(preds, masks, NUM_CLASSES)
                for i in range(NUM_CLASSES):
                    dice_scores[i] += batch_dice_scores[i]

        val_loss /= len(val_loader)
        dice_scores = [s / len(val_loader) for s in dice_scores]
        mean_val_dice = np.mean(dice_scores[1:]) # Ignore BG

        # Scheduler Step
        scheduler.step(mean_val_dice)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Mean Val Dice: {mean_val_dice:.4f} | "
            f"ADC: {dice_scores[1]:.3f} | LCC: {dice_scores[2]:.3f} | SCC: {dice_scores[3]:.3f}"
        )

        # ====================================================
        # SAVE & EARLY STOP
        # ====================================================

        if mean_val_dice > best_val_dice + MIN_DELTA:
            best_val_dice = mean_val_dice
            epochs_without_improve = 0
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_val_dice,
            }, BEST_CKPT)
            print(f"🏆 Best model saved (Dice = {best_val_dice:.4f})")
        else:
            epochs_without_improve += 1
            print(f"⏳ No improvement for {epochs_without_improve} epochs")

        save_checkpoint({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_val_dice,
        }, LAST_CKPT)

        if epochs_without_improve >= PATIENCE:
            print("⏹️ Early stopping triggered")
            break

    print("🎉 Training finished")
    print(f"🏆 Best Validation Dice: {best_val_dice:.4f}")

    # ====================================================
    # FINAL TEST EVALUATION
    # ====================================================
    print("\n🔬 Starting Evaluation on TEST Set...")

    test_dataset = LungCancerDataset(
        data_root=DATA_ROOT,
        split="test",
        image_size=IMAGE_SIZE,
        transform=val_transform # Use same transform as validation (no aug)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Load Best Model
    checkpoint = torch.load(BEST_CKPT, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loss = 0.0
    test_dice_scores = [0, 0, 0, 0]

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            
            batch_dice_scores = dice_per_class(preds, masks, NUM_CLASSES)
            for i in range(NUM_CLASSES):
                test_dice_scores[i] += batch_dice_scores[i]

    test_loss /= len(test_loader)
    test_dice_scores = [s / len(test_loader) for s in test_dice_scores]
    mean_test_dice = np.mean(test_dice_scores[1:])

    print("="*50)
    print(f"📄 TEST RESULTS")
    print(f"Loss: {test_loss:.4f}")
    print(f"Mean Dice: {mean_test_dice:.4f}")
    print(f"ADC Dice: {test_dice_scores[1]:.4f}")
    print(f"LCC Dice: {test_dice_scores[2]:.4f}")
    print(f"SCC Dice: {test_dice_scores[3]:.4f}")
    print("="*50)

if __name__ == "__main__":
    train()
