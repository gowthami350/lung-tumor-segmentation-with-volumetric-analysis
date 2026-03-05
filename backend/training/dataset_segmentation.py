# import os
# import cv2
# import torch
# from torch.utils.data import Dataset
# import numpy as np

# class LungSegmentationDataset(Dataset):
#     def __init__(self, data_root, split="train", image_size=256, transform=None):
#         self.image_size = image_size
#         self.transform = transform
#         self.samples = []

#         ct_root = os.path.join(data_root, split, "CT")
#         mask_root = os.path.join(data_root, split, "MASK")

#         classes = ["ADC", "LCC", "SCC"]

#         for cls in classes:
#             ct_dir = os.path.join(ct_root, cls)
#             mask_dir = os.path.join(mask_root, cls)

#             for fname in os.listdir(ct_dir):
#                 if fname.endswith(('.png', '.jpg', '.jpeg')):
#                     ct_path = os.path.join(ct_dir, fname)
#                     mask_path = os.path.join(mask_dir, fname)

#                     if os.path.exists(mask_path):
#                         self.samples.append((ct_path, mask_path))

#         assert len(self.samples) > 0, "❌ Dataset is empty"

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         img_path, mask_path = self.samples[idx]

#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

#         # CT windowing (soft tissue window)
#         img = np.clip(img, 40 - 200, 40 + 200)
#         img = (img - img.min()) / (img.max() - img.min() + 1e-8)
#         img = (img * 255).astype(np.uint8)

#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

#         if img is None or mask is None:
#             raise RuntimeError("Failed to load image or mask")

#         # Binary mask
#         mask = (mask > 0).astype(np.uint8)

#         if self.transform:
#             augmented = self.transform(image=img, mask=mask)
#             img = augmented["image"]
#             mask = augmented["mask"]
#         else:
#             img = cv2.resize(img, (self.image_size, self.image_size))
#             mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

#             img = img.astype("float32") / 255.0
#             img = torch.from_numpy(img).unsqueeze(0)
#             mask = torch.from_numpy(mask).unsqueeze(0).float()

#         return img, mask


# import os
# import cv2
# import torch
# from torch.utils.data import Dataset
# import numpy as np

# class LungSegmentationDataset(Dataset):
#     def __init__(self, data_root, split="train", image_size=256, transform=None):
#         self.image_size = image_size
#         self.transform = transform
#         self.samples = []

#         ct_root = os.path.join(data_root, split, "CT")
#         mask_root = os.path.join(data_root, split, "MASK")

#         classes = ["ADC", "LCC", "SCC"]

#         for cls in classes:
#             ct_dir = os.path.join(ct_root, cls)
#             mask_dir = os.path.join(mask_root, cls)

#             if not os.path.isdir(ct_dir):
#                 continue

#             for fname in os.listdir(ct_dir):
#                 if fname.lower().endswith((".png", ".jpg", ".jpeg")):
#                     ct_path = os.path.join(ct_dir, fname)
#                     mask_path = os.path.join(mask_dir, fname)

#                     if os.path.exists(mask_path):
#                         self.samples.append((ct_path, mask_path))

#         assert len(self.samples) > 0, "❌ Dataset is empty"

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         img_path, mask_path = self.samples[idx]

#         # -------------------------------
#         # LOAD IMAGE
#         # -------------------------------
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             raise RuntimeError(f"Failed to load image: {img_path}")

#         # Convert to float BEFORE windowing
#         img = img.astype(np.float32)

#         # CT windowing (soft tissue)
#         img = np.clip(img, 40 - 200, 40 + 200)
#         img = (img - img.min()) / (img.max() - img.min() + 1e-8)
#         img = (img * 255).astype(np.uint8)


#         # -------------------------------
#         # LOAD MASK
#         # -------------------------------
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         if mask is None:
#             raise RuntimeError(f"Failed to load mask: {mask_path}")

#         # Binary mask
#         mask = (mask > 0).astype(np.uint8)

#         # -------------------------------
#         # TRANSFORMS OR FALLBACK
#         # -------------------------------
#         if self.transform:
#             augmented = self.transform(image=img, mask=mask)
#             img = augmented["image"]
#             mask = augmented["mask"]
#         else:
#             img = cv2.resize(img, (self.image_size, self.image_size))
#             mask = cv2.resize(
#                 mask,
#                 (self.image_size, self.image_size),
#                 interpolation=cv2.INTER_NEAREST
#             )

#             img = img.astype("float32") / 255.0
#             img = torch.from_numpy(img).unsqueeze(0)
#             mask = torch.from_numpy(mask).unsqueeze(0).float()

#         return img, mask


import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import random

class LungSegmentationDataset(Dataset):
    def __init__(
        self,
        data_root,
        split="train",
        image_size=256,
        crop_size=192,
        transform=None,
        tumor_crop_prob=0.8
    ):
        self.image_size = image_size
        self.crop_size = crop_size
        self.transform = transform
        self.tumor_crop_prob = tumor_crop_prob
        self.split = split
        self.samples = []

        ct_root = os.path.join(data_root, split, "CT")
        mask_root = os.path.join(data_root, split, "MASK")

        classes = ["ADC", "LCC", "SCC"]

        for cls in classes:
            ct_dir = os.path.join(ct_root, cls)
            mask_dir = os.path.join(mask_root, cls)

            if not os.path.isdir(ct_dir):
                continue

            for fname in os.listdir(ct_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    ct_path = os.path.join(ct_dir, fname)
                    mask_path = os.path.join(mask_dir, fname)

                    if os.path.exists(mask_path):
                        self.samples.append((ct_path, mask_path))

        assert len(self.samples) > 0, "❌ Dataset is empty"

    def __len__(self):
        return len(self.samples)

    def tumor_focused_crop(self, img, mask):
        h, w = mask.shape

        ys, xs = np.where(mask > 0)

        # If no tumor pixels → random crop
        if len(xs) == 0 or random.random() > self.tumor_crop_prob:
            x1 = random.randint(0, max(0, w - self.crop_size))
            y1 = random.randint(0, max(0, h - self.crop_size))
        else:
            cx = int(xs.mean())
            cy = int(ys.mean())

            x1 = np.clip(cx - self.crop_size // 2, 0, w - self.crop_size)
            y1 = np.clip(cy - self.crop_size // 2, 0, h - self.crop_size)

        img = img[y1:y1 + self.crop_size, x1:x1 + self.crop_size]
        mask = mask[y1:y1 + self.crop_size, x1:x1 + self.crop_size]

        return img, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # -------- IMAGE --------
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32)

        # CT windowing
        img = np.clip(img, 40 - 200, 40 + 200)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # -------- MASK --------
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        # -------- TUMOR-FOCUSED CROP (TRAIN ONLY) --------
        if self.split == "train":
            img, mask = self.tumor_focused_crop(img, mask)

        # -------- RESIZE TO MODEL INPUT --------
        img = cv2.resize(img, (self.image_size, self.image_size))
        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        has_tumor = 1.0 if mask.sum() > 0 else 0.0

        return img, mask, has_tumor
