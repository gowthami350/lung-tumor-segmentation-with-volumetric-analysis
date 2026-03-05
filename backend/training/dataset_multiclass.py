import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

CLASS_MAP = {
    "ADC": 1,
    "LCC": 2,
    "SCC": 3
}

class LungMultiClassDataset(Dataset):
    def __init__(self, data_root, split="train", image_size=256):
        self.image_size = image_size
        self.samples = []

        ct_root = os.path.join(data_root, split, "CT")
        mask_root = os.path.join(data_root, split, "MASK")

        for cls, cls_id in CLASS_MAP.items():
            ct_dir = os.path.join(ct_root, cls)
            mask_dir = os.path.join(mask_root, cls)

            for fname in os.listdir(ct_dir):
                if fname.endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(ct_dir, fname)
                    mask_path = os.path.join(mask_dir, fname)

                    if os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path, cls_id))

        assert len(self.samples) > 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, cls_id = self.samples[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = np.clip(img, -160, 240)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8) * cls_id

        img = cv2.resize(img, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_NEAREST)

        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).long()

        return img, mask
