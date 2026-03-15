# import os
# import cv2
# import torch
# from torch.utils.data import Dataset
# import numpy as np
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# CLASS_MAP = {
#     "ADC": 1,
#     "LCC": 2,
#     "SCC": 3
# }

# class LungCancerDataset(Dataset):
#     def __init__(self, data_root, split="train", image_size=256, transform=None):
#         self.image_size = image_size
#         self.transform = transform
#         self.samples = []

#         ct_root = os.path.join(data_root, split, "CT")
#         mask_root = os.path.join(data_root, split, "MASK")

#         # Collect all unique filenames
#         all_files = set()
#         for class_name in CLASS_MAP.keys():
#             ct_dir = os.path.join(ct_root, class_name)
#             if os.path.exists(ct_dir):
#                 for fname in os.listdir(ct_dir):
#                     if fname.endswith(('.png', '.jpg', '.jpeg')):
#                         all_files.add(fname)

#         # Create samples with image and mask paths
#         for fname in sorted(all_files):
#             ct_path = None
#             mask_paths = {}
            
#             # Find CT image
#             for class_name in CLASS_MAP.keys():
#                 ct_file = os.path.join(ct_root, class_name, fname)
#                 if os.path.exists(ct_file):
#                     ct_path = ct_file
#                     break
            
#             # Find all mask files for this image
#             for class_name, class_id in CLASS_MAP.items():
#                 mask_file = os.path.join(mask_root, class_name, fname)
#                 if os.path.exists(mask_file):
#                     mask_paths[class_id] = mask_file
            
#             if ct_path and mask_paths:
#                 self.samples.append({
#                     "img": ct_path,
#                     "masks": mask_paths,
#                     "filename": fname
#                 })

#         assert len(self.samples) > 0, "❌ Dataset is empty"

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         img_path = sample["img"]
#         mask_paths = sample["masks"]

#         # Load image
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             raise RuntimeError(f"Failed to load image: {img_path}")

#         # Create multi-class mask
#         mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
#         for class_id, mask_path in mask_paths.items():
#             class_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#             if class_mask is not None:
#                 mask[class_mask > 0] = class_id

#         # Resize
#         if self.transform:
#             augmented = self.transform(image=img, mask=mask)
#             img = augmented['image']
#             mask = augmented['mask']
#         else:
#             img = cv2.resize(img, (self.image_size, self.image_size))
#             mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            
#             # Normalize and convert to tensor
#             img = img.astype("float32") / 255.0
#             img = torch.from_numpy(img).unsqueeze(0)
#             mask = torch.from_numpy(mask)

#         # Ensure mask is Long tensor for CrossEntropyLoss
#         if isinstance(mask, torch.Tensor):
#             mask = mask.long()
#         else:
#             mask = torch.from_numpy(mask).long()

#         return img, mask


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

class LungCancerDataset(Dataset):
    def __init__(self, data_root, split="train", image_size=256, transform=None):
        self.image_size = image_size
        self.transform = transform
        self.samples = []

        ct_root = os.path.join(data_root, split, "CT")
        mask_root = os.path.join(data_root, split, "MASK")

        all_files = set()
        for class_name in CLASS_MAP.keys():
            ct_dir = os.path.join(ct_root, class_name)
            if os.path.exists(ct_dir):
                for fname in os.listdir(ct_dir):
                    if fname.endswith(('.png', '.jpg', '.jpeg')):
                        all_files.add(fname)

        for fname in sorted(all_files):
            ct_path = None
            mask_paths = {}

            for class_name in CLASS_MAP.keys():
                ct_file = os.path.join(ct_root, class_name, fname)
                if os.path.exists(ct_file):
                    ct_path = ct_file
                    break

            for class_name, class_id in CLASS_MAP.items():
                mask_file = os.path.join(mask_root, class_name, fname)
                if os.path.exists(mask_file):
                    mask_paths[class_id] = mask_file

            if ct_path and mask_paths:
                self.samples.append({
                    "img": ct_path,
                    "masks": mask_paths
                })

        assert len(self.samples) > 0, "❌ Dataset is empty"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = cv2.imread(sample["img"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load image")

        mask = np.zeros_like(img, dtype=np.uint8)

        for class_id, mask_path in sample["masks"].items():
            class_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if class_mask is not None:
                mask[class_mask > 0] = class_id

        img = cv2.resize(img, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        img = img.astype("float32") / 255.0
        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).long()

        return img, mask
