# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import numpy as np

# class SegmentationTransform:
#     """
#     Applies identical spatial transforms to image and mask.
#     Designed for binary lung tumor segmentation.
#     """

#     def __init__(self, image_size=256):
#         self.transform = A.Compose(
#             [
#                 A.Resize(image_size, image_size),

#                 # Spatial augmentations (safe for medical images)
#                 A.HorizontalFlip(p=0.5),
#                 A.VerticalFlip(p=0.5),
#                 A.Rotate(limit=10, border_mode=0, p=0.5),

#                 # Intensity normalization (CT safe)
#                 A.Normalize(
#                     mean=(0.0,),
#                     std=(1.0,),
#                     max_pixel_value=255.0
#                 ),

#                 ToTensorV2()
#             ]
#         )

#     def __call__(self, image, mask):
#         """
#         image: numpy array (H, W)
#         mask : numpy array (H, W) with values {0,1}
#         """
#         augmented = self.transform(image=image, mask=mask)

#         image = augmented["image"]          # shape: [1, H, W]
#         mask  = augmented["mask"].unsqueeze(0).float()  # shape: [1, H, W]

#         # Ensure binary mask
#         mask = (mask > 0).float()

#         return image, mask


import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class SegmentationTransform:
    """
    Safe augmentations for binary lung tumor segmentation.
    """

    def __init__(self, image_size=256):
        self.transform = A.Compose(
            [
                A.Resize(image_size, image_size),

                # ✅ SAFE spatial transforms
                A.HorizontalFlip(p=0.5),
                A.Rotate(
                    limit=10,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5
                ),

                # ❌ No Normalize here (already handled in dataset)
                ToTensorV2()
            ]
        )

    def __call__(self, image, mask):
        """
        image: numpy array (H, W) normalized to [0,1]
        mask : numpy array (H, W) with values {0,1}
        """
        augmented = self.transform(image=image, mask=mask)

        image = augmented["image"]           # [1, H, W]
        mask  = augmented["mask"].float()    # [H, W]

        mask = mask.unsqueeze(0)             # [1, H, W]

        return image, mask
