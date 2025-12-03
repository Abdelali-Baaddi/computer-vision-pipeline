import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class AugmentationPipeline:
    """
    Data augmentation pipeline using Albumentations
    """
    
    def __init__(self, img_size: int = 640, mode: str = 'train'):
        self.img_size = img_size
        self.mode = mode
        self.transform = self._build_transform()
    
    def _build_transform(self):
        """Build augmentation pipeline"""
        if self.mode == 'train':
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(p=0.2),
                A.RandomResizedCrop(
                    height=self.img_size, 
                    width=self.img_size, 
                    scale=(0.8, 1.0),
                    p=0.5
                ),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.GaussianBlur(p=0.1),
                ], p=0.3),
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.1),
                ], p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def __call__(self, image, bboxes, class_labels):
        """Apply augmentation"""
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        return transformed