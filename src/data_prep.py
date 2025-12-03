import os
import yaml
from pathlib import Path
import shutil
from typing import Dict, List, Tuple
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class DatasetPreparator:
    """
    Prepares datasets for YOLOv8/Detectron2 training
    Supports COCO, YOLO, and custom formats
    """
    
    def __init__(self, raw_data_path: str, output_path: str = "data/processed"):
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.images_dir = self.output_path / "images"
        self.labels_dir = self.output_path / "labels"
        
        # Create train/val/test splits
        for split in ['train', 'val', 'test']:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
    
    def prepare_yolo_dataset(self, 
                            split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                            class_names: List[str] = None) -> Dict:
        """
        Prepare dataset in YOLO format
        
        Args:
            split_ratios: (train, val, test) ratios
            class_names: List of class names
        
        Returns:
            Dataset configuration dictionary
        """
        print("Preparing YOLO dataset...")
        
        # Get all image files
        image_files = list(self.raw_data_path.glob("**/*.jpg")) + \
                     list(self.raw_data_path.glob("**/*.png"))
        
        # Split dataset
        train_val, test = train_test_split(image_files, test_size=split_ratios[2], random_state=42)
        train, val = train_test_split(train_val, test_size=split_ratios[1]/(split_ratios[0]+split_ratios[1]), random_state=42)
        
        splits = {'train': train, 'val': val, 'test': test}
        
        # Copy files to appropriate directories
        for split_name, files in splits.items():
            for img_file in files:
                # Copy image
                dest_img = self.images_dir / split_name / img_file.name
                shutil.copy2(img_file, dest_img)
                
                # Copy corresponding label if exists
                label_file = img_file.with_suffix('.txt')
                if label_file.exists():
                    dest_label = self.labels_dir / split_name / label_file.name
                    shutil.copy2(label_file, dest_label)
        
        # Create dataset.yaml
        dataset_config = {
            'path': str(self.output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_names) if class_names else 80,
            'names': class_names if class_names else [f'class_{i}' for i in range(80)]
        }
        
        with open(self.output_path / 'dataset.yaml', 'w') as f:
            yaml.dump(dataset_config, f)
        
        print(f"✓ Dataset prepared: {len(train)} train, {len(val)} val, {len(test)} test")
        return dataset_config
    
    def convert_coco_to_yolo(self, coco_json_path: str, split: str = 'train'):
        """Convert COCO format annotations to YOLO format"""
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Create class name mapping
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Process each annotation
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            image_info = next(img for img in coco_data['images'] if img['id'] == image_id)
            
            # Get image dimensions
            img_width = image_info['width']
            img_height = image_info['height']
            
            # Convert bbox from [x, y, width, height] to YOLO format
            x, y, w, h = ann['bbox']
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height
            
            # Write to YOLO format file
            label_file = self.labels_dir / split / f"{image_info['file_name'].rsplit('.', 1)[0]}.txt"
            with open(label_file, 'a') as f:
                f.write(f"{ann['category_id']} {x_center} {y_center} {w_norm} {h_norm}\n")
        
        return categories
    
    def analyze_dataset(self) -> Dict:
        """Analyze prepared dataset statistics"""
        stats = {}
        
        for split in ['train', 'val', 'test']:
            images = list((self.images_dir / split).glob("*"))
            labels = list((self.labels_dir / split).glob("*"))
            
            # Count objects per class
            class_counts = {}
            for label_file in labels:
                with open(label_file, 'r') as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
            
            stats[split] = {
                'num_images': len(images),
                'num_labels': len(labels),
                'class_distribution': class_counts
            }
        
        return stats