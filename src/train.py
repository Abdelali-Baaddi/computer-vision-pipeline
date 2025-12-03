from ultralytics import YOLO
import torch
from pathlib import Path
import yaml
from datetime import datetime
import json

class YOLOTrainer:
    """
    YOLOv8 Training Pipeline
    """
    
    def __init__(self, 
                 model_size: str = 'n',  # n, s, m, l, x
                 img_size: int = 640,
                 device: str = None):
        """
        Initialize trainer
        
        Args:
            model_size: YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)
            img_size: Input image size
            device: Training device ('cuda', 'cpu', or None for auto)
        """
        self.model_size = model_size
        self.img_size = img_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = YOLO(f'yolov8{model_size}.pt')
        
        print(f"✓ YOLOv8{model_size} initialized on {self.device}")
    
    def train(self,
              data_yaml: str,
              epochs: int = 100,
              batch_size: int = 16,
              lr0: float = 0.01,
              project: str = 'runs/train',
              name: str = None,
              resume: bool = False,
              **kwargs) -> dict:
        """
        Train the model
        
        Args:
            data_yaml: Path to dataset configuration
            epochs: Number of training epochs
            batch_size: Batch size
            lr0: Initial learning rate
            project: Project directory
            name: Experiment name
            resume: Resume training from checkpoint
            **kwargs: Additional training arguments
        
        Returns:
            Training results dictionary
        """
        if name is None:
            name = f"yolov8{self.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*80}")
        print(f"Starting training: {name}")
        print(f"{'='*80}")
        
        # Training configuration
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': self.img_size,
            'batch': batch_size,
            'device': self.device,
            'project': project,
            'name': name,
            'resume': resume,
            'lr0': lr0,
            'optimizer': 'AdamW',
            'patience': 50,
            'save': True,
            'save_period': 10,
            'cache': False,
            'workers': 8,
            'amp': True,  # Automatic Mixed Precision
            'verbose': True,
            **kwargs
        }
        
        # Train
        results = self.model.train(**train_args)
        
        print(f"\n✓ Training completed: {name}")
        print(f"  Best weights: {self.model.trainer.best}")
        
        return results
    
    def validate(self, data_yaml: str = None, **kwargs):
        """Validate the model"""
        val_args = {
            'data': data_yaml,
            'imgsz': self.img_size,
            'device': self.device,
            'verbose': True,
            **kwargs
        }
        
        results = self.model.val(**val_args)
        return results
    
    def export_model(self, 
                    format: str = 'onnx',
                    save_path: str = 'models/exported') -> str:
        """
        Export model to different formats
        
        Args:
            format: Export format ('onnx', 'torchscript', 'tflite', 'tensorrt', etc.)
            save_path: Path to save exported model
        
        Returns:
            Path to exported model
        """
        print(f"Exporting model to {format}...")
        
        export_path = Path(save_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        exported = self.model.export(
            format=format,
            imgsz=self.img_size,
            dynamic=False,
            simplify=True
        )
        
        print(f"✓ Model exported to: {exported}")
        return exported
    
    def save_training_config(self, config: dict, save_path: str):
        """Save training configuration"""
        config_path = Path(save_path) / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Config saved to: {config_path}")