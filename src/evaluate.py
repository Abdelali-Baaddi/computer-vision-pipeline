import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

class ModelEvaluator:
    """
    Comprehensive model evaluation
    """
    
    def __init__(self, model_path: str, data_yaml: str):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.data_yaml = data_yaml
    
    def evaluate(self, split: str = 'test') -> Dict:
        """
        Comprehensive evaluation
        
        Args:
            split: Dataset split to evaluate ('val' or 'test')
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating on {split} set...")
        
        # Run validation
        results = self.model.val(data=self.data_yaml, split=split)
        
        # Extract metrics
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'fitness': float(results.fitness),
        }
        
        # Per-class metrics
        if hasattr(results.box, 'maps'):
            metrics['per_class_mAP'] = results.box.maps.tolist()
        
        print("\nEvaluation Results:")
        for key, value in metrics.items():
            if not isinstance(value, list):
                print(f"  {key}: {value:.4f}")
        
        return metrics
    
    def benchmark_speed(self, num_iterations: int = 100) -> Dict:
        """Benchmark inference speed"""
        import time
        import torch
        
        print(f"Benchmarking speed ({num_iterations} iterations)...")
        
        # Dummy input
        dummy_input = torch.randn(1, 3, 640, 640).to(self.model.device)
        
        # Warmup
        for _ in range(10):
            _ = self.model(dummy_input)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = self.model(dummy_input)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.time() - start)
        
        times = np.array(times) * 1000  # Convert to ms
        
        speed_metrics = {
            'mean_inference_time_ms': float(times.mean()),
            'std_inference_time_ms': float(times.std()),
            'min_inference_time_ms': float(times.min()),
            'max_inference_time_ms': float(times.max()),
            'fps': float(1000 / times.mean())
        }
        
        print("\nSpeed Benchmark:")
        for key, value in speed_metrics.items():
            print(f"  {key}: {value:.2f}")
        
        return speed_metrics
    
    def visualize_predictions(self, 
                            image_paths: List[str], 
                            save_dir: str = 'results/predictions'):
        """Visualize predictions on sample images"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for img_path in image_paths:
            results = self.model(img_path)
            
            # Save annotated image
            for idx, result in enumerate(results):
                result.save(filename=str(save_path / f"{Path(img_path).stem}_pred.jpg"))
        
        print(f"✓ Predictions saved to: {save_dir}")
    
    def generate_evaluation_report(self, 
                                  metrics: Dict,
                                  speed_metrics: Dict,
                                  save_path: str = 'results/evaluation_report.json'):
        """Generate comprehensive evaluation report"""
        report = {
            'accuracy_metrics': metrics,
            'speed_metrics': speed_metrics,
            'model_info': {
                'model_path': str(self.model.ckpt_path),
                'model_size': self.model.model.yaml.get('depth_multiple', 'unknown')
            }
        }
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Evaluation report saved to: {save_path}")
        return report