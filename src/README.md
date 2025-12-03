# src — Project source

This folder contains helper modules used by the project:

- `augmentation.py` — `AugmentationPipeline` using Albumentations.
- `data_prep.py` — `DatasetPreparator` to prepare YOLO-format datasets and convert COCO.
- `train.py` — `YOLOTrainer` wrapper around `ultralytics.YOLO` for training, validation, and export.
- `evaluate.py` — `ModelEvaluator` to run validation, benchmarks, and generate reports.

Quick usage examples:

- Import the trainer:

```python
from src.train import YOLOTrainer
trainer = YOLOTrainer(model_size='n')
trainer.train(data_yaml='data/processed/dataset.yaml')
```

- Prepare data:

```python
from src.data_prep import DatasetPreparator
prep = DatasetPreparator(raw_data_path='raw_images', output_path='data/processed')
prep.prepare_yolo_dataset(class_names=['person','car'])
```

Place this file in `src/` for quick reference in the editor.
