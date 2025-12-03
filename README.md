# CV Detection Project

A small toolkit for training, evaluating, running inference, and deploying YOLOv8-based object detection models. Includes data preparation helpers, augmentation, a Streamlit demo UI, and a simple FastAPI inference service.

---

**Contents (high level)**

- Training utilities: `src/train.py` (YOLOTrainer)
- Data preparation: `src/data_prep.py` (DatasetPreparator)
- Augmentations: `src/augmentation.py` (Albumentations pipeline)
- Evaluation: `src/evaluate.py` (ModelEvaluator)
- Web demo: `deployement/web/streamlit_ui.py` (Streamlit UI)
- API: `deployement/api/app.py` (FastAPI inference service)
- Dependencies: `requirements.txt`

---

## Quick Start (Windows PowerShell)

1. Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

2. Prepare your dataset (example using `DatasetPreparator`):

```powershell
python - <<'PY'
from src.data_prep import DatasetPreparator
prep = DatasetPreparator(raw_data_path='path/to/raw_images', output_path='data/processed')
prep.prepare_yolo_dataset(class_names=['person','car'])
PY
```

3. Train a model (example using `YOLOTrainer` in `src/train.py`):

```powershell
python - <<'PY'
from src.train import YOLOTrainer
trainer = YOLOTrainer(model_size='n', img_size=640)
trainer.train(data_yaml='data/processed/dataset.yaml', epochs=50, batch_size=16)
PY
```

4. Run the Streamlit demo UI locally:

```powershell
streamlit run deployement/web/streamlit_ui.py
```

5. Run the FastAPI inference server:

```powershell
# From project root
# Run with uvicorn
uvicorn deployement.api.app:app --reload --host 0.0.0.0 --port 8000
# or (if the file is executable) run directly
python deployement/api/app.py
```

6. Call the API (example curl):

```powershell
# Single image detection
curl -X POST "http://localhost:8000/detect" -F "file=@/path/to/image.jpg" -F "confidence=0.3"

# Get annotated image
curl -X POST "http://localhost:8000/detect/annotated" -F "file=@/path/to/image.jpg" --output annotated.jpg
```

Notes:
- The FastAPI app looks for `best.pt` in the project root; if it isn't present it falls back to `yolov8n.pt` (pretrained). Place your trained `best.pt` at the project root or update `MODEL_PATH` in `deployement/api/app.py`.
- The Streamlit demo also expects a `best.pt` or will fall back to `yolov8n.pt`.

---

## Project Structure (explainers)

- `src/train.py` — `YOLOTrainer` class: initialization, `.train()`, `.validate()`, and `.export_model()` helpers. Uses `ultralytics.YOLO` under the hood.
- `src/data_prep.py` — `DatasetPreparator` to build YOLO-format datasets, convert COCO to YOLO, split data, and produce `dataset.yaml` used by training.
- `src/augmentation.py` — `AugmentationPipeline` (Albumentations) for training and validation augmentation.
- `src/evaluate.py` — `ModelEvaluator` to run validation, collect mAP/precision/recall, benchmark speed, and generate a JSON report.
- `deployement/web/streamlit_ui.py` — Streamlit app for quick demo and manual image/webcam testing.
- `deployement/api/app.py` — FastAPI service exposing `/detect`, `/detect/annotated`, `/detect/batch`, and `/detect/video` endpoints. Served with `uvicorn` in development.
- `requirements.txt` — List of required Python packages (install with `pip`).

---

## Docker (optional)

If you maintain Dockerfiles, put them in `deployement/docker/` (common patterns: `Dockerfile.serve`, `Dockerfile.train`). A typical serve build command looks like:

```powershell
# Example (adjust path to your Dockerfile)
docker build -f deployement/docker/Dockerfile.serve -t cv-serve .
docker run -p 8000:80 cv-serve
```

If your repository doesn't include Dockerfiles yet, you can containerize the FastAPI app and Streamlit app using standard base images (`python:3.11-slim`, `tiangolo/uvicorn-gunicorn`) and copying `best.pt` into the image.

---

## Tips & Troubleshooting

- GPU: Ensure `torch` (and CUDA toolkit) are installed correctly for GPU training and inference. The code checks `torch.cuda.is_available()`.
- Model files: Keep `best.pt` in the same directory as the API/Streamlit app or update paths in the scripts.
- Permissions: When using webcam in Streamlit, run the app locally (web browser camera access requires a secure context).
- Large datasets: Use dataset caching and appropriate `workers`/`batch` settings in `YOLOTrainer.train()`.

---

## Contributing

1. Open an issue describing the change or feature.
2. Create a branch, make changes, and submit a pull request.



---

## License & Contact

This repository currently has no explicit license file. Add a `LICENSE` MIT  to clarify reuse terms.

For questions or help, open an issue or contact the project owner.
