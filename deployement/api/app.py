from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
from ultralytics import YOLO
import io
from PIL import Image
import base64
import json
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="Real-time object detection using YOLOv8",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = Path("best.pt")
if not MODEL_PATH.exists():
    MODEL_PATH = "yolov8n.pt"  # Fallback to pretrained

model = YOLO(MODEL_PATH)

# Configuration
class DetectionConfig(BaseModel):
    confidence: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 300
    classes: Optional[List[int]] = None

class DetectionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

class BatchDetectionResult(BaseModel):
    detections: List[DetectionResult]
    image_shape: List[int]
    inference_time_ms: float

@app.get("/")
def read_root():
    return {
        "message": "YOLOv8 Object Detection API",
        "model": str(MODEL_PATH),
        "endpoints": {
            "POST /detect": "Single image detection",
            "POST /detect/batch": "Batch image detection",
            "POST /detect/video": "Video frame detection",
            "GET /health": "Health check",
            "GET /model/info": "Model information"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.get("/model/info")
def model_info():
    return {
        "model_path": str(MODEL_PATH),
        "model_type": "YOLOv8",
        "input_size": 640,
        "classes": model.names if hasattr(model, 'names') else []
    }

@app.post("/detect", response_model=BatchDetectionResult)
async def detect_objects(
    file: UploadFile = File(...),
    confidence: float = 0.25,
    iou_threshold: float = 0.45,
    classes: Optional[str] = None
):
    """
    Detect objects in a single image
    
    Args:
        file: Image file
        confidence: Confidence threshold (0-1)
        iou_threshold: IoU threshold for NMS
        classes: Comma-separated class IDs to detect (e.g., "0,1,2")
    
    Returns:
        Detection results with bounding boxes
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Parse classes
        class_filter = None
        if classes:
            class_filter = [int(c) for c in classes.split(",")]
        
        # Run inference
        import time
        start_time = time.time()
        
        results = model.predict(
            image,
            conf=confidence,
            iou=iou_threshold,
            classes=class_filter,
            verbose=False
        )[0]
        
        inference_time = (time.time() - start_time) * 1000
        
        # Parse results
        detections = []
        boxes = results.boxes
        
        for box in boxes:
            detection = DetectionResult(
                class_id=int(box.cls[0]),
                class_name=model.names[int(box.cls[0])],
                confidence=float(box.conf[0]),
                bbox=[float(x) for x in box.xyxy[0].tolist()]
            )
            detections.append(detection)
        
        return BatchDetectionResult(
            detections=detections,
            image_shape=list(image.shape),
            inference_time_ms=inference_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/annotated")
async def detect_and_annotate(
    file: UploadFile = File(...),
    confidence: float = 0.25,
    iou_threshold: float = 0.45
):
    """
    Detect objects and return annotated image
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run inference
        results = model.predict(
            image,
            conf=confidence,
            iou=iou_threshold,
            verbose=False
        )[0]
        
        # Annotate image
        annotated = results.plot()
        
        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', annotated)
        
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/batch")
async def batch_detect(
    files: List[UploadFile] = File(...),
    confidence: float = 0.25,
    iou_threshold: float = 0.45
):
    """
    Batch detection on multiple images
    """
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Run inference
            detections = model.predict(
                image,
                conf=confidence,
                iou=iou_threshold,
                verbose=False
            )[0]
            
            # Parse results
            det_list = []
            for box in detections.boxes:
                det_list.append({
                    "class_id": int(box.cls[0]),
                    "class_name": model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": [float(x) for x in box.xyxy[0].tolist()]
                })
            
            results.append({
                "filename": file.filename,
                "detections": det_list,
                "num_detections": len(det_list)
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results, "total_images": len(files)}

@app.post("/detect/video")
async def detect_video_frame(
    file: UploadFile = File(...),
    confidence: float = 0.25
):
    """
    Process video and return detections for each frame
    Note: For production, consider streaming or chunked processing
    """
    try:
        # Save uploaded video temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Process video
        cap = cv2.VideoCapture(temp_path)
        frame_results = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < 100:  # Limit to 100 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model.predict(frame, conf=confidence, verbose=False)[0]
            
            detections = []
            for box in results.boxes:
                detections.append({
                    "class_id": int(box.cls[0]),
                    "class_name": model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0])
                })
            
            frame_results.append({
                "frame": frame_count,
                "detections": detections
            })
            
            frame_count += 1
        
        cap.release()
        
        # Clean up
        Path(temp_path).unlink()
        
        return {
            "total_frames": frame_count,
            "results": frame_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)