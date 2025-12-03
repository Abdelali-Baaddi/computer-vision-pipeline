# File: deployment/web/streamlit_app.py
"""
Streamlit web interface for YOLOv8 detection
"""

import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import time

# Page config
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🎯",
    layout="wide"
)

# Title
st.title("🎯 YOLOv8 Object Detection")
st.markdown("Upload an image or use webcam for real-time detection")

# Sidebar
st.sidebar.header("⚙️ Settings")

# Model selection
model_path = st.sidebar.text_input("Model Path", value="best.pt")

# Load model
@st.cache_resource
def load_model(path):
    try:
        return YOLO(path)
    except:
        st.sidebar.warning("Model not found, using pretrained YOLOv8n")
        return YOLO("yolov8n.pt")
model_path = "best.pt"
model = load_model(model_path)

# Detection parameters
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

# Class filter
if hasattr(model, 'names'):
    available_classes = list(model.names.values())
    selected_classes = st.sidebar.multiselect(
        "Filter Classes",
        available_classes,
        default=[]
    )
else:
    selected_classes = []

# Input method
input_method = st.sidebar.radio("Input Method", ["Upload Image", "Webcam", "Example Images"])

# Main content
col1, col2 = st.columns(2)

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Display original
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_container_width=True)
        
        # Run detection
        with st.spinner("🔍 Detecting objects..."):
            start_time = time.time()
            
            # Get class IDs if filtered
            class_ids = None
            if selected_classes:
                class_ids = [k for k, v in model.names.items() if v in selected_classes]
            
            results = model.predict(
                image_np,
                conf=confidence,
                iou=iou_threshold,
                classes=class_ids,
                verbose=False
            )[0]
            
            inference_time = (time.time() - start_time) * 1000
        
        # Display results
        with col2:
            st.subheader("🎯 Detection Results")
            annotated = results.plot()
            st.image(annotated, use_container_width=True)
            
            # Metrics
            st.metric("Inference Time", f"{inference_time:.2f} ms")
            st.metric("Objects Detected", len(results.boxes))
        
        # Detection details
        st.subheader("📊 Detection Details")
        if len(results.boxes) > 0:
            detections_data = []
            for box in results.boxes:
                detections_data.append({
                    "Class": model.names[int(box.cls[0])],
                    "Confidence": f"{float(box.conf[0]):.3f}",
                    "BBox": f"({int(box.xyxy[0][0])}, {int(box.xyxy[0][1])}, {int(box.xyxy[0][2])}, {int(box.xyxy[0][3])})"
                })
            
            st.dataframe(detections_data, use_container_width=True)
        else:
            st.info("No objects detected")

elif input_method == "Webcam":
    st.info("Webcam feature requires running locally with proper permissions")
    
    enable_webcam = st.checkbox("Enable Webcam")
    
    if enable_webcam:
        picture = st.camera_input("Take a picture")
        
        if picture:
            image = Image.open(picture)
            image_np = np.array(image)
            
            with col1:
                st.subheader("📷 Captured Image")
                st.image(image, use_container_width=True)
            
            results = model.predict(image_np, conf=confidence, iou=iou_threshold, verbose=False)[0]
            
            with col2:
                st.subheader("🎯 Detection Results")
                annotated = results.plot()
                st.image(annotated, use_container_width=True)
                st.metric("Objects Detected", len(results.boxes))

else:  # Example Images
    st.info("Add example images to the 'examples/' directory")
    # You can add pre-loaded example images here

# Model info
with st.sidebar.expander("📌 Model Information"):
    st.write(f"**Model:** {model_path}")
    if hasattr(model, 'names'):
        st.write(f"**Classes:** {len(model.names)}")
        st.write(f"**Device:** {'GPU' if model.device.type == 'cuda' else 'CPU'}")