import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Page config
st.set_page_config(page_title="Helmet Detection", layout="centered")

st.title("Helmet Detection App")
st.write("Upload an image to detect helmet or no-helmet")

# Load model 
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # helmet trained model
    return model

model = load_model()

# Image upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("Run Detection"):
        with st.spinner("Detecting..."):
            results = model(np.array(image))

            # Result image
            result_img = results[0].plot()
            st.image(result_img, caption="Detection Result")

            # Show detections
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                st.success(f"Helmets detected: {len(boxes)}")
            else:
                st.warning("No helmet detected")



