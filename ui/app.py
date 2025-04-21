import streamlit as st
import requests

API_URL = "https://my-fastapi-app-225654315168.us-central1.run.app/predict"

st.title("Image Classification with MobileViT")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    # Send image to FastAPI
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(API_URL, files=files)
    
    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['class_name']} (Confidence: {result['confidence']:.2f})")
    else:
        st.error("Prediction failed. Please try again.")