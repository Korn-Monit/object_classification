import os
import torch
from .mobilevit import mobilevit_xxs
from google.cloud import storage
import asyncio
from fastapi import FastAPI

# Define a predictable path within the container's temporary filesystem
LOCAL_MODEL_PATH = "/tmp/model.pth"

def download_model_from_gcs(bucket_name: str, model_blob_name: str):
    """Downloads a model from GCS if it doesn't already exist locally."""
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"Model already found at {LOCAL_MODEL_PATH}. Skipping download.")
        return

    print(f"Downloading model gs://{bucket_name}/{model_blob_name} to {LOCAL_MODEL_PATH}")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(model_blob_name)
        
        os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
        
        blob.download_to_filename(LOCAL_MODEL_PATH)
        print("Model downloaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to download model from GCS: {e}")

def get_model():
    """Loads the MobileViT model from the local file, assuming it has been downloaded."""
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {LOCAL_MODEL_PATH}.")

    print(f"Loading model from {LOCAL_MODEL_PATH}...")
    try:
        model = mobilevit_xxs(num_classes=10)
        state_dict = torch.load(LOCAL_MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        print("Model loaded successfully.")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

async def load_model_background(app: FastAPI):
    """Asynchronously loads the model in the background."""
    loop = asyncio.get_event_loop()
    try:
        print("Starting background model loading...")
        bucket_name = os.getenv("GCS_BUCKET_NAME")
        model_blob_name = os.getenv("MODEL_BLOB_NAME")

        if not bucket_name or not model_blob_name:
            raise ValueError("GCS_BUCKET_NAME and MODEL_BLOB_NAME environment variables must be set")

        # Run blocking I/O in a separate thread to avoid blocking the event loop
        await loop.run_in_executor(None, download_model_from_gcs, bucket_name, model_blob_name)
        model = await loop.run_in_executor(None, get_model)
        
        app.state.model = model
        app.state.model_status = "ready"
        print("Background model loading complete. Model is ready.")
    except Exception as e:
        app.state.model_status = "error"
        print(f"Failed to load model in background: {e}")