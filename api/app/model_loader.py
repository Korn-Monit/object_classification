import os
import torch
from .mobilevit import mobilevit_xxs
from google.cloud import storage

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
        
        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
        
        blob.download_to_filename(LOCAL_MODEL_PATH)
        print("Model downloaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to download model from GCS: {e}")

def get_model():
    """Loads the MobileViT model from the local file, assuming it has been downloaded."""
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {LOCAL_MODEL_PATH}. Was it downloaded at startup?")

    print(f"Loading model from {LOCAL_MODEL_PATH}...")
    try:
        model = mobilevit_xxs(num_classes=10) # Assuming 10 classes for CIFAR-10/spots10
        # Load the state dictionary
        state_dict = torch.load(LOCAL_MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
