from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import os
import torch
from contextlib import asynccontextmanager
from .model_loader import download_model_from_gcs, get_model
from .mobilevit import transform

# Use a lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Application startup...")
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    model_blob_name = os.getenv("MODEL_BLOB_NAME")

    if not bucket_name or not model_blob_name:
        raise ValueError("GCS_BUCKET_NAME and MODEL_BLOB_NAME environment variables must be set")

    download_model_from_gcs(bucket_name, model_blob_name)
    app.state.model = get_model()
    print("Application startup complete.")
    yield
    # Shutdown logic (if any)
    print("Application shutdown.")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def read_health():
    """Health check endpoint to confirm the service is running."""
    return {"status": "ok"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Receives an image, processes it, and returns the model's prediction."""
    if not app.state.model:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    # Validate input file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Apply the same transformations as used during training
        processed_image = transform(image).unsqueeze(0) # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = app.state.model(processed_image)
            _, predicted = torch.max(outputs, 1)
            # Replace with your actual class names
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 
            prediction = class_names[predicted.item()]

        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
