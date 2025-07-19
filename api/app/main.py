from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import torch
from contextlib import asynccontextmanager
import asyncio
from .model_loader import load_model_background
from .mobilevit import transform

# Use a lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Application startup...")
    app.state.model = None
    app.state.model_status = "loading" # States: loading, ready, error
    
    # Start model loading in the background
    asyncio.create_task(load_model_background(app))
    
    print("Application startup complete. Model is loading in the background.")
    yield
    # Shutdown logic (if any)
    print("Application shutdown.")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def read_health():
    """Health check endpoint to confirm the service is running."""
    return {"status": "ok", "model_status": app.state.model_status}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Receives an image, processes it, and returns the model's prediction."""
    if app.state.model_status == "loading":
        raise HTTPException(status_code=503, detail="Model is not ready yet, please try again later.")
    if app.state.model_status == "error" or not app.state.model:
        raise HTTPException(status_code=500, detail="Model failed to load.")

    # Validate input file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        processed_image = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = app.state.model(processed_image)
            _, predicted = torch.max(outputs, 1)
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 
            prediction = class_names[predicted.item()]

        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")