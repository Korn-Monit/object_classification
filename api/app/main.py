from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from mobilevit import MobileViT
app = FastAPI()

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MODEL_PATH = "api\models\mobilevit_xxs_cifar10.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def mobilevit_xxs(num_classes=10):
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT((32, 32), dims, channels, num_classes=num_classes, expansion=2, patch_size=(1,1))
def load_model():
    model = mobilevit_xxs(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# --- API Endpoint ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process image
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs, 1)
        
        return {
            "class_id": predicted.item(),
            "class_name": CLASS_NAMES[predicted.item()],
            "confidence": torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
        }
    
    except Exception as e:
        return {"error": str(e)}