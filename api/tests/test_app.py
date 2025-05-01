from fastapi.testclient import TestClient
from app import app  # Import your FastAPI app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200