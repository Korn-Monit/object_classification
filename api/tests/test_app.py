import pytest
from fastapi.testclient import TestClient
from app.main import app  # Import your FastAPI app
@pytest.fixture
def client():
    with TestClient(app, base_url="http://testserver") as client:
        yield client

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200