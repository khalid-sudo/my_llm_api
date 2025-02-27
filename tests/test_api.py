from fastapi.testclient import TestClient
from src.api.endpoints import app

client = TestClient(app)

def test_generate_endpoint():
    response = client.post(
        "/generate",
        json={"prompt": "What is the capital of France?"}
    )
    assert response.status_code == 200
    assert "response" in response.json()

def test_generate_endpoint_with_long_prompt():
    response = client.post(
        "/generate",
        json={
            "prompt": "test " * 1000,  # Very long prompt
            "max_length": 100
        }
    )
    assert response.status_code == 200
    assert "response" in response.json()