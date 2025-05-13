import pytest
from fastapi.testclient import TestClient
from coco.app.main import app

@pytest.fixture
def client():
    return TestClient(app)
