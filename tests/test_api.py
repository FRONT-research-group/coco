from fastapi.testclient import TestClient
from coco.app.main import app

client = TestClient(app)

def test_submit_data():
    response = client.post("/data/submit", json={
        "data": [
            {"label": "Privacy", "text": "Respect user data"},
            {"label": "Reliability", "text": "Ensure high uptime"},
            {"label": "Security", "text": "Use encryption"},
            {"label": "Privacy", "text": "Consent management"}
        ]
    })
    assert response.status_code == 200
    assert "count" in response.json()

def test_status():
    response = client.get("/data/status")
    assert response.status_code == 200
    assert "calculating" in response.json()
    assert "data_count" in response.json()

def test_calculate_and_get_nlotw():
    # Calculate
    response = client.post("/lotw/calculate")
    assert response.status_code == 200
    body = response.json()
    assert "nLoTw" in body

    # Get nLoTw
    response = client.get("/lotw/nlotw")
    assert response.status_code == 200
    nlotw = response.json()["nLoTw"]
    assert isinstance(nlotw, dict)
    assert abs(sum(nlotw.values()) - 100.0) < 1e-6
