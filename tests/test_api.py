def test_submit_data(client):
    response = client.post("/data/submit", json={"text": ["some text", "more text"]})
    assert response.status_code == 200
    assert response.json()["count"] == 2

def test_status(client):
    response = client.get("/data/status")
    assert response.status_code == 200
    assert "calculating" in response.json()

def test_calculate_lotw(client):
    client.post("/data/submit", json={"text": ["hello", "world"]})
    response = client.post("/lotw/calculate")
    assert response.status_code == 200
    assert "cLoTw" in response.json()

def test_get_clotw(client):
    client.post("/data/submit", json={"text": ["a", "b"]})
    client.post("/lotw/calculate")
    response = client.get("/lotw/clotw")
    assert response.status_code == 200
    assert isinstance(response.json()["cLoTw"], float)

def test_get_nlotw(client):
    client.post("/data/submit", json={"text": ["x y", "z"]})
    client.post("/lotw/calculate")
    response = client.get("/lotw/nlotw")
    assert response.status_code == 200
    assert isinstance(response.json()["nLoTw"], float)
