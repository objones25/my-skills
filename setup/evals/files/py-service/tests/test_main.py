from fastapi.testclient import TestClient

from app.main import app


def test_get_order_returns_id():
    assert TestClient(app).get("/orders/7").json() == {"id": 7}
