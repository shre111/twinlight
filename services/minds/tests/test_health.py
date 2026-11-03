from fastapi.testclient import TestClient

from twl_minds.app import app


def test_health_reports_version() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.0.0"}
