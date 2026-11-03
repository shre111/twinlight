from fastapi.testclient import TestClient

from twl_runner.app import app


def test_health_reports_engine_version() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "engine_version": "0.0.0"}
