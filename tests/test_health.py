"""Tests for the health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_contract() -> None:
    body = client.get("/health").json()
    assert body == {"status": "ok", "application": "PhD Academic RAG"}
