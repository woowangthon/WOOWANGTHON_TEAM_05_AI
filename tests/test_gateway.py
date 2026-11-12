import json

from fastapi.testclient import TestClient
import httpx
import pytest

from app.main import app, Changes, Explanation, Interpretation


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.setenv("GEMMA_MODEL", "gemma3:4b")


def mock_ollama(monkeypatch, content=None, fail=False):
    class Client:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def post(self, url, json):
            assert json["model"] == "gemma3:4b"
            assert json["format"]["type"] == "object"
            if fail:
                raise httpx.ConnectError("offline")
            return httpx.Response(200, json={"message": {"content": content}}, request=httpx.Request("POST", url))
        async def get(self, url):
            return httpx.Response(200, json={"models": [{"name": "gemma3:4b"}]}, request=httpx.Request("GET", url))
    monkeypatch.setattr("app.main.httpx.AsyncClient", Client)


def test_structured_patch_and_readiness(monkeypatch):
    mock_ollama(monkeypatch, json.dumps({"changes": {"max_commute": 60, "sort_by": "cheapest"}}))
    with TestClient(app) as client:
        response = client.post("/interpret", json={"message": "더 저렴하게, 통근 60분", "profile": {"max_commute": 45}})
        assert response.status_code == 200
        assert response.json()["changes"]["max_commute"] == 60
        assert client.get("/health").json()["status"] == "ready"


@pytest.mark.parametrize("content", ['{"changes":{"city":"Paris"}}', '{"changes":{"max_commute":999}}', 'not-json'])
def test_invalid_model_output_rejected(monkeypatch, content):
    mock_ollama(monkeypatch, content)
    with TestClient(app) as client:
        assert client.post("/interpret", json={"message": "바꿔줘", "profile": {}}).status_code == 503


def test_unknown_citation_and_outage(monkeypatch):
    mock_ollama(monkeypatch, json.dumps({"focus": "budget", "source_ids": ["invented"]}))
    with TestClient(app) as client:
        assert client.post("/explain", json={"options": [], "guides": []}).status_code == 502
    mock_ollama(monkeypatch, fail=True)
    with TestClient(app) as client:
        assert client.post("/interpret", json={"message": "바꿔줘", "profile": {}}).status_code == 503


def test_auth_and_only_gemma(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "secret")
    with TestClient(app) as client:
        assert client.post("/interpret", json={"message": "바꿔줘", "profile": {}}).status_code == 401
    monkeypatch.setenv("GEMMA_MODEL", "different-model")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503
