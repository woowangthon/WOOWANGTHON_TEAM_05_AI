import json

from fastapi.testclient import TestClient
import httpx
import pytest

from app.main import app, Changes, Explanation, Interpretation


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("AI_PROVIDER", "ollama")
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


@pytest.fixture
def google_env(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "google")
    monkeypatch.setenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")
    monkeypatch.setenv("GEMINI_API_KEY", "test-google-key")


def mock_google(monkeypatch, body, status=200):
    requests = []

    async def handle(request):
        requests.append(request)
        return httpx.Response(status, json=body)

    real_client = httpx.AsyncClient
    monkeypatch.setattr("app.main.httpx.AsyncClient", lambda **kwargs: real_client(
        **kwargs, transport=httpx.MockTransport(handle)))
    return requests


def google_response(text, reason="STOP"):
    return {"candidates": [{"finishReason": reason, "content": {"parts": [
        {"text": "Internal reasoning", "thought": True}, {"text": text}]}}]}


def test_google_interpret_and_request_contract(monkeypatch, google_env):
    requests = mock_google(monkeypatch, google_response('```json\n{"changes":{"max_commute":60}}\n```'))
    with TestClient(app) as client:
        response = client.post("/interpret", json={"message": "통근 60분", "profile": {"max_commute": 45, "name": "private"}})
    assert response.status_code == 200
    assert response.json()["changes"]["max_commute"] == 60
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent"
    assert request.headers["x-goog-api-key"] == "test-google-key"
    assert "test-google-key" not in str(request.url)
    body = json.loads(request.content)
    assert "출력 스키마" in body["systemInstruction"]["parts"][0]["text"]
    payload = json.loads(body["contents"][0]["parts"][0]["text"])
    assert payload["profile"] == {"max_commute": 45}
    assert body["generationConfig"]["maxOutputTokens"] == 512
    assert "tools" not in body


def test_google_health_only_checks_metadata(monkeypatch, google_env):
    requests = mock_google(monkeypatch, {"name": "models/gemma-4-26b-a4b-it", "supportedGenerationMethods": ["generateContent"]})
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.json() == {"status": "ready", "provider": "google", "model": "gemma-4-26b-a4b-it"}
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert ":generateContent" not in str(requests[0].url)


@pytest.mark.parametrize("body", [
    {}, {"candidates": []}, {"candidates": [None]},
    google_response('{"changes":{"max_commute":60}}', "MAX_TOKENS"),
    google_response('not-json'), google_response('{"changes":{"max_commute":999}}'),
    google_response('{"changes":{"city":"Paris"}}'), google_response(''),
])
def test_google_invalid_outputs_rejected(monkeypatch, google_env, body):
    mock_google(monkeypatch, body)
    with TestClient(app) as client:
        assert client.post("/interpret", json={"message": "통근 60분", "profile": {}}).status_code == 503


@pytest.mark.parametrize("status,detail", [(429, "한도"), (403, "키"), (400, "설정"), (404, "모델"), (500, "연결")])
def test_google_errors_do_not_retry_or_expose_secrets(monkeypatch, google_env, status, detail):
    requests = mock_google(monkeypatch, {"error": {"message": "test-google-key"}}, status)
    with TestClient(app) as client:
        response = client.post("/interpret", json={"message": "통근 60분", "profile": {}})
    assert response.status_code == 503
    assert detail in response.json()["detail"]
    assert "test-google-key" not in response.text
    assert len(requests) == 1


@pytest.mark.parametrize("provider,model,key", [
    ("google", "gemma-4-26b-a4b-it", ""), ("google", "gemini-paid-model", "test-key"),
    ("google", "gemma3:4b", "test-key"), ("unknown", "gemma-4-26b-a4b-it", "test-key"),
])
def test_invalid_settings_fail_before_network(monkeypatch, provider, model, key):
    monkeypatch.setenv("AI_PROVIDER", provider)
    monkeypatch.setenv("GEMMA_MODEL", model)
    monkeypatch.setenv("GEMINI_API_KEY", key)
    requests = mock_google(monkeypatch, {})
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/interpret", json={"message": "통근 60분", "profile": {}}).status_code == 503
    assert requests == []


def test_google_explain_validates_citations(monkeypatch, google_env):
    mock_google(monkeypatch, google_response('{"focus":"budget","source_ids":["guide-1"]}'))
    payload = {"options": [], "guides": [{"id": "guide-1", "title": "안내", "text": "설명"}]}
    with TestClient(app) as client:
        assert client.post("/explain", json=payload).json() == {"focus": "budget", "source_ids": ["guide-1"]}
        assert client.post("/explain", json={"options": [], "guides": []}).status_code == 502


def test_google_defaults(monkeypatch, google_env):
    monkeypatch.delenv("AI_PROVIDER")
    monkeypatch.delenv("GEMMA_MODEL")
    mock_google(monkeypatch, {"name": "models/gemma-4-26b-a4b-it", "supportedGenerationMethods": ["generateContent"]})
    with TestClient(app) as client:
        assert client.get("/health").json()["model"] == "gemma-4-26b-a4b-it"
