import base64
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from flux_engine.app import create_app
from flux_engine.config import Settings
from flux_engine.engine import InferenceOutOfMemory


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.loaded = False
        self.error = None
        self.entered = None
        self.release = None

    def load(self):
        self.loaded = True

    def close(self):
        self.loaded = False

    def generate(self, request, references, seed):
        self.calls.append((request, references, seed))
        if self.entered:
            self.entered.set()
            assert self.release.wait(5)
        if self.error:
            raise self.error
        return Image.new("RGB", (request.width, request.height), "red")


@pytest.fixture
def service(tmp_path):
    engine = FakeEngine()
    settings = Settings(_env_file=None, api_key="test-secret", task_data_dir=tmp_path)
    with TestClient(create_app(settings, engine)) as client:
        yield client, engine
    assert not engine.loaded


AUTH = {"Authorization": "Bearer test-secret"}


def generate(client, **kwargs):
    return client.post(
        "/v1/images/generations",
        headers=AUTH,
        json={"prompt": "a red cat", "width": 256, "height": 256, **kwargs},
    )


def test_health_and_auth(service):
    client, engine = service
    assert engine.loaded
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").json()["busy"] is False
    for headers in ({}, {"Authorization": "Bearer wrong"}):
        response = client.post("/v1/images/generations", headers=headers, json={"prompt": "cat"})
        assert response.status_code == 401
    assert not engine.calls


def test_text_to_image_png_and_seed_zero(service):
    client, engine = service
    response = generate(client, seed=0)
    assert response.status_code == 200
    result = response.json()
    assert result["seed"] == 0
    assert result["mime_type"] == "image/png"
    image = Image.open(BytesIO(base64.b64decode(result["image_base64"])))
    assert image.size == (256, 256)
    assert engine.calls[0][1] == []
    assert engine.calls[0][0].num_inference_steps == 4


def test_reference_images(service):
    client, engine = service
    buffer = BytesIO()
    Image.new("RGBA", (512, 256)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    assert generate(client, images=[encoded, encoded]).status_code == 200
    assert len(engine.calls[0][1]) == 2
    assert engine.calls[0][1][0].mode == "RGB"


@pytest.mark.parametrize(
    "override",
    [
        {"prompt": "  "},
        {"width": 257},
        {"width": 2048, "height": 2048},
        {"seed": -1},
        {"num_inference_steps": 0},
        {"unknown": True},
        {"images": ["not base64"]},
        {"images": [base64.b64encode(b"not an image").decode()]},
        {"images": ["abcd"] * 5},
    ],
)
def test_invalid_input_never_reaches_model(service, override):
    client, engine = service
    assert generate(client, **override).status_code == 422
    assert not engine.calls
    assert client.get("/health/ready").json()["busy"] is False


@pytest.mark.parametrize("chunked", [False, True])
def test_body_limit(chunked, tmp_path):
    app = create_app(
        Settings(_env_file=None, max_body_bytes=1024, task_data_dir=tmp_path), FakeEngine()
    )
    with TestClient(app) as client:
        body = b'{"prompt":"' + b"x" * 2000 + b'"}'
        content = iter([body[:500], body[500:]]) if chunked else body
        response = client.post(
            "/v1/images/generations", content=content, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 413


def test_busy_gpu_rejects_second_request(service):
    client, engine = service
    engine.entered = threading.Event()
    engine.release = threading.Event()
    with ThreadPoolExecutor() as executor:
        first = executor.submit(generate, client)
        try:
            assert engine.entered.wait(5)
            assert client.get("/health/ready").json()["busy"] is True
            second = generate(client)
            assert second.status_code == 429
            assert second.headers["Retry-After"] == "2"
        finally:
            engine.release.set()
        assert first.result().status_code == 200
    assert len(engine.calls) == 1


@pytest.mark.parametrize(
    ("error", "status"),
    [(InferenceOutOfMemory("oom"), 503), (RuntimeError("internal secret"), 500)],
)
def test_failure_releases_gpu_and_hides_internal_errors(service, error, status):
    client, engine = service
    engine.error = error
    response = generate(client)
    assert response.status_code == status
    assert "internal secret" not in response.text
    engine.error = None
    assert generate(client).status_code == 200


def test_model_not_ready(tmp_path):
    settings = Settings(_env_file=None, task_data_dir=tmp_path)
    with TestClient(create_app(settings, FakeEngine())) as client:
        client.app.state.ready = False
        assert client.get("/health/ready").status_code == 503
        assert generate(client).status_code == 503
