"""验证真实SQLite状态边界、并发准入和HTTP异步任务契约，不调用GPU。"""

import base64
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_api import AUTH, FakeEngine, generate

from flux_engine.app import create_app
from flux_engine.config import Settings
from flux_engine.engine import InferenceOutOfMemory
from flux_engine.schemas import GenerateRequest, GenerateResponse
from flux_engine.task_store import TaskCapacityError, TaskStore

PAYLOAD = {"prompt": "a red cat", "width": 256, "height": 256}


def submit(client, key=None, **override):
    """每次测试显式传入认证及可选幂等键。"""
    headers = {**AUTH, **({"Idempotency-Key": key} if key else {})}
    return client.post("/v1/images/tasks", headers=headers, json={**PAYLOAD, **override})


def wait_status(client, ident, desired):
    """等待真实worker推进终态；超时只用于检测死锁，不测推理速度。"""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/v1/images/tasks/{ident}", headers=AUTH)
        assert response.status_code == 200
        task = response.json()
        if task["status"] in desired:
            return task
        time.sleep(0.01)
    pytest.fail(f"task did not reach {desired}")


@pytest.fixture
def blocked_service(tmp_path):
    """阻塞模型证明HTTP提交不等待GPU，并让测试控制并发交错。"""
    engine = FakeEngine()
    engine.entered, engine.release = threading.Event(), threading.Event()
    settings = Settings(_env_file=None, task_data_dir=tmp_path, api_key="test-secret")
    with TestClient(create_app(settings, engine)) as client:
        try:
            yield client, engine
        finally:
            engine.release.set()


def test_submit_returns_before_inference_and_result_survives_polling(blocked_service):
    client, engine = blocked_service
    response = submit(client, "same-operation", seed=0)
    assert response.status_code == 202
    task = response.json()
    assert engine.entered.wait(5)
    assert task["seed"] == 0
    assert response.headers["Location"] == f"/v1/images/tasks/{task['id']}"
    running = wait_status(client, task["id"], {"running"})
    assert "prompt" not in running and "images" not in running and "image_base64" not in running
    pending = client.get(response.headers["Location"] + "/result", headers=AUTH)
    assert pending.status_code == 409 and pending.headers["Retry-After"] == "2"
    assert client.get("/health/ready").json()["busy"] is True
    assert client.delete(response.headers["Location"], headers=AUTH).status_code == 409
    engine.release.set()
    done = wait_status(client, task["id"], {"succeeded"})
    for _ in range(2):
        result = client.get(done["result_url"], headers=AUTH)
        assert result.status_code == 200 and result.json()["seed"] == 0
        assert Image.open(BytesIO(base64.b64decode(result.json()["image_base64"]))).size == (
            256,
            256,
        )
    assert len(engine.calls) == 1


def test_concurrent_idempotent_retries_create_one_task(blocked_service):
    client, engine = blocked_service
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: submit(client, "network-retry"), range(12)))
    assert all(r.status_code == 202 for r in responses)
    ids = {r.json()["id"] for r in responses}
    seeds = {r.json()["seed"] for r in responses}
    assert len(ids) == len(seeds) == 1
    assert engine.entered.wait(5) and len(engine.calls) == 1
    assert submit(client, "network-retry", prompt="different").status_code == 409
    engine.release.set()
    wait_status(client, ids.pop(), {"succeeded"})
    assert submit(client, "network-retry").json()["status"] == "succeeded"
    assert len(engine.calls) == 1


def test_queue_admission_cancel_and_fifo(tmp_path):
    engine = FakeEngine()
    engine.entered, engine.release = threading.Event(), threading.Event()
    settings = Settings(
        _env_file=None, task_data_dir=tmp_path, task_queue_capacity=3, api_key="test-secret"
    )
    with TestClient(create_app(settings, engine)) as client:
        try:
            first = submit(client, prompt="first").json()
            assert engine.entered.wait(5)
            second = submit(client, "cancel-me", prompt="second").json()
            third = submit(client, prompt="third").json()
            full = submit(client, prompt="rejected")
            assert full.status_code == 429 and full.json()["detail"]["code"] == "queue_full"
            assert generate(client).status_code == 429
            cancelled = client.delete(f"/v1/images/tasks/{second['id']}", headers=AUTH)
            assert cancelled.json()["status"] == "cancelled"
            assert submit(client, "cancel-me", prompt="second").json()["status"] == "cancelled"
            fourth = submit(client, prompt="fourth").json()
        finally:
            engine.release.set()
        for task in (first, third, fourth):
            wait_status(client, task["id"], {"succeeded"})
        assert [call[0].prompt for call in engine.calls] == ["first", "third", "fourth"]
        assert (
            client.get(f"/v1/images/tasks/{second['id']}/result", headers=AUTH).status_code == 409
        )


def test_sync_and_async_never_infer_concurrently(blocked_service):
    client, engine = blocked_service
    with ThreadPoolExecutor() as pool:
        sync = pool.submit(generate, client)
        try:
            assert engine.entered.wait(5)
            task = submit(client).json()
            assert wait_status(client, task["id"], {"queued"})["status"] == "queued"
            assert len(engine.calls) == 1
        finally:
            engine.release.set()
        assert sync.result().status_code == 200
    wait_status(client, task["id"], {"succeeded"})
    assert len(engine.calls) == 2


@pytest.mark.parametrize(
    "error,code",
    [
        (RuntimeError("internal secret"), "generation_failed"),
        (InferenceOutOfMemory("internal secret"), "gpu_out_of_memory"),
    ],
)
def test_failure_is_safe_and_next_task_runs(tmp_path, error, code):
    engine = FakeEngine()
    engine.error = error
    settings = Settings(_env_file=None, task_data_dir=tmp_path, api_key="test-secret")
    with TestClient(create_app(settings, engine)) as client:
        task = submit(client, "failed-task").json()
        failed = wait_status(client, task["id"], {"failed"})
        assert failed["error"]["code"] == code and "internal secret" not in str(failed)
        assert submit(client, "failed-task").json()["status"] == "failed"
        assert len(engine.calls) == 1
        engine.error = None
        next_task = submit(client).json()
        wait_status(client, next_task["id"], {"succeeded"})
        assert len(engine.calls) == 2


def test_auth_validation_and_unknown_ids(tmp_path):
    engine = FakeEngine()
    settings = Settings(_env_file=None, task_data_dir=tmp_path, api_key="test-secret")
    with TestClient(create_app(settings, engine)) as client:
        ident = str(uuid4())
        for method, path in [
            ("POST", "/v1/images/tasks"),
            ("GET", f"/v1/images/tasks/{ident}"),
            ("GET", f"/v1/images/tasks/{ident}/result"),
            ("DELETE", f"/v1/images/tasks/{ident}"),
        ]:
            for headers in ({}, {"Authorization": "Bearer wrong"}):
                assert (
                    client.request(method, path, headers=headers, json=PAYLOAD).status_code == 401
                )
        for override in ({"images": ["bad"]}, {"width": 2048, "height": 2048}):
            assert submit(client, **override).status_code == 422
        for method, suffix in [("GET", ""), ("GET", "/result"), ("DELETE", "")]:
            assert (
                client.request(
                    method, f"/v1/images/tasks/{ident}{suffix}", headers=AUTH
                ).status_code
                == 404
            )
        assert not engine.calls


def test_submit_body_limit_and_bad_idempotency_key(tmp_path):
    settings = Settings(_env_file=None, task_data_dir=tmp_path, max_body_bytes=1024)
    with TestClient(create_app(settings, FakeEngine())) as client:
        assert submit(client, prompt="x" * 2000).status_code == 413
        assert submit(client, "x" * 129).status_code == 422


def test_restart_preserves_queue_results_and_marks_running_failed(tmp_path):
    settings = Settings(_env_file=None, task_data_dir=tmp_path)
    store = TaskStore(settings)
    store.open()
    try:
        completed = store.submit(GenerateRequest(prompt="complete", seed=0), "complete")
        store.claim()
        result = GenerateResponse(
            model=settings.model_id,
            seed=0,
            width=256,
            height=256,
            elapsed_seconds=1,
            image_base64="test-persisted-result",
        )
        store.finish(completed.id, result, None)
        running = store.submit(GenerateRequest(prompt="interrupted"), "running")
        store.claim()
        queued = store.submit(GenerateRequest(prompt="queued"), "queued")
    finally:
        store.close()
    restarted = TaskStore(settings)
    restarted.open()
    try:
        restarted.recover()
        assert restarted.get(completed.id).status == "succeeded"
        assert restarted.result(completed.id) == result
        assert restarted.get(running.id).error.code == "worker_interrupted"
        assert restarted.get(queued.id).status == "queued"
        assert restarted.submit(GenerateRequest(prompt="interrupted"), "running").id == running.id
        claimed_id, request, _ = restarted.claim()
        assert claimed_id == queued.id and request.seed == queued.seed
        assert restarted.claim() is None
    finally:
        restarted.close()


def test_restart_rejects_changed_model_and_single_process_lock(tmp_path):
    settings = Settings(_env_file=None, task_data_dir=tmp_path)
    first = TaskStore(settings)
    first.open()
    try:
        task = first.submit(GenerateRequest(prompt="old model"), None)
        with pytest.raises(RuntimeError, match="active worker"):
            TaskStore(settings).open()
        assert first.get(task.id).status == "queued"
    finally:
        first.close()
    second = TaskStore(settings.model_copy(update={"model_revision": "new-model"}))
    second.open()
    try:
        second.recover()
        assert second.get(task.id).error.code == "model_changed"
        assert second.claim() is None
    finally:
        second.close()


def test_retention_expiry_and_storage_capacity(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None,
        task_data_dir=tmp_path,
        task_queue_capacity=1,
        task_max_records=1,
        task_retention_seconds=60,
    )
    store = TaskStore(settings)
    store.open()
    try:
        task = store.submit(GenerateRequest(prompt="cancel"), "reuse-after-expiry")
        store.cancel(task.id)
        with pytest.raises(TaskCapacityError, match="task_storage_full"):
            store.submit(GenerateRequest(prompt="no space"), None)
        future = time.time() + 61
        monkeypatch.setattr("flux_engine.task_store.time.time", lambda: future)
        assert store.get(task.id) is None
        assert store.submit(GenerateRequest(prompt="cancel"), "reuse-after-expiry").id != task.id
    finally:
        store.close()


def test_persistence_failure_stops_admission_without_repeating_gpu(blocked_service, monkeypatch):
    client, engine = blocked_service

    def broken_write(*args):
        raise OSError("disk full")

    monkeypatch.setattr(client.app.state.task_store, "finish", broken_write)
    task = submit(client).json()
    assert engine.entered.wait(5)
    engine.release.set()
    deadline = time.monotonic() + 5
    while not client.app.state.task_worker.failed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.app.state.task_worker.failed
    assert client.get("/health/ready").status_code == 503
    assert submit(client).status_code == 503
    assert client.get(f"/v1/images/tasks/{task['id']}", headers=AUTH).json()["status"] == "running"
    assert len(engine.calls) == 1


def test_concurrent_distinct_submissions_cannot_overfill_queue(tmp_path):
    engine = FakeEngine()
    engine.entered, engine.release = threading.Event(), threading.Event()
    settings = Settings(
        _env_file=None, task_data_dir=tmp_path, task_queue_capacity=2, api_key="test-secret"
    )
    with TestClient(create_app(settings, engine)) as client:
        try:
            first = submit(client).json()
            assert engine.entered.wait(5)
            with ThreadPoolExecutor(max_workers=8) as pool:
                responses = list(pool.map(lambda i: submit(client, f"different-{i}"), range(10)))
            assert sum(r.status_code == 202 for r in responses) == 1
            assert sum(r.status_code == 429 for r in responses) == 9
        finally:
            engine.release.set()
        accepted = next(r.json() for r in responses if r.status_code == 202)
        for task in (first, accepted):
            wait_status(client, task["id"], {"succeeded"})
        assert len(engine.calls) == 2


def test_graceful_drain_and_restart_keep_results_and_queued_tasks(tmp_path):
    settings = Settings(_env_file=None, task_data_dir=tmp_path, api_key="test-secret")
    engine = FakeEngine()
    engine.entered, engine.release = threading.Event(), threading.Event()
    with TestClient(create_app(settings, engine)) as client:
        first = submit(client, "first-durable").json()
        assert engine.entered.wait(5)
        second = submit(client, "second-durable").json()
        with ThreadPoolExecutor() as pool:
            close = pool.submit(client.app.state.task_worker.close)
            try:
                assert client.app.state.task_worker.stopping.wait(5)
                assert engine.loaded and not close.done()
                assert submit(client).status_code == 503
            finally:
                engine.release.set()
            close.result(timeout=5)
        assert (
            client.get(f"/v1/images/tasks/{first['id']}", headers=AUTH).json()["status"]
            == "succeeded"
        )
        assert (
            client.get(f"/v1/images/tasks/{second['id']}", headers=AUTH).json()["status"]
            == "queued"
        )
    assert not engine.loaded and len(engine.calls) == 1
    replacement = FakeEngine()
    with TestClient(create_app(settings, replacement)) as client:
        wait_status(client, second["id"], {"succeeded"})
        assert submit(client, "first-durable").json()["id"] == first["id"]
        assert client.get(f"/v1/images/tasks/{first['id']}/result", headers=AUTH).status_code == 200
        assert len(replacement.calls) == 1


def test_crash_recovery_does_not_repeat_started_task(tmp_path):
    settings = Settings(_env_file=None, task_data_dir=tmp_path, api_key="test-secret")
    store = TaskStore(settings)
    store.open()
    try:
        interrupted = store.submit(GenerateRequest(**PAYLOAD), "interrupted")
        store.claim()
        queued = store.submit(GenerateRequest(**PAYLOAD), "queued")
    finally:
        store.close()
    engine = FakeEngine()
    with TestClient(create_app(settings, engine)) as client:
        wait_status(client, queued.id, {"succeeded"})
        task = client.get(f"/v1/images/tasks/{interrupted.id}", headers=AUTH).json()
        assert task["status"] == "failed" and task["error"]["code"] == "worker_interrupted"
        assert len(engine.calls) == 1 and engine.calls[0][2] == queued.seed


def test_failed_model_startup_releases_database_ownership(tmp_path):
    class FailingEngine(FakeEngine):
        def load(self):
            raise RuntimeError("model unavailable")

    settings = Settings(_env_file=None, task_data_dir=tmp_path, api_key="test-secret")
    with (
        pytest.raises(RuntimeError, match="model unavailable"),
        TestClient(create_app(settings, FailingEngine())),
    ):
        pass
    with TestClient(create_app(settings, FakeEngine())) as client:
        task = submit(client).json()
        wait_status(client, task["id"], {"succeeded"})


def test_response_loss_after_commit_reuses_task(blocked_service, monkeypatch):
    client, engine = blocked_service
    store = client.app.state.task_store
    original = store.submit
    accepted = []

    def lose_response(request, key):
        accepted.append(original(request, key))
        raise TimeoutError("response lost after durable acceptance")

    monkeypatch.setattr(store, "submit", lose_response)
    with pytest.raises(TimeoutError, match="response lost"):
        submit(client, "retry-after-lost-response")
    monkeypatch.setattr(store, "submit", original)
    retried = submit(client, "retry-after-lost-response")
    assert retried.status_code == 202 and retried.json()["id"] == accepted[0].id
    assert engine.entered.wait(5)
    engine.release.set()
    wait_status(client, accepted[0].id, {"succeeded"})
    assert len(engine.calls) == 1
