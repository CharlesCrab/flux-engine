import logging
import secrets
import threading
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from flux_engine.config import Settings
from flux_engine.engine import FluxEngine, InferenceOutOfMemory
from flux_engine.generation import execute_generation, validate_generation
from flux_engine.schemas import GenerateRequest, GenerateResponse, TaskResponse
from flux_engine.task_store import TaskCapacityError, TaskConflictError, TaskStore
from flux_engine.task_worker import TaskWorker

logger = logging.getLogger(__name__)
bearer = HTTPBearer(auto_error=False)


class BodyLimitMiddleware:
    """Bound JSON memory use, including chunked requests, before parsing the body."""

    def __init__(self, app, limit: int):
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.limit:
                response = JSONResponse({"detail": "request body too large"}, status_code=413)
                return await response(scope, receive, send)
            chunks.append(message)
            if not message.get("more_body", False):
                break
        iterator = iter(chunks)

        async def replay():
            message = next(iterator, None)
            return message if message is not None else await receive()

        await self.app(scope, replay, send)


def create_app(settings: Settings | None = None, engine=None) -> FastAPI:
    """同步和异步接口共享模型、认证及唯一GPU执行权。"""
    settings = settings or Settings()
    engine = engine if engine is not None else FluxEngine(settings)
    lock = threading.Lock()
    store = TaskStore(settings)
    worker = TaskWorker(store, engine, lock)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """持久化执行权先于模型加载，退出时先等待工作线程再卸载模型。"""
        app.state.ready = False
        await run_in_threadpool(store.open)
        try:
            await run_in_threadpool(engine.load)
            try:
                await run_in_threadpool(store.recover)
                worker.start()
                app.state.ready = True
                try:
                    yield
                finally:
                    app.state.ready = False
                    await run_in_threadpool(worker.close)
            finally:
                await run_in_threadpool(engine.close)
        finally:
            store.close()

    app = FastAPI(title="FLUX.2 klein 9B", version="0.2.0", lifespan=lifespan)
    app.state.ready = False
    app.state.task_store = store
    app.state.task_worker = worker
    app.add_middleware(BodyLimitMiddleware, limit=settings.max_body_bytes)

    def authenticate(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ):
        expected = settings.api_key.get_secret_value()
        if expected and (
            credentials is None
            or not secrets.compare_digest(credentials.credentials.encode(), expected.encode())
        ):
            raise HTTPException(401, "invalid API key", headers={"WWW-Authenticate": "Bearer"})

    def require_ready():
        """执行器故障时停止新的推理准入，保留状态查询接口。"""
        if not app.state.ready or worker.failed or worker.stopping.is_set():
            raise HTTPException(503, "model or task worker not ready")

    def validate(request):
        """提交前校验所有输入，坏图片不持久化也不触发GPU。"""
        try:
            validate_generation(request, settings.max_pixels)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready():
        require_ready()
        return {"status": "ready", "model": settings.model_id, "busy": lock.locked()}

    @app.post(
        "/v1/images/generations",
        response_model=GenerateResponse,
        dependencies=[Depends(authenticate)],
    )
    def generate(request: GenerateRequest):
        """保留同步契约；已接收的异步任务优先获得GPU。"""
        require_ready()
        validate(request)
        if store.has_pending() or not lock.acquire(blocking=False):
            raise HTTPException(429, "GPU busy; retry later", headers={"Retry-After": "2"})
        try:
            seed = request.seed if request.seed is not None else secrets.randbits(63)
            return execute_generation(engine, request, seed, settings.model_id)
        except InferenceOutOfMemory:
            raise HTTPException(503, "GPU memory exhausted; reduce image size") from None
        except Exception:
            logger.error("Synchronous image inference failed")
            raise HTTPException(500, "image generation failed") from None
        finally:
            lock.release()
            worker.wakeup.set()

    @app.post(
        "/v1/images/tasks",
        status_code=202,
        response_model=TaskResponse,
        dependencies=[Depends(authenticate)],
    )
    def submit_task(
        request: GenerateRequest,
        response: Response,
        idempotency_key: Annotated[
            str | None,
            Header(alias="Idempotency-Key", min_length=1, max_length=128, pattern=r"^[!-~]+$"),
        ] = None,
    ):
        """先原子持久化任务，再返回202；同键同输入复用原任务。"""
        require_ready()
        validate(request)
        try:
            task = store.submit(request, idempotency_key)
        except TaskConflictError as exc:
            raise HTTPException(409, str(exc)) from None
        except TaskCapacityError as exc:
            raise HTTPException(
                429,
                {"code": exc.code, "message": "Task capacity reached; retry later"},
                headers={"Retry-After": "60" if exc.code == "task_storage_full" else "2"},
            ) from None
        response.headers["Location"] = f"/v1/images/tasks/{task.id}"
        response.headers["Retry-After"] = "2"
        worker.wakeup.set()
        return task

    @app.get(
        "/v1/images/tasks/{task_id}",
        response_model=TaskResponse,
        dependencies=[Depends(authenticate)],
    )
    def task_status(task_id: UUID):
        """状态响应不携带提示词、参考图或大体积图片结果。"""
        task = store.get(str(task_id))
        if task is None:
            raise HTTPException(404, "Task not found or expired")
        return task

    @app.get(
        "/v1/images/tasks/{task_id}/result",
        response_model=GenerateResponse,
        dependencies=[Depends(authenticate)],
    )
    def task_result(task_id: UUID):
        """成功后复用同步结果结构；轮询期间不保持长连接。"""
        task = task_status(task_id)
        if task.status != "succeeded":
            raise HTTPException(
                409,
                {"code": "result_unavailable", "status": task.status},
                headers={"Retry-After": "2"} if task.status in ("queued", "running") else None,
            )
        result = store.result(str(task_id))
        if result is None:
            raise HTTPException(404, "Task result expired")
        return result

    @app.delete(
        "/v1/images/tasks/{task_id}",
        response_model=TaskResponse,
        dependencies=[Depends(authenticate)],
    )
    def cancel_task(task_id: UUID):
        """只能取消尚未执行的任务，运行中的CUDA推理返回409。"""
        try:
            task = store.cancel(str(task_id))
        except TaskConflictError as exc:
            raise HTTPException(409, str(exc)) from None
        if task is None:
            raise HTTPException(404, "Task not found or expired")
        return task

    return app


app = create_app()
