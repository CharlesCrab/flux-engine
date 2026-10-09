import base64
import logging
import secrets
import threading
import time
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from flux_engine.config import Settings
from flux_engine.engine import FluxEngine, InferenceOutOfMemory
from flux_engine.images import decode_reference
from flux_engine.schemas import GenerateRequest, GenerateResponse

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
    settings = settings or Settings()
    engine = engine if engine is not None else FluxEngine(settings)
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        await run_in_threadpool(engine.load)
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            await run_in_threadpool(engine.close)

    app = FastAPI(title="FLUX.2 klein 9B", version="0.1.0", lifespan=lifespan)
    app.state.ready = False
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

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready():
        if not app.state.ready:
            raise HTTPException(503, "model not ready")
        return {"status": "ready", "model": settings.model_id, "busy": lock.locked()}

    @app.post(
        "/v1/images/generations",
        response_model=GenerateResponse,
        dependencies=[Depends(authenticate)],
    )
    def generate(request: GenerateRequest):
        if not app.state.ready:
            raise HTTPException(503, "model not ready")
        if request.width * request.height > settings.max_pixels:
            raise HTTPException(422, f"output exceeds {settings.max_pixels} pixels")
        if not lock.acquire(blocking=False):
            raise HTTPException(429, "GPU busy; retry later", headers={"Retry-After": "2"})
        try:
            try:
                references = [decode_reference(encoded) for encoded in request.images]
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            seed = request.seed if request.seed is not None else secrets.randbits(63)
            start = time.perf_counter()
            try:
                image = engine.generate(request, references, seed)
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                return GenerateResponse(
                    model=settings.model_id,
                    seed=seed,
                    width=image.width,
                    height=image.height,
                    elapsed_seconds=round(time.perf_counter() - start, 3),
                    image_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
                )
            except InferenceOutOfMemory as exc:
                logger.warning("GPU out of memory")
                raise HTTPException(503, "GPU memory exhausted; reduce image size") from exc
            except Exception as exc:
                logger.exception("Image inference failed")
                raise HTTPException(500, "image generation failed") from exc
        finally:
            lock.release()

    return app


app = create_app()
