"""同步与排队任务共用的输入校验和图片生成实现。"""

import base64
import time
from contextlib import contextmanager
from io import BytesIO

from flux_engine.images import decode_reference
from flux_engine.schemas import GenerateRequest, GenerateResponse


@contextmanager
def decoded_references(request: GenerateRequest):
    """确保输入错误、推理错误及成功路径均释放参考图内存。"""
    references = []
    try:
        for encoded in request.images:
            references.append(decode_reference(encoded))
        yield references
    finally:
        for image in references:
            image.close()


def validate_generation(request: GenerateRequest, max_pixels: int) -> None:
    """在任务受理前校验尺寸和图片，拒绝无效请求而不入队。"""
    if request.width * request.height > max_pixels:
        raise ValueError(f"output exceeds {max_pixels} pixels")
    with decoded_references(request):
        pass


def execute_generation(engine, request: GenerateRequest, seed: int, model: str) -> GenerateResponse:
    """调用者持有共享GPU锁；返回完整结果后才允许持久化成功终态。"""
    with decoded_references(request) as references:
        started = time.perf_counter()
        with engine.generate(request, references, seed) as image, BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return GenerateResponse(
                model=model,
                seed=seed,
                width=image.width,
                height=image.height,
                elapsed_seconds=round(time.perf_counter() - started, 3),
                image_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
            )
