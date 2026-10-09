import gc
import logging
from typing import Any

from PIL import Image

from flux_engine.config import Settings
from flux_engine.schemas import GenerateRequest

logger = logging.getLogger(__name__)


class InferenceOutOfMemory(RuntimeError):
    pass


class FluxEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipeline: Any = None

    def load(self) -> None:
        import torch
        from diffusers import Flux2KleinPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("This service requires an NVIDIA CUDA GPU")
        if self.settings.gpu_id >= torch.cuda.device_count():
            raise RuntimeError("FLUX_GPU_ID is outside the available GPU range")
        torch.cuda.set_device(self.settings.gpu_id)
        if self.settings.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not support BF16; set FLUX_DTYPE=float16")
        logger.info(
            "Loading model %s with offload=%s", self.settings.model_id, self.settings.offload
        )
        pipeline = Flux2KleinPipeline.from_pretrained(
            self.settings.model_id,
            revision=self.settings.model_revision,
            torch_dtype=getattr(torch, self.settings.dtype),
            local_files_only=self.settings.local_files_only,
            token=(self.settings.hf_token.get_secret_value() or None)
            if self.settings.hf_token is not None
            else None,
        )
        if self.settings.offload == "model":
            pipeline.enable_model_cpu_offload(gpu_id=self.settings.gpu_id)
        elif self.settings.offload == "sequential":
            pipeline.enable_sequential_cpu_offload(gpu_id=self.settings.gpu_id)
        else:
            pipeline.to(f"cuda:{self.settings.gpu_id}")
        pipeline.vae.enable_tiling()
        pipeline.set_progress_bar_config(disable=True)
        self.pipeline = pipeline

    def generate(self, request: GenerateRequest, references: list[Image.Image], seed: int):
        import torch

        if self.pipeline is None:
            raise RuntimeError("Model is not loaded")
        try:
            with torch.inference_mode():
                return self.pipeline(
                    prompt=request.prompt,
                    image=references or None,
                    width=request.width,
                    height=request.height,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=1.0,
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                    num_images_per_prompt=1,
                    max_sequence_length=512,
                ).images[0]
        except torch.cuda.OutOfMemoryError:
            # Avoid retaining CUDA tensors through the original exception traceback.
            pass
        gc.collect()
        torch.cuda.empty_cache()
        raise InferenceOutOfMemory("GPU memory exhausted")

    def close(self) -> None:
        self.pipeline = None
        gc.collect()
