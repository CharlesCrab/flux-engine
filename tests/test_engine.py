import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flux_engine.config import Settings
from flux_engine.engine import FluxEngine, InferenceOutOfMemory
from flux_engine.schemas import GenerateRequest


@pytest.fixture
def runtime(monkeypatch):
    class FakeOOM(RuntimeError):
        pass

    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        set_device=Mock(),
        is_bf16_supported=lambda: True,
        empty_cache=Mock(),
        OutOfMemoryError=FakeOOM,
    )
    generator = Mock()
    generator.manual_seed.return_value = generator
    torch = SimpleNamespace(
        cuda=cuda,
        bfloat16="bf16",
        float16="fp16",
        Generator=Mock(return_value=generator),
        inference_mode=nullcontext,
    )
    pipeline = Mock()
    pipeline.return_value = SimpleNamespace(images=["output"])
    factory = Mock(return_value=pipeline)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(Flux2KleinPipeline=SimpleNamespace(from_pretrained=factory)),
    )
    return torch, pipeline, factory


@pytest.mark.parametrize("offload", ["none", "model", "sequential"])
def test_loading_and_pipeline_contract(runtime, offload):
    torch, pipeline, factory = runtime
    engine = FluxEngine(Settings(_env_file=None, gpu_id=1, offload=offload, HF_TOKEN="token"))
    engine.load()
    assert factory.call_args.kwargs["token"] == "token"
    assert factory.call_args.kwargs["torch_dtype"] == "bf16"
    if offload == "none":
        pipeline.to.assert_called_once_with("cuda:1")
    elif offload == "model":
        pipeline.enable_model_cpu_offload.assert_called_once_with(gpu_id=1)
    else:
        pipeline.enable_sequential_cpu_offload.assert_called_once_with(gpu_id=1)
    assert engine.generate(GenerateRequest(prompt="cat"), [], 0) == "output"
    kwargs = pipeline.call_args.kwargs
    assert kwargs["num_inference_steps"] == 4
    assert kwargs["guidance_scale"] == 1.0
    assert kwargs["image"] is None
    torch.Generator.assert_called_once_with(device="cpu")
    torch.Generator.return_value.manual_seed.assert_called_once_with(0)


def test_oom_cleanup(runtime):
    torch, pipeline, _ = runtime
    engine = FluxEngine(Settings(_env_file=None))
    engine.load()
    pipeline.side_effect = torch.cuda.OutOfMemoryError("oom")
    with pytest.raises(InferenceOutOfMemory):
        engine.generate(GenerateRequest(prompt="cat"), [], 1)
    torch.cuda.empty_cache.assert_called_once()


def test_no_gpu_fails_at_startup(runtime):
    torch, _, factory = runtime
    torch.cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        FluxEngine(Settings(_env_file=None)).load()
    factory.assert_not_called()
