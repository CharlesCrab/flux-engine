# FLUX Engine

用于部署 **FLUX.2 [klein] 9B 蒸馏模型**的 Python HTTP 服务，基于 FastAPI 和 Diffusers `Flux2KleinPipeline`。支持文生图、单图/多参考图编辑，默认 1024×1024、4 步、guidance scale 1.0，每次返回一张 PNG。

## 运行条件

- Linux + NVIDIA CUDA GPU，Python 3.11+。Docker 镜像使用 PyTorch 2.7.1 / CUDA 12.6，宿主机需要兼容驱动和 NVIDIA Container Toolkit。
- 先在 [官方模型页面](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)接受访问条件，再配置有读取权限的 `HF_TOKEN`。
- 权重采用 FLUX Non-Commercial License。商业自托管需取得相应授权；该模型要求使用内容过滤或人工审核。本工程提供推理接口，未集成内容审核，实际提供服务前需根据授权条件接入审核流程。
- 9B 是生成 Transformer 的规模，还包含 Qwen3 文本编码器，不能只按 9B 参数估算资源。显存需求取决于 offload、精度、分辨率和参考图数量；本工程默认 BF16 + model CPU offload。CPU 内存需容纳完整模型及加载开销，建议从 64 GB RAM、24 GB 或更大显存的机器进行实测；24 GB 是否足够不能保证。显存不足可选 sequential offload，但速度会明显下降。
- 首次启动下载完整权重，需要充足磁盘空间和联网。模型加载失败时进程退出；加载完成后才接收请求。

## Docker 部署

```bash
cp .env.example .env
# 编辑 .env，填写 HF_TOKEN 和 FLUX_API_KEY
docker compose up --build -d
docker compose logs -f flux
```

Compose 需要支持 `gpus` 配置的 Docker Compose 2.30+。模型缓存保存在 `hf-cache` 卷中。默认端口仅绑定宿主机 `127.0.0.1:8000`，可通过带 HTTPS 的反向代理对外提供服务；代理读取超时需覆盖完整推理耗时。

```bash
curl http://localhost:8000/health/ready
```

首次下载可能较久。Docker 健康检查给予 30 分钟启动期，可根据网络情况调整。每个 GPU 仅运行一个 worker；增加 Uvicorn worker 会重复加载模型并导致显存不足。多 GPU 扩容采用独立实例，并为各实例指定 GPU。

## 本地运行（CUDA 主机）

```bash
python3.11 -m venv .venv
source .venv/bin/activate
# 先安装适配宿主机 CUDA/驱动的 PyTorch
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -e '.[model]'
cp .env.example .env
# 编辑 .env
uvicorn flux_engine.app:app --host 0.0.0.0 --port 8000 --workers 1
```

服务自动读取 `.env` 中的配置和 `HF_TOKEN`，shell 环境变量优先。也可使用 `hf auth login` 保存模型访问凭据。Docker Compose 将 `.env` 中的变量传入容器。

## 调用接口

`POST /v1/images/generations` 使用 `Authorization: Bearer <FLUX_API_KEY>` 鉴权。

```bash
curl --fail-with-body http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer replace-with-a-long-random-secret' \
  -d '{"prompt":"一只举着 hello world 牌子的猫","width":1024,"height":1024,"seed":0}' \
  -o response.json
python -c 'import json,base64; d=json.load(open("response.json")); open("result.png","wb").write(base64.b64decode(d["image_base64"]))'
```

响应包含 `model`、`seed`、`width`、`height`、`elapsed_seconds`、`mime_type` 和 `image_base64`。未传 seed 时随机生成并返回实际 seed；同 seed 的结果仍可能因硬件或依赖版本不同而变化。

编辑图片时，在同一请求中添加 `images`，内容为 PNG/JPEG/WebP 文件的原始 base64 数组，无 data URL 前缀：

```python
import base64
from pathlib import Path
import httpx

response = httpx.post(
    "http://localhost:8000/v1/images/generations",
    headers={"Authorization": "Bearer replace-with-a-long-random-secret"},
    json={
        "prompt": "把图中猫的毛色改为灰色，保持背景不变",
        "images": [base64.b64encode(Path("cat.png").read_bytes()).decode()],
        "seed": 42,
    },
    timeout=300,
)
response.raise_for_status()
Path("edited.png").write_bytes(base64.b64decode(response.json()["image_base64"]))
```

示例客户端需要 `pip install httpx`。最多 4 张参考图，每张不超过 8 MiB / 16 MP；解码后保留比例缩放至最长边 1024。整个请求最多 16 MiB，参考图不会从远程 URL 获取。输出宽高需为 16 的倍数，256–2048，默认总像素上限为 1024²。步数可设 1–50，蒸馏模型推荐默认 4 步。

| 接口/状态码 | 含义 |
| --- | --- |
| `GET /health/live` | 进程存活 |
| `GET /health/ready` | 模型就绪和 GPU 是否忙碌；不需要鉴权 |
| `GET /docs` | OpenAPI 调试页面 |
| 401 | API key 缺失或错误 |
| 413 / 422 | 请求过大 / 参数或图片无效 |
| 429 | GPU 忙，按 Retry-After 重试 |
| 503 | 模型未就绪或显存不足 |
| 500 | 推理失败，具体原因查看服务日志 |

推理在线程池执行，健康接口可在推理时响应。GPU 使用非阻塞锁保护，不创建任务队列；客户端断开不会取消正在执行的 CUDA 推理，后续请求仍会收到 429，直到该次推理结束。

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `HF_TOKEN` | 空 | Hugging Face 模型读取凭据 |
| `FLUX_API_KEY` | 空 | 空表示关闭鉴权；对外服务应设置 |
| `FLUX_MODEL_ID` | `black-forest-labs/FLUX.2-klein-9B` | 也可填本地完整 Diffusers 模型目录 |
| `FLUX_MODEL_REVISION` | `main` | 建议部署时指定模型 commit SHA |
| `FLUX_OFFLOAD` | `model` | `none` 全 GPU / `model` 按模块 offload / `sequential` 最省显存但慢 |
| `FLUX_GPU_ID` | `0` | 当前进程可见 GPU 的编号 |
| `FLUX_DTYPE` | `bfloat16` | 也支持 `float16` |
| `FLUX_LOCAL_FILES_ONLY` | `false` | 使用已缓存模型，禁止下载 |
| `FLUX_MAX_PIXELS` | `1048576` | 输出像素预算 |
| `FLUX_MAX_BODY_BYTES` | `16777216` | HTTP 请求体限制 |

离线部署可预先下载完整 Diffusers 权重，挂载至容器后设置 `FLUX_MODEL_ID=/models/flux-klein-9b` 和 `FLUX_LOCAL_FILES_ONLY=true`。裸 `.safetensors` 不是完整 Diffusers 目录，不能直接作为模型目录使用。

## 验证

```bash
pip install -e '.[dev]'
pytest
ruff check .
```

API 测试使用替代推理引擎，覆盖输入验证、鉴权、PNG 返回、多参考图、并发拒绝、异常释放和请求体限制。实际模型加载、GPU 显存、图像质量和推理时延必须在目标 CUDA 主机用真实权重验证。

加载方式依据 [官方模型卡](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)，Pipeline 接口依据 [Diffusers v0.37.1 源码](https://github.com/huggingface/diffusers/blob/v0.37.1/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py)。
