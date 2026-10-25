# FLUX 异步生成任务

## 目标与范围

高峰期由服务端持久化排队，HTTP 提交在输入校验和入库完成后返回；推理时间与排队时间不占用同一条 HTTP 连接。保留原同步接口与结果格式。

基线为 `main / 26e7260aecd5ebd0d5a54a387dfc4d8999bca822`，开发分支 `feature/async-image-tasks`。本变更只涉及 `flux-engine`，没有修改 Kuma A/B 客户端，也没有发布线上。

## 执行与持久化边界

- `app.py` 负责鉴权、输入校验、HTTP 状态及结果投影。
- `generation.py` 唯一拥有同步和异步共用的图片推理及 PNG 返回格式。
- `task_store.py` 的 SQLite 是任务唯一事实源；事务覆盖容量检查、幂等键占用、状态转换和结果保存。
- `task_worker.py` 是单线程执行器；与同步请求共享一个 GPU 锁。异步队列内部 FIFO，已有队列任务优先，同步请求不加入队列。

任务状态：`queued → running → succeeded / failed`；`queued → cancelled`。

POST 在提交事务完成后才返回 202。claim 在 GPU 已可用后原子写入 running，随后才调用模型；图片结果及 succeeded 原子写入。默认 seed 在第一次入队时产生并持久化，重复提交或恢复排队不会换 seed。

生成失败不会自动重试；同键重试返回同一失败任务。创建新的业务操作可以使用新键显式重试。数据库写入失败时执行器停止，健康接口返回 503，停止新推理准入，保留已有 task ID 供恢复；不能因保存结果失败而再次调用模型。

SQLite 目录使用进程级文件锁，一个目录只能有一个服务进程加载模型。多个 Uvicorn worker 会在加载第二份权重前失败。该方案用于当前单机、单 GPU、单进程部署；不是分布式队列，也不支持多个实例共享同一 SQLite 目录。使用本地可靠磁盘，不使用 NFS。

正常关闭先停止准入，等待正在执行的异步推理完成并保存结果，再卸载模型；未执行任务留待下次启动。强制退出后，旧 running 任务标记 worker_interrupted；其是否已产生但尚未保存输出无法确定，禁止自动重复执行。旧 queued 任务仅在模型 ID、revision、精度和像素预算一致时继续执行。

## 排队与存储上限

默认在途容量 32（包含运行中的任务），保留记录上限 1024，终态保留 24 小时。达到任一上限返回 429，不接受新的任务；已有幂等任务的重复提交仍能取回原 ID。可按流量和磁盘预算调整配置。

每个请求仍受 16 MiB 请求体限制，图片仍受原尺寸和参考图门禁。终态立即删除数据库中的原始请求正文；终态结果和幂等映射到期后在访问、提交或认领任务时清理。SQLite 会复用释放的页面，数据库文件不保证立即缩小；磁盘容量需要覆盖队列、结果以及 WAL 开销。

任务 ID 为随机 UUID。全部任务接口复用同一 Bearer key，持有同一 key 的客户端属于同一信任边界；未实现按用户隔离、任务列表、Webhook、SSE 或运行中 CUDA 取消。DELETE 对 queued 生效，running 返回 409，终态重复 DELETE 返回现有状态。

## Python 客户端示例

业务方应先持久化 `operation_id`，提交成功后保存 `task.id`。如果 POST 超时或连接断开，重用原 operation_id 和完全相同的请求体重试；不要换键猜测提交失败。恢复轮询直接使用已保存的 task ID。超过任务保留期后不能依靠旧幂等键去重。

```python
import asyncio
import httpx


async def generate_image(base_url: str, api_key: str, operation_id: str, payload: dict):
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=30) as client:
        # 网络错误时由业务方保存 operation_id，再使用相同参数重新提交。
        response = await client.post(
            "/v1/images/tasks",
            headers={"Idempotency-Key": operation_id},
            json=payload,
        )
        response.raise_for_status()
        task = response.json()
        task_path = response.headers["Location"]
        # 在这里将 task["id"] 持久化到业务记录。
        while task["status"] in {"queued", "running"}:
            await asyncio.sleep(2)
            response = await client.get(task_path, timeout=10)
            response.raise_for_status()
            task = response.json()
        if task["status"] != "succeeded":
            raise RuntimeError(task["error"] or {"code": task["status"]})
        response = await client.get(task["result_url"])
        response.raise_for_status()
        return response.json()  # 与同步响应相同，包含 image_base64。
```

这里每次 HTTP 请求都有短超时，业务层可以单独设置总等待时间。停止客户端轮询不会取消服务端任务；以后仍可继续查同一 ID，或在 queued 阶段显式 DELETE。针对 429/503/临时网络错误应退避重试，不应立即创建新任务。任务失败可从状态的 error.code 获取 `gpu_out_of_memory / generation_failed / invalid_input / worker_interrupted / model_changed`。

## 部署与调用方切换

Docker Compose 已增加 `flux-tasks` 持久卷并将目录设为 `/data/tasks`。原模型缓存卷保留。原生/systemd 可在现有服务中配置：

```ini
[Service]
StateDirectory=flux-engine
Environment=FLUX_TASK_DATA_DIR=/var/lib/flux-engine/tasks
```

运行用户必须能够写入该私有目录。不要把数据库放在临时工作区或只读镜像层，也不要将它打包进镜像。本地默认 `var/tasks` 已加入 Git 和 Docker 忽略规则。

部署前先排空现有 GPU 推理，再由受管服务进行更新。服务正常关闭会等待当前任务落盘，部署工具的停止宽限期必须覆盖实际最长推理时间；不得在 GPU 任务执行中强制重启。强制中断的任务不会自动重做，排队任务和已保存结果可在保留数据目录后恢复。

**仅部署此服务不会改变现有同步调用方的行为。** Kuma A/B 当前使用 `/v1/images/generations`，需另行适配为提交、记录 task ID、轮询、下载结果，才能消除其推理期间的长 HTTP 请求。

回滚到旧版本前停止接收新任务并排空队列，保留数据目录及备份。旧代码没有异步接口，不能继续处理已受理队列；不能把回滚描述为任务已经处理完毕。本版本数据库 user_version 为 1，遇到不支持的版本将拒绝启动，不静默改写未知 schema。

## 验证

测试使用替代模型和真实临时 SQLite，覆盖同步兼容、提前返回、认证、无效输入、并发幂等、并发容量、FIFO、取消、运行中拒绝取消、同步/异步互斥、结果复查、错误脱敏、正常排空、重启恢复、结果保留、异常运行中任务、模型配置改变、存储失败停止准入、容量与到期清理及重复进程拒绝。

```bash
python -m pytest -q
ruff check .
ruff format --check .
python -m compileall -q flux_engine tests
git diff --check
```

未执行真实 GPU 生图与线上升级验证；本地通过不等同于线上已经支持新接口。

本轮本地验证结果：pytest **40 passed**；Ruff lint、format 检查、compileall 和
`git diff --check` 均通过。测试使用替代 GPU 引擎，不产生真实模型调用。
