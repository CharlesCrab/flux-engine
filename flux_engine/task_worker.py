"""单线程执行持久化任务，与同步请求共用唯一GPU锁。"""

import logging
import threading

from flux_engine.engine import InferenceOutOfMemory
from flux_engine.generation import execute_generation
from flux_engine.schemas import TaskError
from flux_engine.task_store import TaskStore

logger = logging.getLogger(__name__)


class TaskWorker:
    """连接断开不取消任务；持久化失败时停止准入，禁止盲目重做推理。"""

    def __init__(self, store: TaskStore, engine, gpu_lock: threading.Lock):
        self.store, self.engine, self.gpu_lock = store, engine, gpu_lock
        self.wakeup = threading.Event()
        self.stopping = threading.Event()
        self.failed = False
        self.thread = None

    def start(self) -> None:
        """服务加载模型并恢复状态后启动唯一工作线程。"""
        self.thread = threading.Thread(target=self._run, name="flux-task-worker", daemon=True)
        self.thread.start()

    def close(self) -> None:
        """等待当前推理完成，保留尚未认领任务，之后才允许卸载模型。"""
        self.stopping.set()
        self.wakeup.set()
        if self.thread is not None:
            self.thread.join()

    def _run(self) -> None:
        """轮询和唤醒结合，避免进程在通知前退出导致任务丢失。"""
        try:
            while not self.stopping.is_set():
                self.wakeup.clear()
                if not self.store.has_pending():
                    self.wakeup.wait(0.5)
                    continue
                if not self.gpu_lock.acquire(timeout=0.1):
                    continue
                try:
                    if self.stopping.is_set():
                        return
                    task = self.store.claim()
                    if task is not None:
                        self._execute(*task)
                finally:
                    self.gpu_lock.release()
                if task is None:
                    self.wakeup.wait(0.5)
        except Exception:
            self.failed = True
            logger.error("Task worker stopped after a persistence error; restart required")

    def _execute(self, ident, request, model) -> None:
        """模型错误公开安全分类；写入错误必须传播，使worker停止而非重新执行。"""
        result, error = None, None
        try:
            result = execute_generation(self.engine, request, request.seed, model)
        except InferenceOutOfMemory:
            error = TaskError(code="gpu_out_of_memory", message="GPU memory exhausted")
        except ValueError:
            error = TaskError(code="invalid_input", message="Stored image input is invalid")
        except Exception:
            error = TaskError(code="generation_failed", message="Image generation failed")
        self.store.finish(ident, result, error)
