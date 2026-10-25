"""SQLite任务事实源：原子准入、幂等、FIFO认领、终态与重启恢复。"""

import fcntl
import hashlib
import json
import os
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager

from flux_engine.config import Settings
from flux_engine.schemas import GenerateRequest, GenerateResponse, TaskError, TaskResponse


class TaskCapacityError(Exception):
    """持久化队列或结果记录达到上限，本次请求未被受理。"""

    def __init__(self, code: str):
        self.code = code


class TaskConflictError(Exception):
    """幂等键冲突或任务已经执行，不能安全地改变其状态。"""


class TaskStore:
    """一个数据目录只允许一个加载模型的服务进程拥有任务执行权。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = settings.task_data_dir / "tasks.sqlite3"
        self._owner = None

    def open(self) -> None:
        """先获取跨进程所有权，再初始化私有数据库；不接触模型。"""
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._owner = (self.path.parent / "worker.lock").open("ab")
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._owner.close()
            self._owner = None
            raise RuntimeError(
                "Task directory already has an active worker; use one process"
            ) from None
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
            os.close(fd)
            with self._connection() as db:
                version = db.execute("PRAGMA user_version").fetchone()[0]
                if version not in (0, 1):
                    raise RuntimeError("Unsupported task database version")
                db.execute("PRAGMA journal_mode=WAL")
                db.executescript("""
                    CREATE TABLE IF NOT EXISTS tasks (
                        id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        model TEXT NOT NULL,
                        model_signature TEXT NOT NULL,
                        seed INTEGER NOT NULL,
                        request_hash TEXT NOT NULL,
                        idempotency_hash TEXT UNIQUE,
                        request_json TEXT,
                        result_json TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        expires_at REAL,
                        error_code TEXT,
                        error_message TEXT
                    );
                    CREATE INDEX IF NOT EXISTS tasks_status ON tasks(status);
                    CREATE INDEX IF NOT EXISTS tasks_expiry ON tasks(expires_at);
                    PRAGMA user_version=1;
                """)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """在GPU工作线程停止后释放执行权。"""
        if self._owner is not None:
            self._owner.close()
            self._owner = None

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def _transaction(self):
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute("DELETE FROM tasks WHERE expires_at <= ?", (time.time(),))
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    @property
    def model_signature(self) -> str:
        """恢复排队任务前核对模型配置，防止升级后静默换模型执行。"""
        return json.dumps(
            [
                self.settings.model_id,
                self.settings.model_revision,
                self.settings.dtype,
                self.settings.max_pixels,
            ]
        )

    def recover(self) -> None:
        """保留未认领任务；中断的生成不可自动重做，避免重复GPU副作用。"""
        now = time.time()
        with self._transaction() as db:
            db.execute(
                """UPDATE tasks SET status='failed', request_json=NULL, updated_at=?,
                   expires_at=?, error_code='worker_interrupted',
                   error_message='Previous worker stopped during generation; not retried'
                   WHERE status='running'""",
                (now, now + self.settings.task_retention_seconds),
            )
            db.execute(
                """UPDATE tasks SET status='failed', request_json=NULL, updated_at=?,
                   expires_at=?, error_code='model_changed',
                   error_message='Model configuration changed before execution'
                   WHERE status='queued' AND model_signature != ?""",
                (now, now + self.settings.task_retention_seconds, self.model_signature),
            )

    @staticmethod
    def _view(row) -> TaskResponse:
        error = (
            TaskError(code=row["error_code"], message=row["error_message"])
            if row["error_code"]
            else None
        )
        return TaskResponse(
            id=row["id"],
            status=row["status"],
            model=row["model"],
            seed=row["seed"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
            error=error,
            result_url=f"/v1/images/tasks/{row['id']}/result"
            if row["status"] == "succeeded"
            else None,
        )

    def submit(self, request: GenerateRequest, idempotency_key: str | None) -> TaskResponse:
        """同一幂等键同一输入复用任务；准入上限检查与插入在同一事务中。"""
        payload = request.model_dump()
        request_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        idem_hash = (
            hashlib.sha256(idempotency_key.encode()).hexdigest() if idempotency_key else None
        )
        with self._transaction() as db:
            if idem_hash:
                existing = db.execute(
                    "SELECT * FROM tasks WHERE idempotency_hash=?", (idem_hash,)
                ).fetchone()
                if existing:
                    if existing["request_hash"] != request_hash:
                        raise TaskConflictError("Idempotency-Key already used for different input")
                    return self._view(existing)
            active = db.execute(
                "SELECT count(*) FROM tasks WHERE status IN ('queued','running')"
            ).fetchone()[0]
            if active >= self.settings.task_queue_capacity:
                raise TaskCapacityError("queue_full")
            if (
                db.execute("SELECT count(*) FROM tasks").fetchone()[0]
                >= self.settings.task_max_records
            ):
                raise TaskCapacityError("task_storage_full")
            ident, now = str(uuid.uuid4()), time.time()
            seed = request.seed if request.seed is not None else secrets.randbits(63)
            payload["seed"] = seed
            db.execute(
                """INSERT INTO tasks (id,status,model,model_signature,seed,request_hash,
                   idempotency_hash,request_json,created_at,updated_at)
                   VALUES (?,'queued',?,?,?,?,?,?,?,?)""",
                (
                    ident,
                    self.settings.model_id,
                    self.model_signature,
                    seed,
                    request_hash,
                    idem_hash,
                    json.dumps(payload),
                    now,
                    now,
                ),
            )
            return self._view(db.execute("SELECT * FROM tasks WHERE id=?", (ident,)).fetchone())

    def get(self, ident: str) -> TaskResponse | None:
        """只返回公开状态；清理已过期的输入、结果和幂等映射。"""
        with self._transaction() as db:
            row = db.execute("SELECT * FROM tasks WHERE id=?", (ident,)).fetchone()
            return self._view(row) if row else None

    def result(self, ident: str) -> GenerateResponse | None:
        """仅读取已成功任务的完整结果。"""
        with self._transaction() as db:
            row = db.execute(
                "SELECT result_json FROM tasks WHERE id=? AND status='succeeded'", (ident,)
            ).fetchone()
            return GenerateResponse.model_validate_json(row[0]) if row else None

    def has_pending(self) -> bool:
        """让已受理的队列任务优先，避免被同步请求持续抢占。"""
        with self._connection() as db:
            return (
                db.execute(
                    "SELECT 1 FROM tasks WHERE status IN ('queued','running') LIMIT 1"
                ).fetchone()
                is not None
            )

    def claim(self):
        """调用者已经持有GPU锁；持久化running之后才进入推理。"""
        with self._transaction() as db:
            row = db.execute(
                "SELECT * FROM tasks WHERE status='queued' ORDER BY rowid LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE tasks SET status='running',updated_at=? WHERE id=?",
                (time.time(), row["id"]),
            )
            return row["id"], GenerateRequest.model_validate_json(row["request_json"]), row["model"]

    def finish(self, ident: str, result: GenerateResponse | None, error: TaskError | None) -> None:
        """结果与终态原子提交；任何持久化错误均不可触发重新生成。"""
        now = time.time()
        with self._transaction() as db:
            changed = db.execute(
                """UPDATE tasks SET status=?,request_json=NULL,result_json=?,updated_at=?,
                   expires_at=?,error_code=?,error_message=? WHERE id=? AND status='running'""",
                (
                    "succeeded" if result else "failed",
                    result.model_dump_json() if result else None,
                    now,
                    now + self.settings.task_retention_seconds,
                    error.code if error else None,
                    error.message if error else None,
                    ident,
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError("Task no longer owned by the worker")

    def cancel(self, ident: str) -> TaskResponse | None:
        """只取消尚未认领的任务；CUDA运行中不假装取消成功。"""
        now = time.time()
        with self._transaction() as db:
            row = db.execute("SELECT * FROM tasks WHERE id=?", (ident,)).fetchone()
            if row is None:
                return None
            if row["status"] == "running":
                raise TaskConflictError("Running GPU inference cannot be cancelled")
            if row["status"] == "queued":
                db.execute(
                    """UPDATE tasks SET status='cancelled',request_json=NULL,updated_at=?,
                       expires_at=? WHERE id=?""",
                    (now, now + self.settings.task_retention_seconds, ident),
                )
            return self._view(db.execute("SELECT * FROM tasks WHERE id=?", (ident,)).fetchone())
