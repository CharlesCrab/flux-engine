from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FLUX_", env_file=".env", extra="ignore")

    model_id: str = "black-forest-labs/FLUX.2-klein-9B"
    model_revision: str = "main"
    api_key: SecretStr = SecretStr("")
    hf_token: SecretStr | None = Field(default=None, validation_alias="HF_TOKEN")
    offload: Literal["none", "model", "sequential"] = "model"
    gpu_id: int = Field(default=0, ge=0)
    dtype: Literal["bfloat16", "float16"] = "bfloat16"
    local_files_only: bool = False
    max_pixels: int = Field(default=1024 * 1024, ge=256 * 256)
    max_body_bytes: int = Field(default=16 * 1024 * 1024, ge=1024)
    task_data_dir: Path = Path("var/tasks")
    task_queue_capacity: int = Field(default=32, ge=1, le=1024)
    task_max_records: int = Field(default=1024, ge=1)
    task_retention_seconds: int = Field(default=86400, ge=60)

    @model_validator(mode="after")
    def validate_task_capacity(self):
        """结果记录上限必须能够容纳全部在途任务。"""
        if self.task_max_records < self.task_queue_capacity:
            raise ValueError("task_max_records must be >= task_queue_capacity")
        return self
