from typing import Literal

from pydantic import Field, SecretStr
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
