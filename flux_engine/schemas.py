from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=8000)
    width: int = Field(default=1024, ge=256, le=2048, multiple_of=16)
    height: int = Field(default=1024, ge=256, le=2048, multiple_of=16)
    num_inference_steps: int = Field(default=4, ge=1, le=50)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    # Raw base64 PNG/JPEG/WebP, without a data URL prefix. Never fetch remote URLs.
    images: list[Annotated[str, Field(min_length=1, max_length=12_000_000)]] = Field(
        default_factory=list, max_length=4
    )

    @field_validator("prompt")
    @classmethod
    def non_blank_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt must not be blank")
        return value


class GenerateResponse(BaseModel):
    model: str
    seed: int
    width: int
    height: int
    elapsed_seconds: float
    mime_type: Literal["image/png"] = "image/png"
    image_base64: str
