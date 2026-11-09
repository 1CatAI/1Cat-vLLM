# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RECIPES = {
    "z-image-turbo": {"steps": 8, "guidance_scale": 0.0, "cfg_truncation": 1.0},
    "z-image": {"steps": 50, "guidance_scale": 4.0, "cfg_truncation": 1.0},
}
RECIPE_VERSION = "z-image-fp16-v1"


@dataclass(frozen=True)
class ImageConfig:
    model: str
    checkpoint: Literal["z-image-turbo", "z-image"] = "z-image-turbo"

    def __post_init__(self):
        if self.checkpoint not in RECIPES:
            raise ValueError("Unsupported native image checkpoint")


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=20000)
    model: str | None = None
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    size: str | None = None
    seed: int = Field(default=42, ge=0, lt=2**63)
    n: Literal[1] = 1
    response_format: Literal["url", "b64_json"] = "url"

    @model_validator(mode="after")
    def validate_size(self):
        if not self.prompt.strip():
            raise ValueError("Prompt must contain text")
        if self.size:
            try:
                width, height = (int(x) for x in self.size.split("x"))
            except ValueError as exc:
                raise ValueError("Image size must be WIDTHxHEIGHT") from exc
            for name, value in (("width", width), ("height", height)):
                if name in self.model_fields_set and getattr(self, name) != value:
                    raise ValueError("Conflicting image dimensions")
                setattr(self, name, value)
        if any(x < 256 or x > 2048 or x % 16 for x in (self.width, self.height)):
            raise ValueError(
                "Image dimensions must be multiples of 16 from 256 to 2048"
            )
        if self.width * self.height > 2048 * 1024:
            raise ValueError("Requested image exceeds the native pixel budget")
        return self
