# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import threading
import time
from pathlib import Path

from vllm.media.progress import DeviceProgress, ProgressCallback, report, reporting

from .config import RECIPE_VERSION, RECIPES, ImageConfig, ImageRequest


class ImageEngine:
    def __init__(self, config: ImageConfig):
        from vllm.video.gpu import GPUGroupLease, acquire_gpu_group, worker_device_mask

        self.config = config
        self._closed = False
        self._lock = threading.Lock()
        self._gpu_lease: GPUGroupLease | None = acquire_gpu_group(1)
        self.gpu_ids = self._gpu_lease.gpu_ids
        os.environ["CUDA_VISIBLE_DEVICES"] = worker_device_mask(self.gpu_ids)
        try:
            from vllm.model_executor.models.z_image.pipeline import ZImagePipeline

            self.pipeline = ZImagePipeline(config)
        except BaseException:
            self.close()
            raise

    def generate(
        self,
        request: ImageRequest,
        output_dir: Path,
        *,
        on_progress: ProgressCallback | None = None,
    ):
        with self._lock, DeviceProgress(on_progress) as observer, reporting(observer):
            if self._closed:
                raise RuntimeError("Native image engine is closed")
            started = time.perf_counter()
            image = self.pipeline(request)
            report("saving")
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "image.png"
            temporary = output_dir / "image.png.tmp"
            image.save(temporary, format="PNG")
            from PIL import Image

            with Image.open(temporary) as saved:
                saved.verify()
            temporary.replace(path)
            result = {
                "width": image.width,
                "height": image.height,
                "seed": request.seed,
                "model": self.config.checkpoint,
                "recipe_version": RECIPE_VERSION,
                "recipe": RECIPES[self.config.checkpoint],
                "precision": (
                    "fp16-weights-fp32-attention-residual-and-vae"
                    if self.config.checkpoint == "z-image"
                    else "fp16-fp32-projection-output-and-vae"
                ),
                "stage_seconds": self.pipeline.stage_seconds,
                "end_to_end_seconds": time.perf_counter() - started,
            }
            (output_dir / "run.json").write_text(json.dumps(result, indent=2))
            return result

    def close(self):
        self._closed = True
        if hasattr(self, "pipeline"):
            del self.pipeline
        if self._gpu_lease is not None:
            self._gpu_lease.close()
            self._gpu_lease = None
