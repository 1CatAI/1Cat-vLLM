# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Sampler adapted from Tongyi-MAI/Z-Image (Apache-2.0), commit
# 26f23eda626ffadda020b04ff79488e1d72004cd, src/zimage/pipeline.py.
"""Native single-device Z-Image sampler using local, versioned components.

Diffusers supplies checkpoint-compatible building blocks, not a separate service.
The native runtime owns conditioning, sampling, precision and progress. There
are no Hub downloads, remote model code, or BF16/FlashAttention requirements.
"""

import json
import time
from pathlib import Path

import torch
from PIL import Image

from vllm.image.config import RECIPES, ImageConfig, ImageRequest
from vllm.media.progress import report


class ZImagePipeline:
    def __init__(self, config: ImageConfig, device="cuda:0"):
        from diffusers import (
            AutoencoderKL,
            FlowMatchEulerDiscreteScheduler,
            ZImageTransformer2DModel,
        )
        from transformers import AutoModel, AutoTokenizer

        self.config = config
        self.device = torch.device(device)
        root = Path(config.model).expanduser().resolve(strict=True)
        for component in (
            "transformer",
            "text_encoder",
            "tokenizer",
            "vae",
            "scheduler",
        ):
            if not (root / component).is_dir():
                raise ValueError(f"Missing local Z-Image component: {component}")
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        def loading(completed, component):
            print(
                "ONECAT_MEDIA_PROGRESS "
                + json.dumps(
                    {
                        "stage": "loading_weights",
                        "completed": completed,
                        "total": 3,
                        "unit": "components",
                        "component": component,
                    }
                ),
                flush=True,
            )

        loading(0, "transformer")
        self.transformer = (
            ZImageTransformer2DModel.from_pretrained(
                str(root / "transformer"),
                torch_dtype=torch.float16,
                local_files_only=True,
            )
            .eval()
            .to(self.device)
        )
        from .precision import preserve_projection_range

        preserve_projection_range(
            self.transformer, attention_fp32=config.checkpoint == "z-image"
        )
        loading(1, "text_encoder")
        self.text_encoder = AutoModel.from_pretrained(
            str(root / "text_encoder"),
            dtype=torch.float16,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(root / "tokenizer"),
            local_files_only=True,
            trust_remote_code=False,
        )
        loading(2, "vae")
        # Official recipe decodes in FP32. FP16 VAE can produce blank images.
        self.vae = AutoencoderKL.from_pretrained(
            str(root / "vae"),
            torch_dtype=torch.float32,
            local_files_only=True,
        ).eval()
        self.vae.enable_tiling()
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(root / "scheduler"),
            local_files_only=True,
        )
        loading(3, "ready")
        self.stage_seconds = {}

    def encode(self, prompt, guided):
        prompts = [prompt, ""] if guided else [prompt]
        formatted = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for text in prompts
        ]
        inputs = self.tokenizer(
            formatted,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        self.text_encoder.to(self.device)
        try:
            mask = inputs.attention_mask.bool()
            hidden = self.text_encoder(
                input_ids=inputs.input_ids,
                attention_mask=mask,
                output_hidden_states=True,
            ).hidden_states[-2]
            return [row[row_mask] for row, row_mask in zip(hidden, mask)]
        finally:
            self.text_encoder.to("cpu")

    @torch.inference_mode()
    def __call__(self, request: ImageRequest) -> Image.Image:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        recipe = RECIPES[self.config.checkpoint]
        self.stage_seconds = {}
        report("encoding")
        started = time.perf_counter()
        embeddings = self.encode(request.prompt, recipe["guidance_scale"] > 0)
        if not all(torch.isfinite(value).all() for value in embeddings):
            raise RuntimeError("Z-Image text encoder produced non-finite conditioning")
        self.stage_seconds["encoding"] = time.perf_counter() - started
        generator = torch.Generator(self.device).manual_seed(request.seed)
        latents = torch.randn(
            (
                1,
                self.transformer.config.in_channels,
                request.height // 8,
                request.width // 8,
            ),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        scheduler = self.scheduler
        sc = scheduler.config
        slope = (sc.get("max_shift", 1.15) - sc.get("base_shift", 0.5)) / (
            sc.get("max_image_seq_len", 4096) - sc.get("base_image_seq_len", 256)
        )
        shift = sc.get("base_shift", 0.5) + slope * (
            seq_len - sc.get("base_image_seq_len", 256)
        )
        steps = recipe["steps"]
        # Explicit nonzero sample points: eight updates for Turbo, not nine
        # sigma points advertised as nine denoise steps. Terminal sigma is zero.
        scheduler.set_timesteps(
            sigmas=torch.linspace(1, 1 / steps, steps).tolist(),
            device=self.device,
            mu=shift,
        )
        times = scheduler.timesteps
        normalized = ((1000 - times.detach().cpu()) / 1000).tolist()
        report("denoising", completed=0, total=len(times))
        started = time.perf_counter()
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            for index, timestep in enumerate(times):
                guided = (
                    recipe["guidance_scale"] > 0
                    and normalized[index] <= recipe["cfg_truncation"]
                )
                typed = latents.to(torch.float16)
                model_input = typed.repeat(2, 1, 1, 1) if guided else typed
                t = ((1000 - timestep) / 1000).expand(model_input.shape[0])
                prediction = self.transformer(
                    list(model_input.unsqueeze(2).unbind(0)),
                    t,
                    embeddings if guided else embeddings[:1],
                    return_dict=False,
                )[0]
                if guided:
                    positive, negative = prediction[0].float(), prediction[1].float()
                    noise = (
                        positive + recipe["guidance_scale"] * (positive - negative)
                    )[None]
                else:
                    noise = torch.stack([x.float() for x in prediction])
                latents = scheduler.step(
                    -noise.squeeze(2), timestep, latents, return_dict=False
                )[0]
                report("denoising", completed=index + 1, total=len(times))
        self.stage_seconds["denoising"] = time.perf_counter() - started
        if not torch.isfinite(latents).all():
            raise RuntimeError("Z-Image denoiser produced non-finite latents")
        report("decoding")
        started = time.perf_counter()
        self.vae.to(self.device)
        try:
            shift_factor = self.vae.config.shift_factor or 0.0
            image = self.vae.decode(
                latents.float() / self.vae.config.scaling_factor + shift_factor,
                return_dict=False,
            )[0]
            if not torch.isfinite(image).all():
                raise RuntimeError(
                    "Z-Image produced non-finite pixels; output was not saved"
                )
            pixels = (image[0].float() / 2 + 0.5).clamp(0, 1)
            pixels = (
                (pixels.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
            )
            self.stage_seconds["decoding"] = time.perf_counter() - started
            return Image.fromarray(pixels)
        finally:
            self.vae.to("cpu")
