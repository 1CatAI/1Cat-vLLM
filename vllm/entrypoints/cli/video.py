# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native H3 offline and serial HTTP serving entrypoints."""

import argparse
import json
from pathlib import Path

from vllm.entrypoints.cli.types import CLISubcommand


class VideoSubcommand(CLISubcommand):
    name = "video"

    def subparser_init(self, subparsers):
        from vllm.model_executor.models.minimax_h3.config import (
            BASE_MODEL,
            BASE_REVISION,
            DEFAULT_PROMPT,
        )

        parser = subparsers.add_parser("video", help="Native audio/video generation")
        modes = parser.add_subparsers(dest="video_mode", required=True)
        for name in ("generate", "serve"):
            mode = modes.add_parser(name)
            mode.add_argument("--model", default=BASE_MODEL)
            mode.add_argument("--revision", default=BASE_REVISION)
            mode.add_argument(
                "--partition", choices=("fl2va", "ref2va"), default="fl2va"
            )
            mode.add_argument("--transformer-path")
            mode.add_argument(
                "--lora-path",
                help="LightX2V Turbo, FlashGen or FastH3 Dense safetensors",
            )
            mode.add_argument("--tensor-parallel-size", "-tp", type=int, default=4)
            mode.add_argument(
                "--attention-backend",
                choices=(
                    "FLASH_ATTN_V100",
                    "FLASHINFER_SM70",
                    "TORCH_SDPA",
                    "FASTVIDEO_VSA",
                ),
                default="FLASH_ATTN_V100",
            )
            mode.add_argument("--fastvideo-vsa-topk", type=int, default=64)
            mode.add_argument("--fp16-weight-cache-gib", type=float, default=0)
            mode.add_argument(
                "--weight-offload",
                choices=("component", "layer"),
                default="component",
                help="Stage complete components or individual DiT/encoder layers",
            )
            mode.add_argument(
                "--share-host-vae-weights",
                action="store_true",
                help="Share immutable pageable VAE masters across TP workers",
            )
            mode.add_argument(
                "--disable-host-weight-pinning",
                dest="host_weight_pin_memory",
                action="store_false",
                help="Keep weight masters pageable when pinned copies exceed host RAM",
            )
            mode.add_argument("--fp16-cache-layer", action="append", default=[])
            mode.add_argument(
                "--int8-weight-layout", choices=["row", "column"], default="column"
            )
            mode.add_argument(
                "--fp16-weight-layout", choices=["row", "column"], default="row"
            )
            mode.add_argument(
                "--residual-sequence-parallel",
                action="store_true",
                help="Experimental FP32 residual sharding for TP2/TP4; TP1 is a no-op",
            )
            mode.add_argument("--output-dir", type=Path, default=Path("h3-output"))
            mode.add_argument(
                "--video-encoder",
                choices=("libx264", "h264_nvenc"),
                default="libx264",
                help="MP4 encoder; NVENC requires a capable IMAGEIO_FFMPEG_EXE",
            )
            if name == "generate":
                mode.add_argument("--prompt", default=DEFAULT_PROMPT)
                mode.add_argument("--width", type=int, default=1344)
                mode.add_argument("--height", type=int, default=768)
                mode.add_argument("--num-frames", type=int, default=243)
                mode.add_argument("--duration", type=float)
                mode.add_argument("--seed", type=int, default=42)
                mode.add_argument(
                    "--num-inference-steps",
                    type=int,
                    help="Default 50; LightX2V uses 5/9 points, FlashGen/FastH3 use 4",
                )
                mode.add_argument(
                    "--lora-scale",
                    type=float,
                    default=1.0,
                    help="Adapter multiplier; 0 bypasses the loaded adapter",
                )
                mode.add_argument("--task", choices=("t2va", "fl2va", "ref2va"))
                mode.add_argument("--flow-shift", type=float)
                mode.add_argument("--audio-flow-shift", type=float)
                mode.add_argument(
                    "--reference-video-start-times", nargs="+", type=float
                )
                mode.add_argument("--image", action="append", default=[])
                mode.add_argument("--video", action="append", default=[])
                mode.add_argument("--audio", action="append", default=[])
                mode.add_argument("--keyframe-indices", nargs="+", type=int)
            else:
                mode.add_argument("--host", default="127.0.0.1")
                mode.add_argument("--port", type=int, default=8000)
        return parser

    @staticmethod
    def cmd(args):
        from vllm.model_executor.models.minimax_h3.config import (
            H3Config,
            H3Request,
            sampling_for_deployment,
        )
        from vllm.video.engine import H3Engine

        config = H3Config(
            model=args.model,
            revision=args.revision,
            partition=args.partition,
            transformer_path=args.transformer_path,
            tensor_parallel_size=args.tensor_parallel_size,
            attention_backend=args.attention_backend,
            vsa_topk=args.fastvideo_vsa_topk,
            fp16_weight_cache_gib=args.fp16_weight_cache_gib,
            fp16_cache_layers=tuple(args.fp16_cache_layer),
            lora_path=args.lora_path,
            int8_weight_layout=args.int8_weight_layout,
            fp16_weight_layout=args.fp16_weight_layout,
            residual_sequence_parallel=args.residual_sequence_parallel,
            video_encoder=args.video_encoder,
            host_weight_pin_memory=args.host_weight_pin_memory,
            share_host_vae_weights=args.share_host_vae_weights,
            weight_offload=args.weight_offload,
        )
        if args.video_mode == "serve":
            from vllm.video.server import serve

            serve(config, host=args.host, port=args.port, output_dir=args.output_dir)
            return
        extra = {}
        for key in ("task", "flow_shift", "audio_flow_shift"):
            if getattr(args, key) is not None:
                extra[key] = getattr(args, key)
        if args.duration is not None:
            extra["duration_seconds"] = args.duration
        if args.keyframe_indices is not None:
            extra["frame_indices"] = args.keyframe_indices
        if args.reference_video_start_times is not None:
            extra["start_time_seconds"] = args.reference_video_start_times
        request = H3Request(
            prompt=args.prompt,
            sampling=sampling_for_deployment(
                config,
                width=args.width,
                height=args.height,
                num_frames=args.num_frames,
                seed=args.seed,
                num_inference_steps=args.num_inference_steps,
                lora_scale=args.lora_scale,
                extra_args=extra,
            ),
            media={
                key: getattr(args, key)
                for key in ("image", "video", "audio")
                if getattr(args, key)
            },
        )
        with H3Engine(config) as engine:
            result = engine.generate(request, args.output_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_init():
    return [VideoSubcommand()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    command = VideoSubcommand()
    command.subparser_init(subparsers)
    command.cmd(parser.parse_args())
