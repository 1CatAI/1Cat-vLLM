# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run directly with torchrun; full small H3 forward and TP cache agreement."""

import os

import torch
from torch import nn


def main():
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.models.minimax_h3.pipeline import MiniMaxH3Pipeline
    from vllm.model_executor.models.minimax_h3.request_cache import (
        CACHE_DIT_DEFAULTS,
        CachePlan,
        request_cache,
    )
    from vllm.model_executor.models.minimax_h3.transformer import MiniMaxH3DiTModel
    from vllm.video.metrics import DenoiseWorkCounter

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.set_num_threads(2)
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world))
    ):
        init_distributed_environment(
            world, rank, "env://", int(os.environ["LOCAL_RANK"]), "nccl"
        )
        initialize_model_parallel(world)
        try:
            with torch.inference_mode():
                model = (
                    MiniMaxH3DiTModel(
                        dict(
                            num_layers=4,
                            hidden_size=512,
                            num_attention_heads=4,
                            ffn_hidden_size=1024,
                            text_dim=32,
                            adaln_curve_grid=2,
                            adaln_out_features=18 * 512,
                            final_adaln_out_features=2 * 512,
                        ),
                        residual_sequence_parallel=True,
                    )
                    .cuda()
                    .eval()
                )
                torch.manual_seed(190)
                for name, value in model.named_parameters():
                    value.fill_(1) if "norm" in name else value.normal_(0, 0.01)
                for name, value in model.named_buffers():
                    value.normal_(0, 0.01)
                pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
                nn.Module.__init__(pipeline)
                pipeline.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
                inputs = pipeline._build_denoise_inputs(
                    task="t2va",
                    text_embeddings=torch.randn(3, 32, device="cuda"),
                    text_tags=torch.ones(3, dtype=torch.long, device="cuda"),
                    seed=42,
                    latent_t=3,
                    latent_h=4,
                    latent_w=8,
                    audio_t=3,
                    num_frames=22,
                    num_steps=5,
                    video_shift=12,
                    audio_shift=3,
                    base_schedule=None,
                    visual_condition=None,
                    visual_condition_shape=None,
                    audio_condition=None,
                    ref_audio_t=None,
                )
                branch = inputs["branch"]
                kwargs = branch.forward_kwargs(
                    video_rows=inputs["video_rows"],
                    audio_rows=inputs["audio_rows"],
                    t_video=0.9,
                    t_audio=0.7,
                    imgvid_cond_timestep=0.1,
                    audio_ref_cond_timestep=0.1,
                )
                expected = model(**kwargs)
                plans = [
                    CachePlan("tea_cache", {"rel_l1_thresh": 0}, 4),
                    CachePlan("tea_cache", {"rel_l1_thresh": 0.5}, 4),
                    CachePlan(
                        "cache_dit",
                        {
                            **CACHE_DIT_DEFAULTS,
                            "max_warmup_steps": 1,
                            "residual_diff_threshold": 1,
                            "max_continuous_cached_steps": 2,
                        },
                        4,
                    ),
                ]
                for plan in plans:
                    repeated = []
                    for request in range(2):
                        with DenoiseWorkCounter(
                            model,
                            used_length=branch.used_len,
                            video_outputs=int(branch.update_mask.sum()),
                            audio_outputs=int(branch.audio_update_mask.sum()),
                        ) as counter:
                            with request_cache(model, plan):
                                for step in range(4):
                                    with counter.step(step):
                                        actual = model(**kwargs)
                                    for a, b in zip(actual, expected):
                                        torch.testing.assert_close(
                                            a, b, atol=1e-5, rtol=1e-5
                                        )
                            torch.accelerator.synchronize()
                            steps = counter.finish_steps()
                        assert not getattr(model, "_h3_cache_active", False)
                        assert not hasattr(model, "_h3_tea_cache")
                        assert counter.calls == 4
                        counts = [s["executed_blocks"] for s in steps]
                        if plan.options.get("rel_l1_thresh") == 0:
                            assert counts == [6] * 4
                        else:
                            assert min(counts) < 6
                        assert sum(s["useful_flops"] for s in steps) == sum(
                            counter.by_layer.values()
                        )
                        repeated.append(counts)
                    assert repeated[0] == repeated[1]
                    print(
                        f"rank={rank} tp={world} {plan.backend} "
                        f"{plan.options} counts={repeated[0]} PASS",
                        flush=True,
                    )
        finally:
            cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
