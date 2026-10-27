# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load the real DSpark checkpoint on TP4 without loading the target weights.

This qualifies weight coverage and draft dense/MoE execution, not speculative
generation. Attention correctness has separate paged, non-causal component tests.
"""

import argparse
import json
import socket
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp


def _worker(rank, address, root, draft, output):
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader import get_model

    torch.accelerator.set_device_index(rank)
    model_path = next((root / "gguf").glob("*-00001-of-00004.gguf"))
    config = EngineArgs(
        model=str(model_path),
        hf_config_path=str(root / "base"),
        tokenizer=str(root / "base"),
        dtype="half",
        quantization="gguf",
        load_format="gguf",
        tensor_parallel_size=4,
        max_model_len=16384,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_custom_all_reduce=True,
        speculative_config={
            "method": "dspark",
            "model": str(draft),
            "quantization": "fp8",
            "num_speculative_tokens": 5,
            "enforce_eager": True,
            "draft_load_config": {"load_format": "safetensors"},
        },
        kernel_config={"sm70_ring": {"enabled": False}},
    ).create_engine_config()
    started = time.monotonic()
    with set_current_vllm_config(config):
        init_distributed_environment(4, rank, address, rank, backend="nccl")
        initialize_model_parallel(4)
        spec = config.speculative_config
        model = get_model(
            vllm_config=config,
            model_config=spec.draft_model_config,
            load_config=spec.draft_load_config,
            prefix="eagle_head",
        )
        loaded = time.monotonic() - started
        with torch.inference_mode(), torch.device("cuda", rank):
            torch.manual_seed(415)
            auxiliary = torch.randn(64, 15360, dtype=torch.float16) * 0.01
            context = model.combine_hidden_states(auxiliary)
            positions = torch.arange(64)
            model.precompute_and_store_context_kv(context, positions)
            embeddings = torch.randn(5, 5120, dtype=torch.float16) * 0.01
            with set_forward_context(None, config, num_tokens=5):
                hidden = model(
                    torch.full((5,), 128799), torch.arange(64, 69), embeddings
                )
            transition = model.markov_bias(model.markov_embed(torch.tensor([22])))
            confidence = model.confidence_logits(
                hidden, model.markov_embed(torch.full((5,), 22))
            )
            for value in (context, hidden, transition, confidence):
                if not torch.isfinite(value).all():
                    raise ValueError("Nonfinite values in native V4.1 draft probe")
        torch.accelerator.synchronize(rank)
        report = dict(
            rank=rank,
            passed=True,
            scope="Draft weights/dense/MoE only; speculative acceptance untested",
            load_seconds=loaded,
            allocated_bytes=torch.accelerator.memory_allocated(rank),
            peak_bytes=torch.accelerator.max_memory_allocated(rank),
            hidden_abs_max=float(hidden.abs().max()),
            markov_abs_max=float(transition.abs().max()),
        )
        (output / f"draft-checkpoint-rank{rank}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        torch.distributed.barrier()
        del model
    cleanup_dist_env_and_memory()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(
        _worker,
        args=(f"tcp://127.0.0.1:{port}", args.model_root, args.draft, args.output),
        nprocs=4,
        join=True,
    )


if __name__ == "__main__":
    main()
