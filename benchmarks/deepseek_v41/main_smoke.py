# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real target/DSpark generation and finite-logit correctness probe."""

import argparse
import json
import os
import time
from pathlib import Path


def _install_finite_checks(worker):
    import torch

    runner = worker.model_runner
    models = [runner.model]
    drafter = getattr(runner, "drafter", None)
    if drafter is not None:
        models.append(drafter.model)
    for module in models:
        original = module.compute_logits

        def checked(hidden, original=original):
            logits = original(hidden)
            if logits is not None:
                finite = torch.isfinite(logits).all()
                if logits.is_cuda:
                    # Correctness probes can wrap the head before its lazy
                    # graph capture. A Python bool would synchronize that
                    # capture; the device assertion is checked on drain.
                    torch._assert_async(finite, "Nonfinite V4.1 correctness logits")
                elif not finite:
                    raise FloatingPointError(
                        "Nonfinite logits in V4.1 correctness probe"
                    )
            return logits

        module.compute_logits = checked


def trim_allocator(worker):
    import torch

    torch.accelerator.synchronize(worker.device)
    # Release dead profiling allocations before any lazy cubin loads. The
    # raw experts, canonical workspace, KV and loaded parameters stay live.
    torch.accelerator.empty_cache()


def _memory_report(worker):
    import torch

    from benchmarks.deepseek_v41.shared_checkpoint_probe import _pss
    from benchmarks.deepseek_v41.shared_host_probe import memory_available

    device = worker.device
    target = worker.model_runner.model.model
    cache = target.layers[0].ffn.experts.quant_method.expert_cache
    counts = cache.runtime.statistics.cpu().tolist()
    loads = writes = 0
    for spec, (_, misses) in zip(cache.specs, counts):
        for geometry in spec.layouts(cache.rank):
            loads += misses * geometry.device_rows * geometry.device_row_bytes
            writes += (
                misses
                * geometry.device_rows
                * (geometry.write_end_block - geometry.first_block)
                * geometry.block_bytes
            )
    return dict(
        device=str(device),
        allocated_bytes=torch.accelerator.memory_allocated(device),
        peak_allocated_bytes=torch.accelerator.max_memory_allocated(device),
        reserved_bytes=torch.accelerator.memory_reserved(device),
        host_available_bytes=memory_available(),
        expert_pool_identity=vars(cache.data.identity),
        expert_pool_pss_bytes=_pss(),
        expert_decode_requests_and_misses_by_layer=counts,
        expert_decode_load_bytes=loads,
        expert_decode_writeback_bytes=writes,
        prefill_cold_transcodes_by_layer=cache.tm.cold_transcodes.cpu().tolist(),
        expert_workspace_bytes=cache.tm.storage_bytes(),
        engram_requests_rows_bytes_reader_ns=target.engram_io.runtime.statistics.cpu().tolist(),
    )


def create_llm(
    root,
    chunk=1024,
    draft=None,
    offload=0.30,
    *,
    kv_cache_mib=None,
    decode_graph=False,
    draft_graph=False,
    custom_allreduce=False,
):
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # Developer-only callable RPCs run in this private local worker group.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from vllm import LLM

    paths = sorted((root / "gguf").glob("*-00001-of-00004.gguf"))
    sidecar = root / "gguf/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
    if len(paths) != 1 or not sidecar.is_file():
        raise ValueError("Expected the audited backbone and explicit Engram sidecar")
    if kv_cache_mib is None:
        kv_cache_mib = 256 if draft is not None else 384
    return LLM(
        model=str(paths[0]),
        hf_config_path=str(root / "base"),
        tokenizer=str(root / "base"),
        dtype="half",
        quantization="gguf",
        load_format="gguf",
        tensor_parallel_size=4,
        max_model_len=int(os.environ.get("VLLM_V41_MAX_LEN", "16384")),
        max_num_seqs=1,
        max_num_batched_tokens=chunk,
        model_loader_extra_config={"engram_path": str(sidecar)},
        gpu_memory_utilization=0.96,
        kv_cache_memory_bytes=kv_cache_mib * 1024**2,
        enforce_eager=not decode_graph,
        compilation_config=(
            {
                "mode": 0,
                "cudagraph_mode": "FULL",
                "cudagraph_capture_sizes": [1, 2, 3, 4, 5, 6],
                "max_cudagraph_capture_size": 6,
            }
            if decode_graph
            else None
        ),
        enable_prefix_caching=os.environ.get("VLLM_V41_SESSION_PREFIX", "0") == "1",
        enable_chunked_prefill=True,
        async_scheduling=os.environ.get("VLLM_V41_ASYNC", "0") == "1",
        disable_custom_all_reduce=not custom_allreduce,
        speculative_config=(
            {
                "method": "dspark",
                "model": str(draft),
                "quantization": "fp8",
                "num_speculative_tokens": 5,
                "draft_load_config": {"load_format": "safetensors"},
                "enforce_eager": not draft_graph,
            }
            if draft is not None
            else None
        ),
        kernel_config={
            "sm70_ring": {"enabled": False},
            "sm70_gguf": {
                "expert_storage": "original",
                "expert_arena": True,
                "expert_cache_host_storage": "shared_original",
                "expert_cache_offload_fraction": offload,
                "expert_cache_fraction": 0,
                "expert_cache_host_slack": 0,
                "expert_load_prefetch_banks": 1,
                "expert_cache_prefill_tokens": chunk,
                "dense_storage": "original",
                "embedding_storage": "original",
                "prefill_min_m": 1,
                "dequant_workspace_bytes": 32 * 1024**2,
            },
        },
    )


def run(root, contexts, chunk, output, draft=None, debug_finite=False):
    from vllm import SamplingParams

    started = time.monotonic()
    llm = create_llm(root, chunk, draft)
    load_seconds = time.monotonic() - started
    llm.collective_rpc(trim_allocator)
    if debug_finite:
        from benchmarks.deepseek_v41.finite_debug import install

        llm.collective_rpc(install)
    llm.collective_rpc(_install_finite_checks)
    tokenizer = llm.get_tokenizer()
    question = tokenizer.encode(
        "\n请用中文简短回答：2+2等于几？\n答：", add_special_tokens=False
    )
    source = tokenizer.encode(
        "这是一段用于验证模型的测试上下文。" * 2048, add_special_tokens=False
    )
    reports = []
    for context in contexts:
        new_tokens = 8 if context == 64 else 32
        count = context - new_tokens
        prompt = source[: count - len(question)] + question
        if len(prompt) != count:
            raise ValueError("Probe prompt cannot fill requested context")
        begin = time.monotonic()
        results = llm.generate(
            [{"prompt_token_ids": prompt}],
            # Longer probes exercise the exact total-context boundary,
            # including positions after an early EOS. Quality is checked
            # separately with the ordinary EOS stopping policy.
            SamplingParams(
                temperature=0, max_tokens=new_tokens, ignore_eos=context != 64
            ),
        )
        elapsed = time.monotonic() - begin
        generated = results[0].outputs[0]
        report = dict(
            status="completed",
            scope="Context-bound execution; not a steady-state performance result",
            dspark=draft is not None,
            context_limit=context,
            prompt_tokens=len(prompt),
            output_tokens=len(generated.token_ids),
            token_ids=list(generated.token_ids),
            text=generated.text,
            finish_reason=generated.finish_reason,
            load_seconds=load_seconds,
            generation_wall_seconds=elapsed,
            tokens_per_second_including_prefill=len(generated.token_ids) / elapsed,
            memory_by_rank=llm.collective_rpc(_memory_report),
        )
        reports.append(report)
        output.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    llm.llm_engine.engine_core.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--context", type=int, choices=(64, 2048, 16384), nargs="+", default=[64]
    )
    parser.add_argument("--chunk", type=int, choices=(512, 1024, 2048), default=2048)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--debug-finite", action="store_true")
    args = parser.parse_args()
    run(
        args.model_root,
        args.context,
        args.chunk,
        args.output,
        args.draft,
        args.debug_finite,
    )


if __name__ == "__main__":
    main()
