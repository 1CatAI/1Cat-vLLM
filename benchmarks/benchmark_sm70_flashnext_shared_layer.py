# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture complete non-PLE decoder layers using retained real decode inputs.

Run only while owning every TP GPU lock. This measures standalone layer graphs,
not endpoint TPOT. Eager collection preserves real attention metadata; graph
capture uses decode semantics. M5 is collected from actual MTP4 verification.
"""

import argparse
import copy
import dataclasses
import hashlib
import json
import statistics
from pathlib import Path

import torch


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if dataclasses.is_dataclass(value):
        result = copy.copy(value)
        for field in dataclasses.fields(value):
            object.__setattr__(result, field.name, snapshot(getattr(value, field.name)))
        return result
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(snapshot(v) for v in value)
    if isinstance(value, list):
        return [snapshot(v) for v in value]
    return value


class SharedLayerWorkerExtension:
    def enable_retained_qsa_jointprep(self, layers):
        from benchmarks.kernels.sm70_qsa_jointprep_research import attach

        self._qsa_jointprep = True
        for index in layers:
            layer = self._shared_layer_records[index][0]
            if layer.layer_type != "full_attention":
                raise ValueError("Joint preparation requires a QSA layer")
            attach(layer.self_attn)
        return {"rank": self.rank, "qsa_layers": layers}

    def retain_shared_layer_inputs(
        self, width, layers, qsa_jointprep=False, gdn_conv_chain=False
    ):
        from vllm.compilation.sm70_decode_graph import sm70_decode_graph_compilation
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context
        from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpDecoderLayer
        from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

        self._qsa_jointprep = qsa_jointprep
        self._gdn_conv_chain = gdn_conv_chain
        self._shared_layer_records = {}
        self._shared_layer_hooks = []
        model = self.model_runner.get_model()
        prepared = sum(
            bool(getattr(m, "_sm70_qwen38_shared_chain", False))
            for m in model.modules()
        )
        if prepared != 48:
            raise RuntimeError(f"Expected 48 prepared shared experts, got {prepared}")

        runner = self.model_runner
        self._shared_original_dispatch = runner.cudagraph_manager.dispatch
        self._shared_original_forward = model.forward

        def collect_dispatch(num_reqs, num_tokens, uniform_token_count):
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE, num_tokens=num_tokens, num_reqs=num_reqs
            )

        def collect_forward(*args, **kwargs):
            context = get_forward_context()
            context.skip_compiled = True
            decode = any(
                getattr(meta, "num_decode_tokens", 0)
                or getattr(meta, "num_spec_decode_tokens", 0)
                for meta in context.attn_metadata.values()
            )
            with sm70_decode_graph_compilation(bool(decode)):
                return self._shared_original_forward(*args, **kwargs)

        runner.cudagraph_manager.dispatch = collect_dispatch
        model.forward = collect_forward

        def retain(layer, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            if (
                layer.layer_idx in self._shared_layer_records
                or hidden.shape[0] != width
            ):
                return
            context = get_forward_context()
            gdn = getattr(layer, "linear_attn", None)
            metadata = (
                context.attn_metadata.get(gdn.prefix) if gdn is not None else None
            )
            if width == 5 and not any(
                getattr(meta, "num_spec_decode_tokens", 0) == 5
                and getattr(meta, "num_spec_decodes", 0) == 1
                for meta in context.attn_metadata.values()
            ):
                return
            if width == 1 and not any(
                getattr(meta, "num_decode_tokens", 0) == 1
                and getattr(meta, "num_prefill_tokens", 0) == 0
                for meta in context.attn_metadata.values()
            ):
                return
            saved_context = copy.copy(context)
            saved_context.attn_metadata = snapshot(context.attn_metadata)
            saved_context.slot_mapping = snapshot(context.slot_mapping)
            saved_context.all_moe_layers = None
            saved_context.moe_layer_index = 0
            state = []
            if gdn is not None:
                indices = [
                    getattr(metadata, name)
                    for name in (
                        "spec_state_indices_tensor",
                        "non_spec_state_indices_tensor",
                    )
                    if getattr(metadata, name) is not None
                ]
                indices = torch.cat([v.reshape(-1) for v in indices]).long().unique()
                indices = indices[indices >= 0]
                for cache in gdn.kv_cache:
                    state.append(
                        (cache, indices, cache.index_select(0, indices).clone())
                    )
            self._shared_layer_records[layer.layer_idx] = (
                layer,
                snapshot(args),
                snapshot(kwargs),
                saved_context,
                state,
            )

        for layer in model.modules():
            if isinstance(layer, Qwen4ExpDecoderLayer) and layer.layer_idx in layers:
                if layer.ple is not None:
                    raise ValueError("Select non-PLE layers for this GPU-only capture")
                if qsa_jointprep and layer.layer_type == "full_attention":
                    from benchmarks.kernels.sm70_qsa_jointprep_research import attach

                    attach(layer.self_attn)
                if gdn_conv_chain and layer.layer_type == "linear_attention":
                    from benchmarks.kernels.sm70_gdn_conv_chain_research import attach

                    attach(layer.linear_attn)
                self._shared_layer_hooks.append(
                    layer.register_forward_pre_hook(retain, with_kwargs=True)
                )
        return {"rank": self.rank, "prepared_shared_experts": prepared}

    @torch.inference_mode()
    def measure_shared_layers(
        self, width, layers, output, eager_layer=False, repeats=7, replays=100
    ):
        from benchmarks.kernels.sm70_chain_screen_utils import graph_kernel_geometry
        from vllm.compilation.sm70_decode_graph import sm70_decode_graph_compilation
        from vllm.config import set_current_vllm_config
        from vllm.distributed import graph_capture
        from vllm.forward_context import override_forward_context

        for hook in self._shared_layer_hooks:
            hook.remove()
        self.model_runner.cudagraph_manager.dispatch = self._shared_original_dispatch
        self.model_runner.get_model().forward = self._shared_original_forward
        missing = set(layers) - self._shared_layer_records.keys()
        if missing:
            raise RuntimeError(f"No real M{width} decode inputs for layers {missing}")
        results = []
        for index in layers:
            layer, args, kwargs, context, state = self._shared_layer_records[index]
            shared = layer.mlp.shared_expert
            graphs, outputs = {}, {}
            gdn = getattr(layer, "linear_attn", None)
            original_cache = None
            if gdn is not None:
                # Hybrid caches expose FP16 convolution and FP32 recurrent
                # views of one byte allocation. AOT cannot functionalize those
                # aliasing mutations. Independent benchmark storage retains
                # every original stride and the captured active state.
                original_cache = gdn.kv_cache
                cloned_cache = tuple(
                    torch.empty_strided(
                        value.shape,
                        value.stride(),
                        dtype=value.dtype,
                        device=value.device,
                    ).copy_(value)
                    for value in original_cache
                )
                remap = {id(a): b for a, b in zip(original_cache, cloned_cache)}
                state = [(remap[id(a)], ids, values) for a, ids, values in state]
                gdn.kv_cache = cloned_cache

            def restore(saved_state=state):
                for cache, ids, values in saved_state:
                    cache.index_copy_(0, ids, values)

            try:
                with (
                    override_forward_context(context),
                    sm70_decode_graph_compilation(),
                    set_current_vllm_config(self.vllm_config),
                ):
                    for arm in ("control", "candidate"):
                        shared._sm70_qwen38_shared_chain = (
                            True
                            if self._qsa_jointprep or self._gdn_conv_chain
                            else arm == "candidate"
                        )
                        if self._qsa_jointprep and layer.layer_type == "full_attention":
                            layer.self_attn._qsa_jointprep_enabled = arm == "candidate"
                        if self._gdn_conv_chain and gdn is not None:
                            gdn._gdn_conv_chain_enabled = arm == "candidate"
                        forward = (
                            layer.forward
                            if eager_layer
                            else torch.compile(
                                layer.forward, backend="inductor", fullgraph=True
                            )
                        )
                        restore()
                        for _ in range(3):
                            forward(*args, **kwargs)
                        torch.cuda.synchronize()
                        restore()
                        with graph_capture(self.device) as capture:
                            graph = torch.cuda.CUDAGraph(keep_graph=True)
                            with torch.cuda.graph(graph, stream=capture.stream):
                                outputs[arm] = forward(*args, **kwargs)
                        graphs[arm] = graph
                    checked = {}
                    checked_states = {}
                    selected_ids = {}
                    for arm, graph in graphs.items():
                        restore()
                        graph.replay()
                        torch.cuda.synchronize()
                        checked[arm] = [x.clone() for x in outputs[arm]]
                        checked_states[arm] = [
                            cache.index_select(0, ids).clone()
                            for cache, ids, _ in state
                        ]
                        if self._qsa_jointprep:
                            selected_ids[arm] = layer.self_attn.topk_indices_buffer[
                                :width
                            ].clone()
                    errors = [
                        {
                            "max_abs": float((a.float() - b.float()).abs().max()),
                            "rel_l2": float(
                                (a.float() - b.float()).norm()
                                / a.float().norm().clamp_min(1e-30)
                            ),
                        }
                        for a, b in zip(checked["control"], checked["candidate"])
                    ]
                    timings = {arm: [] for arm in graphs}
                    for repetition in range(repeats):
                        order = list(graphs)[:: 1 if repetition % 2 == 0 else -1]
                        for arm in order:
                            graph = graphs[arm]
                            for _ in range(10):
                                graph.replay()
                            start, end = (
                                torch.cuda.Event(enable_timing=True),
                                torch.cuda.Event(enable_timing=True),
                            )
                            start.record()
                            for _ in range(replays):
                                graph.replay()
                            end.record()
                            end.synchronize()
                            timings[arm].append(start.elapsed_time(end) / replays)
                    geometry = {
                        arm: graph_kernel_geometry(
                            graph,
                            Path(output),
                            f"M{width}.layer{index}.{arm}.rank{self.rank}",
                        )
                        for arm, graph in graphs.items()
                    }
                    results.append(
                        {
                            "layer": index,
                            "type": layer.layer_type,
                            "width": width,
                            "errors": errors,
                            "state_errors": [
                                {
                                    "max_abs": float(
                                        (a.float() - b.float()).abs().max()
                                    ),
                                    "rel_l2": float(
                                        (a.float() - b.float()).norm()
                                        / a.float().norm().clamp_min(1e-30)
                                    ),
                                }
                                for a, b in zip(
                                    checked_states["control"],
                                    checked_states["candidate"],
                                )
                            ],
                            "selected_ids_equal": (
                                bool(
                                    torch.equal(
                                        selected_ids["control"],
                                        selected_ids["candidate"],
                                    )
                                )
                                if selected_ids
                                else None
                            ),
                            "samples_ms": timings,
                            "medians_ms": {
                                arm: statistics.median(v) for arm, v in timings.items()
                            },
                            "geometry": geometry,
                        }
                    )
            finally:
                if original_cache is not None:
                    gdn.kv_cache = original_cache
                shared._sm70_qwen38_shared_chain = True
                restore()
        return {"rank": self.rank, "layers": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--width", type=int, choices=(1, 5), required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eager-layer", action="store_true")
    parser.add_argument("--qsa-jointprep", action="store_true")
    parser.add_argument("--also-qsa-jointprep", action="store_true")
    parser.add_argument("--gdn-conv-chain", action="store_true")
    parser.add_argument("--input-len", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.94)
    args = parser.parse_args()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=4,
        dtype="half",
        kv_cache_dtype="float16",
        mamba_ssm_cache_dtype="float32",
        max_model_len=args.max_model_len,
        max_num_batched_tokens=8192,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        language_model_only=True,
        enforce_eager=False,
        speculative_config={"method": "mtp", "num_speculative_tokens": 4}
        if args.width == 5
        else None,
        worker_extension_cls="benchmarks.benchmark_sm70_flashnext_shared_layer.SharedLayerWorkerExtension",
    )
    try:
        routes = llm.collective_rpc(
            "retain_shared_layer_inputs",
            args=(args.width, args.layers, args.qsa_jointprep, args.gdn_conv_chain),
        )
        tokenizer = llm.get_tokenizer()
        piece = tokenizer.encode(
            "This fixed benchmark prompt is used to create a deterministic "
            "tokenized input for single-request decode measurement. ",
            add_special_tokens=False,
        )
        ids = (piece * ((args.input_len + len(piece) - 1) // len(piece)))[
            : args.input_len
        ]
        llm.generate(
            [{"prompt_token_ids": ids}],
            SamplingParams(temperature=0, max_tokens=128, ignore_eos=True),
            use_tqdm=False,
        )
        result = llm.collective_rpc(
            "measure_shared_layers",
            args=(args.width, args.layers, str(args.output), args.eager_layer),
            timeout=600,
        )
        additional = {}
        if args.also_qsa_jointprep:
            if args.qsa_jointprep or 3 not in args.layers:
                raise ValueError(
                    "Additional QSA screen requires shared layer 3 capture"
                )
            llm.collective_rpc("enable_retained_qsa_jointprep", args=([3],))
            additional["qsa_jointprep"] = llm.collective_rpc(
                "measure_shared_layers",
                args=(
                    args.width,
                    [3],
                    str(args.output.with_name(args.output.stem + ".qsa.json")),
                    args.eager_layer,
                ),
                timeout=600,
            )
        import vllm._C as native

        args.output.write_text(
            json.dumps(
                {
                    "scope": (
                        "complete non-PLE decoder layers; standalone layer CUDA graphs"
                    ),
                    "compiler": "eager" if args.eager_layer else "standalone_inductor",
                    "candidate": (
                        "gdn_conv_chain"
                        if args.gdn_conv_chain
                        else "qsa_jointprep"
                        if args.qsa_jointprep
                        else "shared_expert_m1"
                    ),
                    "endpoint_speed_acceptance": False,
                    "actual_mtp_verifier": args.width == 5,
                    "input_tokens": args.input_len,
                    "max_model_len": args.max_model_len,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "prompt_sha256": hashlib.sha256(
                        json.dumps(ids).encode()
                    ).hexdigest(),
                    "native_sha256": hashlib.sha256(
                        Path(native.__file__).read_bytes()
                    ).hexdigest(),
                    "routes": routes,
                    "ranks": result,
                    "additional_screens": additional,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
