# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU execution trace of the real Flash-V100 Python backend.

Only device allocation and native/JIT boundaries are replaced. No routing,
candidate, layout, metadata or forward method is mocked. Addresses are compared
within a case and serialized as storage identities, never as machine addresses.
Run with ``python -m tools.sm70.flash_v100_trace --output trace.json``.
"""

import argparse
import inspect
import itertools
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

BASELINE = "8c96e32e56c09d4a3e3112cb5d1a367571f69476"
PACKAGE = "vllm.v1.attention.backends.flash_v100"


class CPUTensor(torch.Tensor):
    """Real CPU storage with CUDA admission properties; never a CUDA kernel."""

    recorder = None

    @property
    def device(self):
        return torch.device("cuda:0")

    @property
    def is_cuda(self):
        return True

    def cpu(self, *args, **kwargs):
        return self.as_subclass(torch.Tensor)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if cls.recorder is not None and func.__name__ in (
            "copy_",
            "fill_",
            "zero_",
            "index_copy_",
            "__setitem__",
        ):
            cls.recorder.events.append(
                ["write", func.__name__, cls.recorder.describe(args[0])]
            )
        return super().__torch_function__(func, types, args, kwargs or {})


@contextmanager
def cpu_cuda(monkeypatch, capture):
    """Redirect explicit CUDA factories and transfers, preserving real views."""
    original_to = torch.Tensor.to

    def transfer(tensor, *args, **kwargs):
        args = list(args)
        device = kwargs.get("device")
        if args and isinstance(args[0], (str, torch.device)):
            device = args[0]
            args[0] = "cpu"
        if "device" in kwargs:
            kwargs["device"] = "cpu"
        result = original_to(tensor, *args, **kwargs)
        if device is not None:
            cls = CPUTensor if torch.device(device).type == "cuda" else torch.Tensor
            result = result.as_subclass(cls)
        return result

    def factory(fn):
        def allocate(*args, **kwargs):
            device = kwargs.get("device")
            if device is not None:
                kwargs["device"] = "cpu"
            kwargs.pop("pin_memory", None)
            result = fn(*args, **kwargs)
            if device is not None and torch.device(device).type == "cuda":
                return result.as_subclass(CPUTensor)
            return result

        return allocate

    monkeypatch.setattr(torch.Tensor, "to", transfer)
    for name in (
        "empty",
        "zeros",
        "ones",
        "full",
        "arange",
        "tensor",
        "empty_like",
        "zeros_like",
        "ones_like",
        "full_like",
    ):
        monkeypatch.setattr(torch, name, factory(getattr(torch, name)))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capture)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda *a, **k: SimpleNamespace(cuda_stream=1)
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability

    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda *a, **k: DeviceCapability(7, 0),
    )
    monkeypatch.setattr(current_platform, "is_device_capability", lambda x: x == 70)
    monkeypatch.setattr(current_platform, "fp8_dtype", lambda: torch.float8_e4m3fn)
    yield


@contextmanager
def strict_shim():
    """Test-only guard: stale private patch targets must fail immediately."""
    from vllm.v1.attention.backends import flash_attn_v100 as legacy

    original = type(legacy)

    class StrictModule(original):
        def __setattr__(self, name, value):
            if not name.startswith("__") and not legacy._owners(name):
                raise AttributeError(f"Flash-V100 patch has no owner: {name}")
            super().__setattr__(name, value)

        def __delattr__(self, name):
            if not legacy._owners(name):
                raise AttributeError(f"Flash-V100 patch has no owner: {name}")
            super().__delattr__(name)

    legacy.__class__ = StrictModule
    try:
        yield legacy
    finally:
        legacy.__class__ = original


class Recorder:
    def __init__(self):
        self.events = []
        self.storages = {}
        self.keepalive = []

    def describe(self, value):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            pointer = storage.data_ptr()
            if pointer not in self.storages:
                self.storages[pointer] = len(self.storages)
                self.keepalive.append(storage)
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "stride": list(value.stride()),
                "storage": self.storages[pointer],
                "offset": value.storage_offset(),
            }
        if isinstance(value, (tuple, list)):
            return [self.describe(x) for x in value]
        if isinstance(value, dict):
            return {str(k): self.describe(v) for k, v in value.items()}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return type(value).__name__

    def op(self, name, out_position=None, decline=False):
        def execute(*args, **kwargs):
            self.events.append(["op", name, self.describe(args), self.describe(kwargs)])
            if decline:
                return None
            if name == "bridge":
                args[4].zero_()
                args[5].zero_()
                return None
            out = kwargs.get("out")
            if out is None and out_position is not None:
                out = args[out_position]
            if out is None:
                out = torch.empty_like(args[0])
            out.zero_()
            return out

        return execute

    def profile(self, frame, event, result):
        if not frame.f_globals.get("__name__", "").startswith(PACKAGE):
            return
        name = frame.f_code.co_name
        if event == "call" and name == "_record_route":
            self.events.append(
                ["route", frame.f_locals.get("name", frame.f_locals.get("route"))]
            )
        if event == "call" and (
            name == "_reset_decode_cache"
            or (
                name == "invalidate"
                and frame.f_globals["__name__"] == PACKAGE + ".workspace"
            )
        ):
            caller = frame.f_back
            source, start = inspect.getsourcelines(caller.f_code)
            sites = [
                start + i
                for i, text in enumerate(source)
                if "self._reset_decode_cache()" in text
                or "self.workspace.decode_cache.invalidate()" in text
            ]
            self.events.append(
                ["reset", caller.f_code.co_name, sites.index(caller.f_lineno)]
            )
        if (
            name.startswith(("_should_", "_try_", "_supports_", "_run_"))
            or name
            in ("select_route", "_small_query_decode_enabled", "_has_prefix_context")
        ) and event in ("call", "return"):
            self.events.append(
                [event, name, self.describe(result) if event == "return" else None]
            )


def install_ops(monkeypatch, recorder, legacy, case):
    original_read = legacy.FlashAttnV100Impl.__getattribute__

    def read_predicate(instance, name):
        value = original_read(instance, name)
        if name.startswith("use_") and sys._getframe(1).f_code.co_name == "forward":
            recorder.events.append(["predicate", name, value])
        return value

    monkeypatch.setattr(legacy.FlashAttnV100Impl, "__getattribute__", read_predicate)
    names = (
        "dense",
        "bhmd",
        "decode",
        "xqa",
        "wmma",
        "paged",
        "paged_bhmd",
        "bfla",
        "splitkv",
    )
    native = tuple(
        None if name in case.get("unavailable", ()) else recorder.op(name)
        for name in names
    )
    if native[3] is not None:
        native[3].shared_decode_strategy_revision = case.get("native_revision", 0)
    monkeypatch.setattr(legacy, "_get_flash_ops", lambda: native)
    monkeypatch.setattr(
        legacy, "_get_flash_grouped_verify_op", lambda: recorder.op("grouped_verify")
    )
    for name in ("_get_fp8_e5m2_paged_kv_bridge_op", "_get_sm70_v37_e4m3_bridge_op"):
        monkeypatch.setattr(legacy, name, lambda: recorder.op("bridge", 5))
    monkeypatch.setattr(
        legacy,
        "_get_sm70_splitd_d256_ops",
        lambda: (
            recorder.op("splitd_dense", 3, case.get("decline", False)),
            recorder.op("splitd_paged", 4, case.get("decline", False)),
            None,
        ),
    )
    monkeypatch.setattr(legacy, "_get_sm70_d256_gqa_architecture_op", lambda: None)
    monkeypatch.setattr(legacy, "_get_paged_kv_utils", lambda: None)
    from triton.runtime.jit import JITFunction

    from vllm.v1.attention.backends import triton_attn
    from vllm.v1.attention.backends.flash_v100 import impl
    from vllm.v1.attention.ops import sm70_grouped_scalar

    monkeypatch.setattr(triton_attn, "unified_attention", recorder.op("triton_unified"))

    monkeypatch.setattr(
        impl, "load_grouped_fp16_fp32", lambda: recorder.op("grouped_fp16", 5)
    )
    monkeypatch.setattr(
        impl, "load_grouped_e4m3_fp32", lambda: recorder.op("grouped_e4m3", 5)
    )
    monkeypatch.setattr(
        sm70_grouped_scalar, "scalar_tail_attention_available", lambda: False
    )

    def jit_run(kernel, *args, **kwargs):
        recorder.events.append(
            ["jit", kernel.__name__, recorder.describe(args), recorder.describe(kwargs)]
        )
        assert kernel.__name__ == "_ddtree_paged_attention_kernel", kernel.__name__
        args[5].zero_()

    monkeypatch.setattr(JITFunction, "run", jit_run)


def cases():
    axes = itertools.product(
        ("prefill", "prefix", "decode", "mixed"),
        ("auto", "fp8_e4m3"),
        (128, 256),
        (1, 6),
        (False, True),
        ("none", "dflash2", "ddtree", "mtp"),
        ("none", "swa", "anchor"),
    )
    for stage, codec, head, gqa, capture, spec, mask in axes:
        yield dict(
            stage=stage,
            codec=codec,
            head=head,
            gqa=gqa,
            capture=capture,
            spec=spec,
            mask=mask,
        )
    manifest = Path(__file__).with_name("flash_v100_trace_cases.jsonl")
    yield from (json.loads(line) for line in manifest.read_text().splitlines())


def prepare_spec(case, legacy, metadata, layer, lengths):
    builder = object.__new__(legacy.FlashAttnV100MetadataBuilder)
    builder.device = torch.device("cuda:0")
    builder.block_size = case.get("page", 16)
    builder.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=len(lengths)),
        model_config=SimpleNamespace(max_model_len=131072),
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=None, cudagraph_capture_sizes=[]
        ),
    )
    builder._is_speculative_draft_model = False
    from vllm.v1.attention.backends.flash_v100.spec.hooks import METADATA_HOOKS

    METADATA_HOOKS.initialize(builder, None)
    common = SimpleNamespace(
        num_reqs=len(lengths),
        query_start_loc_cpu=metadata.query_start_loc_cpu,
        seq_lens_cpu=metadata.seq_lens_cpu,
    )

    def update():
        if case.get("draft"):
            layer.is_dflash_draft_attn = True
            layer.dflash_expected_causal = metadata.causal = False
            layer.dflash_expected_sliding_window = 32 if case["mask"] == "swa" else None
            builder._stabilize_draft_graph_metadata(metadata, common)
        elif case["spec"] == "ddtree":
            parent = [
                [0] + [max(0, j - 2) for j in range(1, max(lengths))] for _ in lengths
            ]
            builder._attach_ddtree_metadata(
                metadata,
                ddtree_parent_ids=torch.tensor(
                    parent, dtype=torch.int32, device="cuda"
                ),
                ddtree_num_tree_tokens_cpu=torch.tensor(lengths, dtype=torch.int32),
            )
        elif case["spec"] in ("dflash2", "mtp"):
            metadata.is_dflash_selector_target = case["spec"] == "dflash2"
            builder._update_smallq_decode_metadata(
                metadata, common, force=case["capture"]
            )

    return builder, update


def run_case(case):
    import pytest

    import vllm.envs as envs

    recorder = Recorder()
    clean_env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("VLLM_FLASH_V100_", "VLLM_SM70_DFLASH", "VLLM_DFLASH"))
    }
    clean_env.update(case.get("env", {}))
    with (
        patch.dict(os.environ, clean_env, clear=True),
        pytest.MonkeyPatch.context() as mp,
    ):
        envs.disable_envs_cache()
        with cpu_cuda(mp, case["capture"]), strict_shim() as legacy:
            install_ops(mp, recorder, legacy, case)
            from vllm.v1.attention.backends.flash_v100 import dense_prefill, state

            dense_prefill.clear_flash_attn_v100_workspaces()
            for name, value in vars(state).items():
                if isinstance(value, bool):
                    mp.setattr(state, name, False)
            config = dict(
                num_heads=case["gqa"],
                num_kv_heads=1,
                head_size=case["head"],
                scale=case["head"] ** -0.5,
                alibi_slopes=None,
                kv_cache_dtype=case["codec"],
                sliding_window=32 if case["mask"] == "swa" else None,
            )
            if case["mask"] == "anchor":
                config["prefix_anchored_decode_window"] = 32
            qlen = case.get("qlen", 4 if case["spec"] != "none" else 32)
            lengths = {
                "prefill": [qlen],
                "prefix": [qlen],
                "decode": [1],
                "mixed": [qlen, 1, 4],
            }[case["stage"]]
            lengths = case.get("lengths", lengths)
            total = sum(lengths)
            seq = [q + (0 if case["stage"] == "prefill" else 64) for q in lengths]
            qsl = torch.tensor(
                [0] + list(itertools.accumulate(lengths)), dtype=torch.int32
            )
            tensor = lambda data: torch.tensor(data, dtype=torch.int32, device="cuda")
            shape = (total, case["gqa"], case["head"])
            query = torch.zeros(shape, dtype=torch.float16, device="cuda")
            key = torch.zeros(
                (total, 1, case["head"]), dtype=torch.float16, device="cuda"
            )
            output = torch.full(shape, 7, dtype=torch.float16, device="cuda")
            page = case.get("page", 16)
            blocks = (max(seq) + page - 1) // page
            cache = torch.zeros(
                (blocks * len(seq), 2, page, 1, case["head"]),
                dtype=torch.uint8 if case["codec"] != "auto" else torch.float16,
                device="cuda",
            )
            if case.get("contiguous_kv"):
                cache = tuple(x.contiguous() for x in cache.unbind(1))
            metadata = SimpleNamespace(
                num_actual_tokens=total,
                max_query_len=max(lengths),
                query_start_loc=qsl.to("cuda"),
                query_start_loc_cpu=qsl,
                seq_lens=tensor(seq),
                seq_lens_cpu=torch.tensor(seq),
                max_seq_len=max(seq),
                max_model_len=131072,
                block_table=tensor(list(range(blocks * len(seq)))).view(
                    len(seq), blocks
                ),
                slot_mapping=tensor(list(range(total))),
                causal=True,
                prefix_anchor_lens=tensor([16] * len(seq)),
                decode_sliding_window=32,
                use_cascade=False,
                seq_threshold_3D=0,
                num_par_softmax_segments=16,
                softmax_segm_output=None,
                softmax_segm_max=None,
                softmax_segm_expsum=None,
                mm_prefix_range_tensor=None,
            )
            layer = SimpleNamespace(
                _k_scale_float=0.5,
                _v_scale_float=0.75,
                layer_name="trace.attn",
                is_dflash_draft_attn=False,
                _k_scale=torch.tensor(0.5),
                _v_scale=torch.tensor(0.75),
            )
            for name, value in case.get("metadata", {}).items():
                setattr(metadata, name, value)
            builder, update_metadata = prepare_spec(
                case, legacy, metadata, layer, lengths
            )
            # Register inputs first; transient allocation addresses cannot affect IDs.
            recorder.describe(
                [
                    query,
                    key,
                    cache,
                    output,
                    metadata.query_start_loc,
                    metadata.seq_lens,
                    metadata.block_table,
                ]
            )
            old_profile = sys.getprofile()

            def observe(frame, event, result):
                recorder.profile(frame, event, result)
                if old_profile is not None:
                    old_profile(frame, event, result)

            try:
                implementation = legacy.FlashAttnV100Impl(**config)
                for name, value in case.get("after_init_env", {}).items():
                    mp.setenv(name, value)
                sys.setprofile(observe)
                CPUTensor.recorder = recorder
                for repeat in range(2):
                    recorder.events.append(["forward", repeat])
                    update_metadata()
                    result = implementation.forward(
                        layer,
                        query,
                        key,
                        key,
                        cache,
                        None if case.get("metadata_none") else metadata,
                        output=output,
                    )
                    assert result.data_ptr() == output.data_ptr()
                    buffers = {
                        name: recorder.describe(value)
                        for name, value in vars(builder).items()
                        if isinstance(value, torch.Tensor)
                    }
                    recorder.events.append(["buffers", buffers])
            except (ValueError, RuntimeError) as error:
                expected = []
                if case["mask"] == "anchor" and case["codec"] != "auto":
                    expected.append("prefix-anchored SWA requires")
                if case["mask"] == "anchor" and case["spec"] == "ddtree":
                    expected.append("does not support ddtree")
                if (
                    case["spec"] == "ddtree"
                    and case["codec"] != "auto"
                    and case["capture"]
                ):
                    expected.append("unsupported key cache dtype")
                if not any(message in str(error) for message in expected):
                    raise
                recorder.events.append(["rejected", type(error).__name__, str(error)])
            finally:
                CPUTensor.recorder = None
                sys.setprofile(old_profile)
                envs.disable_envs_cache()
    return {"case": case, "events": recorder.events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    traces = [run_case(case) for case in itertools.islice(cases(), args.limit)]
    args.output.write_text(
        json.dumps({"baseline": BASELINE, "traces": traces}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
