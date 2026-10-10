# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import regex as re

from vllm.triton_utils import jit_monitor

try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_CUDA = False

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def _reset_state() -> None:
    jit_monitor._active = False
    jit_monitor._reported.clear()
    jit_monitor._recorded.clear()
    jit_monitor._compile_started.clear()


@pytest.fixture(autouse=True)
def _reset_monitor(monkeypatch):
    """Reset global monitor state and the real Triton hooks between tests."""
    monkeypatch.delenv("VLLM_TRITON_JIT_MANIFEST", raising=False)
    saved_hooks = None
    if _HAS_TRITON:
        from triton import knobs

        saved_hooks = (
            knobs.runtime.jit_cache_hook,
            knobs.runtime.jit_post_compile_hook,
        )
    _reset_state()
    yield
    _reset_state()
    if saved_hooks is not None:
        knobs.runtime.jit_cache_hook, knobs.runtime.jit_post_compile_hook = saved_hooks


# ------------------------------------------------------------------
# Helpers — lightweight stand-ins for triton.knobs
# ------------------------------------------------------------------


def _make_fake_knobs(*, autotuning_print=False, jit_hook=None, cache_hook=None):
    """Build a minimal fake ``triton.knobs`` namespace."""
    autotuning = SimpleNamespace(print=autotuning_print)
    runtime = SimpleNamespace(jit_cache_hook=cache_hook, jit_post_compile_hook=jit_hook)
    return SimpleNamespace(autotuning=autotuning, runtime=runtime)


def _patch_triton_knobs(fake_knobs):
    """Context manager that makes ``from triton import knobs`` return *fake_knobs*."""
    fake_triton = SimpleNamespace(knobs=fake_knobs)
    return mock.patch.dict(sys.modules, {"triton": fake_triton})


def _activate_fake():
    fake = _make_fake_knobs()
    with _patch_triton_knobs(fake):
        jit_monitor.activate()
    return fake


def _compile_kwargs(key="k1", *, module="vllm.fake_ops", warmup=False):
    """Keyword arguments Triton passes to its JIT hooks."""
    return dict(
        key=key,
        repr="r",
        fn=SimpleNamespace(
            name="fake_kernel",
            module=module,
            jit_function=SimpleNamespace(arg_names=["x_ptr", "BLOCK"]),
        ),
        compile={
            "key": key,
            "constants": {(1,): 64},
            "specialization_data": json.dumps({"key": key}),
        },
        is_manual_warmup=warmup,
        already_compiled=False,
    )


def _manifest_entry(key="k1", module="vllm.fake_ops"):
    return {
        "version": 1,
        "module": module,
        "qualname": "fake_kernel",
        "key": key,
        "specialization_data": json.dumps({"key": key}),
    }


# ------------------------------------------------------------------
# Unit tests (no GPU required, triton is mocked)
# ------------------------------------------------------------------


class TestActivateBasic:
    def test_sets_active(self):
        assert not jit_monitor.is_active()
        with _patch_triton_knobs(_make_fake_knobs()):
            jit_monitor.activate()
        assert jit_monitor.is_active()

    def test_idempotent(self):
        fake = _make_fake_knobs()
        with _patch_triton_knobs(fake):
            jit_monitor.activate()
            first_hook = fake.runtime.jit_post_compile_hook
            jit_monitor.activate()
            assert fake.runtime.jit_post_compile_hook is first_hook

    def test_logs_info_on_activation(self):
        with (
            mock.patch.object(jit_monitor.logger, "info") as m,
            _patch_triton_knobs(_make_fake_knobs()),
        ):
            jit_monitor.activate()
        m.assert_called_once()
        assert "Kernel JIT monitor activated" in m.call_args[0][0]


class TestAutotuningPrint:
    def test_enables_autotuning_print(self):
        fake = _make_fake_knobs(autotuning_print=False)
        with _patch_triton_knobs(fake):
            jit_monitor.activate()
        assert fake.autotuning.print is True

    def test_respects_user_opt_out(self):
        fake = _make_fake_knobs(autotuning_print=False)
        with (
            mock.patch.dict(os.environ, {"TRITON_PRINT_AUTOTUNING": "0"}),
            _patch_triton_knobs(fake),
        ):
            jit_monitor.activate()
        assert fake.autotuning.print is False

    def test_noop_when_user_already_enabled(self):
        fake = _make_fake_knobs(autotuning_print=True)
        with (
            mock.patch.dict(os.environ, {"TRITON_PRINT_AUTOTUNING": "1"}),
            _patch_triton_knobs(fake),
        ):
            jit_monitor.activate()
        assert fake.autotuning.print is True


class TestJitHook:
    def test_hook_registered(self):
        fake = _make_fake_knobs()
        assert fake.runtime.jit_post_compile_hook is None
        with _patch_triton_knobs(fake):
            jit_monitor.activate()
        assert fake.runtime.jit_post_compile_hook is not None
        assert fake.runtime.jit_cache_hook is not None

    def test_hook_logs_warning(self):
        fake = _make_fake_knobs()
        with _patch_triton_knobs(fake):
            jit_monitor.activate()

        hook = fake.runtime.jit_post_compile_hook
        mock_fn = SimpleNamespace(name="test_kernel")

        with mock.patch.object(jit_monitor.logger, "warning") as m:
            hook(
                key="some_key",
                repr="some_repr",
                fn=mock_fn,
                compile=lambda: None,
                is_manual_warmup=False,
                already_compiled=False,
            )

        m.assert_called_once()
        msg = m.call_args[0][0] % m.call_args[0][1:]
        assert "Triton kernel JIT compilation during inference" in msg
        assert "test_kernel" in msg

    def test_hook_warns_once_per_specialization(self):
        hook = _activate_fake().runtime.jit_post_compile_hook

        with mock.patch.object(jit_monitor.logger, "warning") as m:
            hook(**_compile_kwargs("a"))
            hook(**_compile_kwargs("a"))
            hook(**_compile_kwargs("b"))

        assert m.call_count == 2

    def test_warning_names_constexprs_and_compile_time(self):
        fake = _activate_fake()
        kwargs = _compile_kwargs()

        with mock.patch.object(jit_monitor.logger, "warning") as m:
            assert fake.runtime.jit_cache_hook(**kwargs) is None
            fake.runtime.jit_post_compile_hook(**kwargs)

        msg = m.call_args[0][0] % m.call_args[0][1:]
        assert re.search(r"fake_kernel \(BLOCK=64; \d+ ms\)", msg)

    def test_cache_hook_chains_existing_hook(self):
        existing = mock.MagicMock(return_value=True)
        fake = _make_fake_knobs(cache_hook=existing)
        with _patch_triton_knobs(fake):
            jit_monitor.activate()

        kwargs = _compile_kwargs()
        assert fake.runtime.jit_cache_hook(**kwargs) is True
        existing.assert_called_once_with(**kwargs)

    def test_hook_chains_existing_hook(self):
        existing = mock.MagicMock(return_value="existing_result")
        fake = _make_fake_knobs(jit_hook=existing)
        with _patch_triton_knobs(fake):
            jit_monitor.activate()

        hook = fake.runtime.jit_post_compile_hook
        mock_fn = SimpleNamespace(name="chained_kernel")
        kwargs = dict(
            key="k",
            repr="r",
            fn=mock_fn,
            compile=lambda: None,
            is_manual_warmup=False,
            already_compiled=False,
        )
        result = hook(**kwargs)

        existing.assert_called_once()
        assert result == "existing_result"

    def test_hook_works_without_existing_hook(self):
        fake = _make_fake_knobs(jit_hook=None)
        with _patch_triton_knobs(fake):
            jit_monitor.activate()

        hook = fake.runtime.jit_post_compile_hook
        mock_fn = SimpleNamespace(name="solo_kernel")
        result = hook(
            key="k",
            repr="r",
            fn=mock_fn,
            compile=lambda: None,
            is_manual_warmup=False,
            already_compiled=False,
        )
        assert result is None


class TestNoTritonFallback:
    def test_activate_without_triton(self):
        with mock.patch.object(jit_monitor, "HAS_TRITON", False):
            jit_monitor.activate()
        assert jit_monitor.is_active()


class TestManifestRecording:
    def test_records_each_inference_specialization_once(self, tmp_path, monkeypatch):
        manifest = tmp_path / "jit.jsonl"
        monkeypatch.setenv("VLLM_TRITON_JIT_MANIFEST", str(manifest))
        hook = _activate_fake().runtime.jit_post_compile_hook

        with mock.patch.object(jit_monitor.logger, "warning"):
            hook(**_compile_kwargs("a"))
            hook(**_compile_kwargs("a"))
            hook(**_compile_kwargs("warmup", warmup=True))
            hook(**_compile_kwargs("foreign", module="triton.fake_ops"))
            hook(**_compile_kwargs("b"))

        entries = [json.loads(line) for line in manifest.read_text().splitlines()]
        assert entries == [_manifest_entry("a"), _manifest_entry("b")]

    def test_skips_specializations_other_workers_recorded(self, tmp_path, monkeypatch):
        manifest = tmp_path / "jit.jsonl"
        manifest.write_text(json.dumps(_manifest_entry("a")) + "\n")
        monkeypatch.setenv("VLLM_TRITON_JIT_MANIFEST", str(manifest))
        hook = _activate_fake().runtime.jit_post_compile_hook
        # Another worker records "b" after this one activated.
        with manifest.open("a") as f:
            f.write(json.dumps(_manifest_entry("b")) + "\n")

        with mock.patch.object(jit_monitor.logger, "warning") as warning:
            for key in ("a", "b", "c"):
                hook(**_compile_kwargs(key))

        assert warning.call_count == 3
        lines = manifest.read_text().splitlines()
        assert [json.loads(line)["key"] for line in lines] == ["a", "b", "c"]

    def test_read_manifest_skips_malformed_and_duplicate_lines(self, tmp_path):
        manifest = tmp_path / "jit.jsonl"
        manifest.write_text(
            "not json\n"
            + json.dumps({**_manifest_entry("old"), "version": 0})
            + "\n"
            + json.dumps(_manifest_entry("a"))
            + "\n"
            + json.dumps(_manifest_entry("a"))
            + "\n"
            + '{"version": 1, "module": "vllm.fake_ops"'
        )

        assert jit_monitor._read_manifest(str(manifest)) == [_manifest_entry("a")]


class TestPreload:
    def _use_manifest(self, tmp_path, monkeypatch, entries, jit_function):
        manifest = tmp_path / "jit.jsonl"
        manifest.write_text("".join(json.dumps(e) + "\n" for e in entries))
        monkeypatch.setenv("VLLM_TRITON_JIT_MANIFEST", str(manifest))
        monkeypatch.setattr(jit_monitor, "HAS_TRITON", True)
        monkeypatch.setattr(jit_monitor, "_current_device", lambda: 0)
        monkeypatch.setattr(
            jit_monitor,
            "_resolve_jit_function",
            lambda module, qualname: (
                jit_function if module == "vllm.fake_ops" else None
            ),
        )
        return manifest

    def test_preloads_recorded_specializations(self, tmp_path, monkeypatch):
        kernel = SimpleNamespace(_init_handles=mock.MagicMock())
        jit_function = SimpleNamespace(
            device_caches={0: ({"warm": object()},)},
            preload=mock.MagicMock(return_value=kernel),
        )
        manifest = self._use_manifest(
            tmp_path,
            monkeypatch,
            [
                _manifest_entry("cold"),
                _manifest_entry("warm"),
                _manifest_entry("gone", module="vllm.missing_ops"),
            ],
            jit_function,
        )

        with mock.patch.object(jit_monitor.logger, "info") as info:
            jit_monitor.preload_recorded_kernels()

        jit_function.preload.assert_called_once_with(json.dumps({"key": "cold"}))
        kernel._init_handles.assert_called_once_with()
        assert info.call_args[0][1:] == (1, str(manifest), mock.ANY, 1, 1, 0)
        assert ("vllm.fake_ops", "fake_kernel", "cold") in jit_monitor._recorded

    def test_counts_failed_preloads_without_raising(self, tmp_path, monkeypatch):
        jit_function = SimpleNamespace(
            device_caches={0: ({},)},
            preload=mock.MagicMock(side_effect=RuntimeError("signature changed")),
        )
        manifest = self._use_manifest(
            tmp_path, monkeypatch, [_manifest_entry("a")], jit_function
        )

        with mock.patch.object(jit_monitor.logger, "info") as info:
            jit_monitor.preload_recorded_kernels()

        assert info.call_args[0][1:] == (0, str(manifest), mock.ANY, 0, 0, 1)

    def test_noop_without_manifest(self, tmp_path, monkeypatch):
        device = mock.MagicMock(side_effect=AssertionError("must not run"))
        monkeypatch.setattr(jit_monitor, "_current_device", device)

        jit_monitor.preload_recorded_kernels()
        monkeypatch.setenv("VLLM_TRITON_JIT_MANIFEST", str(tmp_path / "none.jsonl"))
        jit_monitor.preload_recorded_kernels()

        device.assert_not_called()


# ------------------------------------------------------------------
# Integration tests (real Triton, GPU where required)
# ------------------------------------------------------------------

_skip_no_gpu = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_TRITON),
    reason="Requires CUDA GPU and Triton",
)


if _HAS_TRITON:

    @triton.jit
    def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    @triton.heuristics({"BLOCK": lambda args: 16})
    @triton.jit
    def _heuristics_kernel(x_ptr, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(x_ptr + offs, 0.0, mask=offs < n)


def _run_add_kernel(n: int, block: int = 256) -> None:
    """Launch ``_add_kernel`` with vectors of length *n*."""
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty(n, device="cuda")
    grid = ((n + block - 1) // block,)
    _add_kernel[grid](x, y, out, n, BLOCK=block)
    torch.accelerator.synchronize()


@pytest.mark.skipif(not _HAS_TRITON, reason="Requires Triton")
def test_resolve_unwraps_wrappers_and_rejects_foreign_modules(monkeypatch):
    assert jit_monitor._resolve_jit_function("os", "path") is None

    monkeypatch.setattr(jit_monitor, "_KERNEL_MODULE_PREFIX", "")
    resolve = jit_monitor._resolve_jit_function
    assert resolve(__name__, "_heuristics_kernel") is _heuristics_kernel.fn
    assert resolve(__name__, "_add_kernel") is _add_kernel
    assert resolve(__name__, "missing_kernel") is None


@_skip_no_gpu
class TestTritonJitHookIntegration:
    """End-to-end: real Triton kernel, real GPU, real hook."""

    def test_no_warning_on_cached_shape(self):
        _run_add_kernel(1024)

        jit_monitor.activate()
        with mock.patch.object(jit_monitor.logger, "warning") as w:
            _run_add_kernel(1024)
        w.assert_not_called()

    def test_warning_on_new_constexpr(self):
        _run_add_kernel(1024, block=256)

        jit_monitor.activate()
        with mock.patch.object(jit_monitor.logger, "warning") as w:
            # Different BLOCK (a tl.constexpr) forces recompilation.
            _run_add_kernel(1024, block=512)
        w.assert_called()
        msg = w.call_args[0][0] % w.call_args[0][1:]
        assert "_add_kernel" in msg

    def test_recorded_specialization_is_preloaded(self, tmp_path, monkeypatch):
        manifest = tmp_path / "jit.jsonl"
        monkeypatch.setenv("VLLM_TRITON_JIT_MANIFEST", str(manifest))
        # This test's kernel is defined outside the vllm package.
        monkeypatch.setattr(jit_monitor, "_KERNEL_MODULE_PREFIX", "")

        jit_monitor.activate()
        with mock.patch.object(jit_monitor.logger, "warning") as w:
            # n == 1 is its own integer specialization.
            _run_add_kernel(1, block=64)
        w.assert_called_once()
        assert len(manifest.read_text().splitlines()) == 1

        # A new server process starts with empty in-process kernel caches.
        _add_kernel.device_caches.clear()
        jit_monitor._reported.clear()
        jit_monitor.preload_recorded_kernels()
        with mock.patch.object(jit_monitor.logger, "warning") as w:
            _run_add_kernel(1, block=64)
        w.assert_not_called()
