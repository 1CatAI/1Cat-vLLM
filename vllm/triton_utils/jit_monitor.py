# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Monitor unexpected Triton kernel JIT compilation during inference.

After server warmup completes, any Triton JIT compilation or autotuning
event indicates a cache miss or unexpected input shape that causes a
latency spike. This module registers hooks in the Triton runtime to
detect and log such events so they can be investigated.

Currently monitors:
- Triton ``@triton.autotune`` cache misses (via ``knobs.autotuning.print``)
- Triton ``@triton.jit`` first-time compilations, once per specialization
  (via ``knobs.runtime.jit_cache_hook`` and
  ``knobs.runtime.jit_post_compile_hook``)

When ``VLLM_TRITON_JIT_MANIFEST`` names a file, every specialization first
compiled during inference is appended to it, and
:func:`preload_recorded_kernels` compiles the recorded specializations while
the next server warms up, before it accepts requests.
"""

import contextlib
import importlib
import json
import os
import threading
import time
from collections.abc import Iterable, Iterator
from typing import IO, Any

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.triton_utils.importing import HAS_TRITON

logger = init_logger(__name__)

_active: bool = False

_MANIFEST_VERSION = 1
# Preloading imports the module that defines each recorded kernel, so only
# kernels defined inside vLLM are recorded or preloaded.
_KERNEL_MODULE_PREFIX = "vllm."

_lock = threading.Lock()
# (module, qualname, cache key) of specializations already warned about.
_reported: set[tuple[str, str, str]] = set()
# (module, qualname, cache key) of specializations present in the manifest.
_recorded: set[tuple[str, str, str]] = set()
# perf_counter() at the jit_cache_hook of each compilation in progress, keyed
# by (id(JITFunction), cache key).
_compile_started: dict[tuple[int, str], float] = {}


def is_active() -> bool:
    """Return whether the JIT compilation monitor is currently active."""
    return _active


def activate() -> None:
    """Enable JIT compilation monitoring after warmup.

    Call once per worker process at the end of
    :func:`compile_or_warm_up_model`.  After activation every Triton
    kernel compilation or autotuning benchmark that happens during
    inference will be logged as a warning.

    Safe to call multiple times — subsequent calls are no-ops.

    If the user has explicitly set ``TRITON_PRINT_AUTOTUNING=0`` in
    their environment, autotuning printing is left disabled; the JIT
    compilation hook is still registered regardless.
    """
    global _active
    if _active:
        return
    _active = True

    manifest = envs.VLLM_TRITON_JIT_MANIFEST
    if manifest:
        variants = [_entry_variant(entry) for entry in _read_manifest(manifest)]
        with _lock:
            _recorded.update(variants)

    _setup_triton_autotuning_print()
    _setup_triton_jit_hook()

    logger.info(
        "Kernel JIT monitor activated — Triton JIT compilations "
        "during inference will be logged as warnings%s.",
        f" and recorded to {manifest}" if manifest else "",
    )


# ------------------------------------------------------------------
# Triton autotuning print
# ------------------------------------------------------------------


def _setup_triton_autotuning_print() -> None:
    """Enable ``TRITON_PRINT_AUTOTUNING`` unless the user opted out."""
    if not HAS_TRITON:
        return
    from triton import knobs  # type: ignore[import-untyped]

    user_val = os.environ.get("TRITON_PRINT_AUTOTUNING")
    if user_val == "0":
        logger.debug(
            "TRITON_PRINT_AUTOTUNING=0 set by user — "
            "autotuning messages will stay suppressed."
        )
        return

    knobs.autotuning.print = True


# ------------------------------------------------------------------
# Triton JIT compilation hooks
# ------------------------------------------------------------------


def _setup_triton_jit_hook() -> None:
    """Register JIT hooks that time and warn on each new specialization."""
    if not HAS_TRITON:
        return
    from triton import knobs  # type: ignore[import-untyped]

    existing_cache_hook = getattr(knobs.runtime, "jit_cache_hook", None)
    existing_hook = knobs.runtime.jit_post_compile_hook

    def _on_jit_cache_miss(**kwargs):
        # Triton calls this before compiling, or loading from its disk cache,
        # a specialization missing from the in-process cache. A truthy return
        # value skips the compilation, so only a chained hook may return one.
        with _lock:
            _compile_started[_timing_key(kwargs)] = time.perf_counter()
        if existing_cache_hook is not None:
            return existing_cache_hook(**kwargs)
        return None

    def _on_jit_compile(**kwargs):
        # `jit_post_compile_hook` is Triton internal API and its
        # signature has changed across releases (kwargs added/renamed).
        # Accept **kwargs so an upstream change cannot crash this hook
        # with TypeError, and forward the full kwarg set to any
        # pre-existing hook unchanged.
        _report_compilation(kwargs)
        if existing_hook is not None:
            return existing_hook(**kwargs)
        return None

    knobs.runtime.jit_cache_hook = _on_jit_cache_miss
    knobs.runtime.jit_post_compile_hook = _on_jit_compile


def _timing_key(kwargs: dict[str, Any]) -> tuple[int, str]:
    # Each hook call receives a new JitFunctionInfo; the JITFunction it wraps
    # identifies the kernel across the cache and post-compile hooks.
    fn = kwargs.get("fn")
    return id(getattr(fn, "jit_function", fn)), str(kwargs.get("key"))


def _report_compilation(kwargs: dict[str, Any]) -> None:
    fn = kwargs.get("fn")
    fn_name = getattr(fn, "name", "<unknown>")
    module = getattr(fn, "module", None)
    module_name = module if isinstance(module, str) else getattr(module, "__name__", "")
    variant = (module_name, fn_name, str(kwargs.get("key")))
    with _lock:
        started = _compile_started.pop(_timing_key(kwargs), None)
        if variant in _reported:
            return
        _reported.add(variant)

    details = _describe_specialization(fn, kwargs.get("compile"))
    if started is not None:
        details += f"; {(time.perf_counter() - started) * 1000:.0f} ms"
    logger.warning(
        "Triton kernel JIT compilation during inference: %s (%s). "
        "This causes a latency spike; consider extending warmup "
        "to cover this shape/config.",
        fn_name,
        details,
    )
    if not kwargs.get("is_manual_warmup", False):
        _record_specialization(variant, kwargs.get("compile"))


def _describe_specialization(fn: Any, compile_info: Any) -> str:
    """Name the constexpr values that selected this specialization."""
    constants = (
        compile_info.get("constants") if isinstance(compile_info, dict) else None
    )
    if not constants:
        return "no constexpr arguments"
    arg_names = getattr(getattr(fn, "jit_function", None), "arg_names", None) or []
    parts = []
    for path, value in constants.items():
        index = path[0] if isinstance(path, tuple) and len(path) == 1 else None
        if isinstance(index, int) and 0 <= index < len(arg_names):
            parts.append(f"{arg_names[index]}={value}")
        else:
            parts.append(f"{path}={value}")
    return ", ".join(parts)[:300]


# ------------------------------------------------------------------
# Specialization manifest
# ------------------------------------------------------------------


def _record_specialization(variant: tuple[str, str, str], compile_info: Any) -> None:
    """Append a specialization compiled during inference to the manifest."""
    manifest = envs.VLLM_TRITON_JIT_MANIFEST
    module_name, qualname, key = variant
    data = (
        compile_info.get("specialization_data")
        if isinstance(compile_info, dict)
        else None
    )
    if (
        not manifest
        or not module_name.startswith(_KERNEL_MODULE_PREFIX)
        or "<locals>" in qualname
        or not isinstance(data, str)
    ):
        return
    with _lock:
        if variant in _recorded:
            return
        _recorded.add(variant)
    entry = {
        "version": _MANIFEST_VERSION,
        "module": module_name,
        "qualname": qualname,
        "key": key,
        "specialization_data": data,
    }
    try:
        _append_manifest_entry(manifest, entry)
    except OSError as exc:
        logger.warning_once(
            "Could not record Triton kernel specializations to %s: %s",
            manifest,
            exc,
        )


def _append_manifest_entry(manifest: str, entry: dict[str, Any]) -> None:
    """Append ``entry`` unless another worker process already recorded it."""
    os.makedirs(os.path.dirname(os.path.abspath(manifest)), exist_ok=True)
    variant = _entry_variant(entry)
    with open(manifest, "a+", encoding="utf-8") as f, _exclusive_lock(f):
        f.seek(0)
        if any(_entry_variant(recorded) == variant for recorded in _parse_manifest(f)):
            return
        f.write(json.dumps(entry) + "\n")


@contextlib.contextmanager
def _exclusive_lock(f: IO[str]) -> Iterator[None]:
    """Serialize manifest appends across the worker processes of a server."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX platforms
        yield
        return
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _parse_manifest(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield well-formed entries, skipping partial or foreign lines."""
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if (
            isinstance(entry, dict)
            and entry.get("version") == _MANIFEST_VERSION
            and all(
                isinstance(entry.get(field), str)
                for field in ("module", "qualname", "key", "specialization_data")
            )
        ):
            yield entry


def _read_manifest(manifest: str) -> list[dict[str, Any]]:
    """Return the first entry recorded for each specialization."""
    entries: dict[tuple[str, str, str], dict[str, Any]] = {}
    try:
        with open(manifest, encoding="utf-8") as f:
            for entry in _parse_manifest(f):
                entries.setdefault(_entry_variant(entry), entry)
    except FileNotFoundError:
        return []
    except OSError as exc:
        logger.warning("Could not read Triton JIT manifest %s: %s", manifest, exc)
        return []
    return list(entries.values())


def _entry_variant(entry: dict[str, Any]) -> tuple[str, str, str]:
    return entry["module"], entry["qualname"], entry["key"]


# ------------------------------------------------------------------
# Startup preloading
# ------------------------------------------------------------------


def preload_recorded_kernels() -> None:
    """Compile every specialization recorded in ``VLLM_TRITON_JIT_MANIFEST``.

    Call once per worker process after warmup and before :func:`activate`.
    Triton stores a preloaded specialization under the in-process cache key
    that a launch with the same argument specialization computes, so that
    launch neither compiles nor loads a binary. Entries whose kernel no longer
    resolves, or no longer accepts the recorded signature, are skipped.
    """
    manifest = envs.VLLM_TRITON_JIT_MANIFEST
    if not manifest or not HAS_TRITON:
        return
    entries = _read_manifest(manifest)
    if not entries:
        return

    start = time.perf_counter()
    device = _current_device()
    preloaded = warm = unresolved = failed = 0
    for entry in entries:
        with _lock:
            _recorded.add(_entry_variant(entry))
        jit_function = _resolve_jit_function(entry["module"], entry["qualname"])
        if jit_function is None:
            unresolved += 1
            continue
        try:
            if _is_compiled(jit_function, device, entry["key"]):
                warm += 1
                continue
            kernel = jit_function.preload(entry["specialization_data"])
            # Load the binary and build its launcher now instead of at the
            # first launch.
            init_handles = getattr(kernel, "_init_handles", None)
            if callable(init_handles):
                init_handles()
            preloaded += 1
        except Exception as exc:
            failed += 1
            logger.debug(
                "Skipping recorded Triton kernel %s.%s: %s",
                entry["module"],
                entry["qualname"],
                exc,
            )
    logger.info(
        "Preloaded %d recorded Triton kernel specializations from %s in %.1f s "
        "(%d already compiled by warmup, %d unresolved, %d failed).",
        preloaded,
        manifest,
        time.perf_counter() - start,
        warm,
        unresolved,
        failed,
    )


def _current_device() -> Any:
    from triton.runtime.driver import driver  # type: ignore[import-untyped]

    return driver.active.get_current_device()


def _is_compiled(jit_function: Any, device: Any, key: str) -> bool:
    try:
        return key in jit_function.device_caches[device][0]
    except (AttributeError, IndexError, KeyError, TypeError):
        return False


def _resolve_jit_function(module_name: str, qualname: str) -> Any:
    """Return the ``JITFunction`` defined at ``module_name.qualname``, if any."""
    if not module_name.startswith(_KERNEL_MODULE_PREFIX) or "<locals>" in qualname:
        return None
    from triton.runtime.jit import JITFunction  # type: ignore[import-untyped]

    try:
        target: Any = importlib.import_module(module_name)
        for attribute in qualname.split("."):
            target = getattr(target, attribute)
    except Exception:
        return None
    # triton.autotune and triton.heuristics keep the wrapped kernel in ``fn``.
    for _ in range(4):
        if isinstance(target, JITFunction):
            return target
        target = getattr(target, "fn", None)
    return None
