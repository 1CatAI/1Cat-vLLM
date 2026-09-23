# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import vllm.envs as envs
from vllm.compilation.piecewise_backend import PiecewiseBackend, RangeEntry
from vllm.config.utils import Range


def test_piecewise_autotune_configs_do_not_overwrite_each_other(tmp_path, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH", True)
    monkeypatch.setattr(envs, "VLLM_USE_AOT_COMPILE", True)
    monkeypatch.setattr(envs, "VLLM_DISABLE_COMPILE_CACHE", False)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "original"))
    monkeypatch.setattr(PiecewiseBackend, "_log_compile_start", lambda *_: None)
    monkeypatch.setattr(
        "vllm.compilation.piecewise_backend.get_fake_args_from_graph",
        lambda _: [],
    )

    def compile_graph(index, shape_range, best_config):
        def compile_stub(*args, **kwargs):
            cache_dir = os.environ["TORCHINDUCTOR_CACHE_DIR"]
            os.makedirs(cache_dir, exist_ok=True)
            with open(os.path.join(cache_dir, "same_kernel.best_config"), "w") as f:
                f.write(best_config)
            return lambda: None

        backend = PiecewiseBackend.__new__(PiecewiseBackend)
        backend.graph = object()
        backend.piecewise_compile_index = index
        backend.total_piecewise_compiles = 2
        backend.range_entries = {shape_range: RangeEntry(shape_range)}
        backend.vllm_backend = SimpleNamespace(
            compiler_manager=SimpleNamespace(
                cache_dir=str(tmp_path), compile=compile_stub
            ),
            inductor_config={},
            is_encoder=False,
        )
        backend.compilation_config = SimpleNamespace()
        backend.compile_all_ranges()
        return (
            tmp_path
            / "inductor_cache"
            / f"subgraph_{index}"
            / f"range_{shape_range.start}_{shape_range.end}"
            / "same_kernel.best_config"
        )

    first = compile_graph(0, Range(1, 8), "XBLOCK=64")
    second = compile_graph(1, Range(1, 8), "XBLOCK=256")
    third = compile_graph(0, Range(9, 16), "XBLOCK=128")

    assert [path.read_text() for path in (first, second, third)] == [
        "XBLOCK=64",
        "XBLOCK=256",
        "XBLOCK=128",
    ]
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "original")
