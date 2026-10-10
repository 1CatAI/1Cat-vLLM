# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load pinned QSA operator sources without constructing either serving engine.

This is an operator comparison, not installed-engine dispatch validation.
Python function bodies and device kernels are loaded from the supplied trees.
Only package imports, capability/configuration injection and binding glue are
provided here. Both arms share Torch, CUDA, logical inputs and timing code.
"""

import __future__

import ast
import functools
import math
import os
import sys
import types
from pathlib import Path

import regex as re
import torch
from torch.utils.cpp_extension import load


class QuietLogger:
    def __getattr__(self, name):
        return lambda *a, **kw: None


class Sm70:
    def is_device_capability(self, version):
        return version == 70

    def has_device_capability(self, version):
        return version <= 70

    def is_device_capability_family(self, version):
        return version == 70


class SparseDefaults:
    # c39f53ab: Sm70SparseConfig legacy defaults + SM70_QSA_TUNING.
    values = dict(
        qsa_indexer_cublas=True,
        qsa_mtp_topk=True,
        qsa_score_tile_mb=64,
        qsa_cublas_min_rows=512,
        qsa_cublas_min_score_elements=1024**2,
        qsa_xqa_page4=True,
        qsa_xqa_page4_min_rows=64,
        qsa_grouped_page4=True,
        qsa_grouped_pad_fix=True,
    )

    def value(self, name):
        return self.values[name]


def write_changed(path, content):
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def source_module(name, path, injected=None, names=None):
    """Keep original function source/locations; omit package imports only."""
    path = Path(path)
    tree = ast.parse(path.read_text(), filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if names is not None and getattr(node, "name", None) not in names:
            continue
        body.append(node)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__dict__.update(
        torch=torch,
        triton=triton,
        tl=tl,
        math=math,
        os=os,
        re=re,
        Path=Path,
        functools=functools,
        lru_cache=functools.lru_cache,
        HAS_TRITON=True,
        logger=QuietLogger(),
        init_logger=lambda _: QuietLogger(),
        current_platform=Sm70(),
    )
    module.__dict__.update(injected or {})
    sys.modules[name] = module
    exec(
        compile(
            ast.Module(body=body, type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        module.__dict__,
    )
    return module


def build_native(onecat, sglang, cache):
    cache.mkdir(parents=True, exist_ok=True)
    flags = ["-O3", "--expt-relaxed-constexpr"]

    def build(name, sources, includes=()):
        dest = cache / name
        dest.mkdir(exist_ok=True)
        return load(
            name=name,
            sources=[str(p) for p in sources],
            build_directory=str(dest),
            extra_include_paths=list(map(str, includes)),
            extra_cuda_cflags=flags,
            verbose=True,
        )

    indexer = build(
        "qsa_ab_onecat_indexer",
        [onecat / "csrc/sm70_turbomind/ops/qsa_indexer_sm70.cu"],
    )
    sgl_decode = build(
        "qsa_ab_sgl_decode",
        [sglang / "python/sglang/kernels/jit/csrc/sm70_longctx_decode.cu"],
    )
    build(
        "qsa_ab_onecat_device",
        [onecat / "csrc/sm70_turbomind/ops/qsa_device_history_sm70.cu"],
    )
    # Binding only: the unchanged header is the production vLLM selector.
    bridge = cache / "onecat_topk.cu"
    write_changed(
        bridge,
        """#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "qsa_lexicographic_topk.cuh"
void run(torch::Tensor x, torch::Tensor n, torch::Tensor y, int64_t k, bool batch) {
  TORCH_CHECK(k == 512, "comparison supports topk512");
  const c10::cuda::CUDAGuard guard(x.device());
  vllm::qsa::launch_qsa_lexicographic_topk<512>(x.data_ptr<float>(), n.data_ptr<int>(),
    y.data_ptr<int>(), x.size(0), x.size(1), x.stride(0),
    at::cuda::getCurrentCUDAStream(), batch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
TORCH_LIBRARY_FRAGMENT(_C, m) {
 m.def("qsa_lexicographic_topk(Tensor x, Tensor n, Tensor(a!) y, "
       "int k, bool batch=False) -> ()");
 m.impl("qsa_lexicographic_topk", torch::kCUDA, &run);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
""",
    )
    topk = build("qsa_ab_onecat_topk", [bridge], [onecat / "csrc"])
    # tvm-ffi bindings use the original SGLang headers and wrapper functions.
    from tvm_ffi.cpp import load_inline

    include = sglang / "python/sglang/kernels/jit/include"
    csrc = sglang / "python/sglang/kernels/jit/csrc"
    (cache / "sgl_ffi").mkdir(exist_ok=True)
    native = load_inline(
        name="qsa_ab_sgl_ffi",
        build_directory=str(cache / "sgl_ffi"),
        cuda_sources="""#include "elementwise/fast_topk.cuh"
#include "elementwise/sm70_qsa_combine.cuh"
TVM_FFI_DLL_EXPORT_TYPED_FUNC(topk, (sglang::FastTopKKernel<512, false>::run));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(combine, (sglang::sm70_qsa_combine::combine));
""",
        extra_include_paths=[str(include), str(csrc)],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "-DSGL_CUDA_ARCH=700",
        ],
    )
    return indexer, sgl_decode, topk, native


def load_operators(onecat, sglang, cache):
    if os.environ.get("VLLM_SM70_QSA_TOPK_LIBRARY") or hasattr(
        torch.ops._C_qsa_sm70, "qsa_lexicographic_topk"
    ):
        raise RuntimeError("Run in a fresh process without a QSA top-k overlay")
    # Use the project's supported import shim from the pinned source tree.
    # This initializes the package but does not construct a serving engine.
    global tl, triton
    sys.path.insert(0, str(onecat))
    from vllm.triton_utils import tl, triton

    indexer, sgl_decode, topk, ffi = build_native(onecat, sglang, cache)
    pkg = types.ModuleType("qsa_ab_onecat")
    pkg.__path__ = []
    sys.modules[pkg.__name__] = pkg
    ops = onecat / "vllm/models/qwen4_exp/nvidia/ops"
    fp8 = source_module(
        "qsa_ab_fp8", onecat / "vllm/models/deepseek_v4/common/ops/fp8_software.py"
    )
    reader = source_module("qsa_ab_reader", ops / "host_kv_reader.py")
    shared = source_module("qsa_ab_onecat.qsa_shared_key", ops / "qsa_shared_key.py")
    shared.load_operator = lambda: True  # compiled above, same torch registration
    qsa = source_module(
        "qsa_ab_onecat.qsa",
        ops / "qsa.py",
        dict(
            pairwise=__import__("itertools").pairwise,
            sparse_policy=lambda: SparseDefaults(),
            fp8_e4m3fn_bits_to_fp32=fp8.fp8_e4m3fn_bits_to_fp32_bitcast,
            load_host_kv=reader.load_host_kv,
            workspace_cache=lambda name, standalone: standalone,
            retain_for_capture=lambda *a: None,
        ),
    )
    # Flash-V100 is built by its own unmodified setup.py in the source tree.
    sys.path.insert(0, str(onecat / "flash-attention-v100"))
    import flash_attn_v100_cuda

    iface = types.ModuleType("flash_attn_v100.flash_attn_interface")
    iface.flash_attn_v100_cuda = flash_attn_v100_cuda
    sys.modules[iface.__name__] = iface

    import tilelang
    import tilelang.language as T

    # Same TileLang adapter workaround applied by SGLang's _kernels_paged.py.
    from tilelang.jit.adapter.base import BaseKernelAdapter

    if not getattr(BaseKernelAdapter, "_legalize_result_idx_patched", False):
        original = BaseKernelAdapter._legalize_result_idx

        def legalize(self, indices):
            return original(
                self, list(indices) if isinstance(indices, list) else indices
            )

        BaseKernelAdapter._legalize_result_idx = legalize
        BaseKernelAdapter._legalize_result_idx_patched = True
    tile = sglang / "python/sglang/srt/layers/attention/tilelang_fa_v100"
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
    if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
        pass_configs[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = True
    else:
        pass_configs[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False
    combine_mod = source_module(
        "qsa_ab_tile_combine",
        tile / "_kernels_paged_decode.py",
        dict(tilelang=tilelang, T=T, pass_configs=pass_configs),
        names={"_decode_combine_kernel"},
    )
    expand_mod = source_module(
        "qsa_ab_sgl_expand",
        sglang / "python/sglang/srt/layers/attention/qsa/kernel.py",
        names={
            "_expand_qsa_block_indices_kernel",
            "triton_expand_qsa_block_indices",
            "expand_qsa_block_indices",
        },
    )

    def combine(partial, lse, lengths, width):
        m, splits, heads, dim = partial.shape
        if m <= 4:
            out = torch.empty(
                (m, heads, dim), dtype=partial.dtype, device=partial.device
            )
            ffi.combine(partial, lse, lengths, out, width, 32)
            return out
        return combine_mod._decode_combine_kernel(
            m, heads, dim, splits, 256, 32, selected_tokens=width
        )(partial, lse, lengths)

    def sgl_attention(q, k, v, table, req, indices, lengths):
        m = q.shape[0]
        splits = max(1, math.ceil(160 / m))
        partial = torch.empty((m, splits, 6, 256), dtype=q.dtype, device=q.device)
        lse = torch.empty((m, splits, 6), dtype=torch.float32, device=q.device)
        sgl_decode.sm70_qsa_decode(
            q, k, v, table, req, indices, lengths, splits, 32, 0.0625, partial, lse
        )
        return combine(partial, lse, lengths, indices.shape[1])

    def sgl_score(q, cache, table, lengths, columns):
        out = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
        sgl_decode.sm70_qsa_indexer_decode(
            q, cache, table, lengths, columns, math.sqrt(128), out
        )
        return out

    def sgl_topk(scores, lengths):
        starts = torch.zeros_like(lengths)
        selected = torch.empty(
            (scores.shape[0], 512), dtype=torch.int32, device=scores.device
        )
        ffi.topk(scores, starts, selected, lengths)
        return selected

    return types.SimpleNamespace(
        onecat=qsa,
        sgl_attention=sgl_attention,
        sgl_expand=expand_mod.expand_qsa_block_indices,
        sgl_score=sgl_score,
        sgl_topk=sgl_topk,
    )
