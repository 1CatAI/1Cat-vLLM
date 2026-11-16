"""Build v41prefmoe.so, the SM70 prefill IQ GEMM library of VLLM_V41_PREFILL_IQ_GEMM.

csrc/prefill_iq.cu reuses the OpenMMLab TurboMind / 1Cat SM70 GEMM headers in the
repository's csrc/. The library is loaded from next to
vllm/model_executor/layers/quantization/gguf_prefill_iq.py, so the result is copied
there.

  cd benchmarks/deepseek_v41/v41prefmoe && TORCH_CUDA_ARCH_LIST=7.0 python build.py
"""

import os
import shutil
from pathlib import Path

from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parent
src = root.parents[2]  # repository root
if os.environ.get("TORCH_CUDA_ARCH_LIST") != "7.0":
    raise RuntimeError(
        "set TORCH_CUDA_ARCH_LIST=7.0: this library is built for SM70 only"
    )
build = root / "build"
build.mkdir(exist_ok=True)
load(
    name="v41prefmoe",
    sources=[str(root / "csrc/prefill_iq.cu")],
    build_directory=str(build),
    extra_include_paths=[
        str(src / "csrc"),
        str(src / "csrc/sm70_turbomind/lmdeploy"),
    ],
    extra_cflags=["-O3", "-std=c++20", "-fvisibility=hidden"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "-UNDEBUG",
        "--expt-relaxed-constexpr",
        "-Xptxas=-v",
        "--compiler-options=-fvisibility=hidden",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ],
    with_cuda=True,
    is_python_module=False,
    verbose=True,
)
dest = src / "vllm/model_executor/layers/quantization/v41prefmoe.so"
shutil.copyfile(build / "v41prefmoe.so", dest)
print(f"v41prefmoe.so -> {dest}")
