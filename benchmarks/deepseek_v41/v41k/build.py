# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the V4.1 CUDA extensions (SM70) into ./build.

v41k.so: the runtime loads it from VLLM_V41K_LIB, defaulting to this build
directory. v41sk.so: the skinny MXFP4 grouped MoE kernels with the V4.1 split-K
cases (native-weights target experts), loaded from the same directory.
"""

import os

from torch.utils import cpp_extension as ce

here = os.path.dirname(os.path.abspath(__file__))
src = os.path.abspath(os.path.join(here, "..", "..", ".."))
os.makedirs(os.path.join(here, "build"), exist_ok=True)


def generate_v41q8():
    """The QPN8 FP8 kernels with one scale per 32 K values (V4.1 dense)."""
    with open(os.path.join(src, "csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu")) as f:
        lines = f.read().split("\n")
    end = lines.index("}  // namespace")  # the anonymous kernel namespace
    body = "\n".join(lines[: end + 1])
    # Every scale-group index (16-value groups, 8 per 128-K scale) becomes a
    # 32-K scale index. Exact counts: a changed kernel must be re-reviewed.
    if body.count("group >> 3") != 4 or body.count("group_begin >> 3") != 1:
        raise RuntimeError("QPN8 source changed: expected five scale indices")
    body = body.replace("group >> 3", "group >> 1")
    body = body.replace("group_begin >> 3", "group_begin >> 1")
    with open(os.path.join(here, "v41q8_entry.cuh")) as f:
        entry = f.read()
    out = os.path.join(here, "build", "v41q8.cu")
    with open(out, "w") as f:
        f.write(body + "\n\n" + entry)
    return out


sources = {
    "v41k": os.path.join(here, "v41k.cu"),
    "v41sk": os.path.join(here, "v41sk.cu"),
}
sources["v41q8"] = generate_v41q8()
for name in ("v41k", "v41sk", "v41q8"):
    ce.load(
        name=name,
        sources=[sources[name]],
        extra_include_paths=[
            src + "/csrc",
            src + "/csrc/sm70_turbomind/lmdeploy",
            src + "/csrc/sm70_turbomind/ops",
        ],
        extra_cuda_cflags=[
            "-O3",
            "-gencode",
            "arch=compute_70,code=sm_70",
            "--expt-relaxed-constexpr",
            "-std=c++20",
            "-DENABLE_SM70_TURBOMIND=1",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        extra_cflags=["-O3", "-std=c++20"],
        is_python_module=False,
        build_directory=os.path.join(here, "build"),
        verbose=False,
    )
    print("built", os.path.join(here, "build", name + ".so"))
