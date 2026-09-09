# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build a private wrapper around the frozen PyTorch CUDA key/value sorter.

This has no CUDA implementation of its own: the loaded PyTorch library owns
the final sort, including its unstable tie order. Build with an empty
CUDA_VISIBLE_DEVICES and an explicit CUDA_HOME; no GPU context is needed.
The wrapper is only for the paired SM70 sparse/dense top-k experiment.
"""

import argparse
import hashlib
import json
from pathlib import Path

SOURCE = r"""
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/all.h>
#include <torch/library.h>
#include <ATen/native/cuda/Sort.h>
#include <c10/cuda/CUDAGuard.h>

void sort_pairs(torch::Tensor values, torch::Tensor ids) {
  TORCH_CHECK(values.is_cuda() && ids.device() == values.device());
  TORCH_CHECK(values.scalar_type() == at::kFloat && ids.scalar_type() == at::kLong);
  TORCH_CHECK(values.dim() == 2 && ids.sizes() == values.sizes());
  TORCH_CHECK(values.is_contiguous() && ids.is_contiguous());
  const at::cuda::OptionalCUDAGuard guard(device_of(values));
  at::native::sortKeyValueInplace(values, ids, 1, true, false);
}
TORCH_LIBRARY_FRAGMENT(quasar_native_sort, m) {
  m.def("sort_pairs(Tensor(a!) values, Tensor(b!) ids) -> ()");
  m.impl("sort_pairs", torch::kCUDA, &sort_pairs);
}
"""


def main():
    import torch
    from torch.utils.cpp_extension import load

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if torch.__version__.split("+")[0] != "2.10.0":
        parser.error("This experiment requires the frozen PyTorch 2.10.0 sorter")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.output_dir / "native_sort.cpp"
    source.write_text(SOURCE.lstrip())
    torch_root = Path(torch.__file__).resolve().parent
    header = torch_root / "include/ATen/native/cuda/Sort.h"
    result = {
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "cuda_version": torch.version.cuda,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "native_header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
        "library_sha256": None,
        "serving_route_installed": False,
    }
    if args.build:
        build_dir = args.output_dir / "build"
        build_dir.mkdir(exist_ok=True)
        library = load(
            name="quasar_native_sort",
            sources=[str(source.resolve())],
            build_directory=str(build_dir.resolve()),
            extra_cflags=["-O3"],
            with_cuda=True,
            is_python_module=False,
            verbose=True,
        )
        result["library_sha256"] = hashlib.sha256(
            Path(library).read_bytes()
        ).hexdigest()
    (args.output_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
