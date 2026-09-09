# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build only the native H3 extensions against the active Torch installation.

Use the regular 1Cat wheel build for a complete release. This helper leaves
existing vLLM binaries untouched and builds the three independent H3 modules.
"""

import hashlib
import json
import os
import shutil
import sysconfig
from pathlib import Path

from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parents[2]
output = root / "vllm"
records = []
cutlass_path = os.environ.get("VLLM_CUTLASS_SRC_DIR")
if not cutlass_path:
    raise RuntimeError("Set VLLM_CUTLASS_SRC_DIR to CUTLASS v4.4.2 source")
cutlass = Path(cutlass_path)
for name, source, includes, libraries in (
    ("_h3_w8a16_C", "csrc/sm70_turbomind/ops/h3_w8a16.cu", [], ["-lcublas"]),
    (
        "_h3_flashinfer_C",
        "flashinfer-sm70/csrc/h3_noncausal_sm70.cu",
        [str(root / "flashinfer-sm70/include")],
        [],
    ),
    (
        "_h3_flashattn_C",
        "flash-attention-v100/kernel/h3/forward.cu",
        [
            str(cutlass / "include"),
            str(cutlass / "examples/41_fused_multi_head_attention"),
        ],
        [],
    ),
):
    module = load(
        name=name,
        sources=[str(root / source)],
        extra_include_paths=includes,
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
        extra_ldflags=libraries,
        verbose=True,
    )
    target = output / (name + sysconfig.get_config_var("EXT_SUFFIX"))
    shutil.copy2(module.__file__, target)
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    records.append({"module": name, "source": source, "binary_sha256": digest})
print(json.dumps(records, indent=2))
