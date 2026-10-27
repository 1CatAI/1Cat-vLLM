# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed KV rows reproduce attention_qdq FP16 values bit for bit."""

import torch

from vllm.models.deepseek_v41.kv_pack import pack_rows, unpack_rows
from vllm.models.deepseek_v41.quant_rope import attention_qdq

torch.manual_seed(0)
ok = True
for scale in (1e-6, 1e-3, 0.05, 1.0, 30.0, 3000.0):
    for compressed in (True, False):
        dim = 512 if compressed else 128
        x = (torch.randn(4096, dim, device="cuda") * scale).half()
        x[0].zero_()
        x[1, :17] = 0  # zero groups
        ref = attention_qdq(
            x, fp4=True, group=16 if compressed else 32, e4m3_scale=compressed
        )
        got = unpack_rows(pack_rows(x, compressed), compressed)
        same = torch.equal(ref.view(torch.int16), got.view(torch.int16))
        ok &= same
        print(
            "scale %g compressed=%s bit-exact=%s mismatches=%d"
            % (
                scale,
                compressed,
                same,
                int((ref.view(torch.int16) != got.view(torch.int16)).sum()),
            )
        )
print("ALL OK" if ok else "FAILED")
