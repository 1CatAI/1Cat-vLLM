# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch

os.environ.setdefault(
    "VLLM_V41K_LIB", "/home/ymzx/ds41-dev/tools/v41k_g3/build/v41k.so"
)
import vllm.models.deepseek_v41.attention_ops as ops
from vllm.models.deepseek_v41.index_topk import index_topk
from vllm.models.deepseek_v41.kv_pack import pack_rows

torch.manual_seed(0)
H, T = 8, 6


def tm(fn, n=10):
    fn()
    torch.accelerator.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.accelerator.synchronize()
    return a.elapsed_time(b) / n


for rows, start in ((16384, 15000), (524288, 15000), (524288, 400000)):
    packed = pack_rows((torch.randn(rows, 128, device="cuda") * 0.5).half(), False)
    q = (torch.randn(T, H, 128, device="cuda") * 0.3).half()
    w = (torch.randn(T, H, device="cuda") * 0.2).half()
    pos = torch.arange(start, start + T, device="cuda")
    valid = torch.ones(T, dtype=torch.bool, device="cuda")
    scores = torch.empty(T, rows, dtype=torch.float16, device="cuda")
    ident = lambda x: x
    f = lambda: ops.select_packed_positions(q, w, packed, pos, valid, 1, ident)
    print(
        "rows %d start %d select_packed (decode, incl topk): %.3f ms"
        % (rows, start, tm(f))
    )
    res = torch.empty(T, 512, dtype=torch.int32, device="cuda")
    print(
        "   index_topk: %.3f ms"
        % tm(lambda: index_topk(res, scores, pos, valid, 1, 512))
    )
