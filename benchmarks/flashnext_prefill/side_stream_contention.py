# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Micro-test: does side-stream work slow concurrent GEMM/elementwise kernels?

Compares the cold-expert transcode (zero-copy pinned host reads), the same
transcode from device memory and a copy-engine H2D copy.

Run on an idle GPU (service stopped): python side_stream_contention.py
Env D=<gpu>, NX=<working-set copies>.
"""

import os
import time

import torch
import vllm._C  # noqa: F401

from vllm.model_executor.layers.quantization.gguf_expert_cache import _native

torch.manual_seed(0)

dev = torch.device("cuda:" + os.environ.get("D", "0"))
torch.accelerator.set_device_index(dev)
M, K, N = 8160, 10240, 88
XS = [
    torch.empty(M, K, device=dev, dtype=torch.half).normal_()
    for _ in range(int(os.environ.get("NX", "1")))
]
ES = [
    torch.randn(M * 2560, device=dev).half()
    for _ in range(int(os.environ.get("NX", "1")))
]
rot = [0]


def nxt(lst):
    rot[0] += 1
    return lst[rot[0] % len(lst)]


x = XS[0]
w = (torch.randn(N, K, device=dev) * 0.02).half()
a = torch.randn(M, 4096, device=dev).half()
b = torch.randn(4096, 4096, device=dev).half()
e = torch.randn(M * 2560, device=dev).half()
# IQ3_S gate|up transcode geometry: 64 experts, 512+512 rows, K=4096
E, R, KK = 64, 512, 4096
stride = 110 * KK // 256
one = 2 * E * R * stride
SPAN = 18  # cycle through ~2 GB like the service's arena
host = torch.randint(0, 255, (one,), dtype=torch.uint8).pin_memory()
devsrc = host.to(dev)
big_pin = torch.empty(SPAN * one, dtype=torch.uint8).pin_memory()
big_reg = torch.empty(SPAN * one + 4096, dtype=torch.uint8)
off = (-big_reg.data_ptr()) % 4096
big_reg = big_reg[off : off + SPAN * one]
big_pin.copy_(host.repeat(SPAN))
big_reg.copy_(host.repeat(SPAN))
_native().host_register(big_reg.data_ptr(), big_reg.numel())
reg_view = _native().mapped_view(big_reg, dev.index)
dst_w = torch.empty(E * (2 << 20), dtype=torch.uint8, device=dev)
dst_s = torch.empty_like(dst_w)
err = torch.zeros(1, dtype=torch.int32, device=dev)
experts = torch.arange(E, dtype=torch.int32, device=dev)
count = torch.tensor([E], dtype=torch.int32, device=dev)


def srcs(base):
    g = base + torch.arange(E, dtype=torch.int64) * 2 * R * stride
    return g.to(dev), (g + R * stride).to(dev)


zc = srcs(host.data_ptr())
dv = srcs(devsrc.data_ptr())
pin_c = [srcs(big_pin.data_ptr() + j * one) for j in range(SPAN)]
reg_c = [srcs(reg_view.data_ptr() + j * one) for j in range(SPAN)]
cyc = [0]


def tcc(lst, ctas):
    cyc[0] += 1
    tc(lst[cyc[0] % SPAN], ctas)


hbuf = torch.empty(64 << 20, dtype=torch.uint8).pin_memory()
dbuf = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
side = torch.cuda.Stream(dev, priority=-1)


def tc(src, ctas):
    torch.ops._C.gguf_prefill_transcode_sm70(
        dst_w,
        dst_s,
        2 << 20,
        2 << 20,
        src[0],
        src[1],
        experts,
        count,
        ctas,
        21,
        2 * R,
        KK,
        0,
        stride,
        err,
    )


bg = {
    "idle": None,
    "registered 2GB cycle": lambda: tcc(reg_c, 8),
    "device tc 8 CTAs": lambda: tc(dv, 8),
    "H2D memcpy (DMA)": lambda: dbuf.copy_(hbuf, non_blocking=True),
}
fg = {
    "HC down mm": lambda: torch.mm(nxt(XS), w.T),
    "4096 GEMM": lambda: torch.mm(a, b),
    "elementwise": lambda: nxt(ES).mul_(1.0001),
}


def timeit(fn, it=30):
    s0, s1 = torch.cuda.Event(True), torch.cuda.Event(True)
    fn()
    s0.record()
    for _ in range(it):
        fn()
    s1.record()
    s1.synchronize()
    return s0.elapsed_time(s1) / it * 1e3


for bname, bfn in bg.items():
    if bfn:
        with torch.cuda.stream(side):
            s0, s1 = torch.cuda.Event(True), torch.cuda.Event(True)
            s0.record()
            bfn()
            s1.record()
        torch.accelerator.synchronize()
        one = s0.elapsed_time(s1)
        with torch.cuda.stream(side):
            for _ in range(int(6000 / max(one, 0.1)) + 5):
                bfn()  # ~2 s of background work
        time.sleep(1.0)
    res = {f: timeit(fn) for f, fn in fg.items()}
    busy = side.query() is False
    torch.accelerator.synchronize()
    print(
        f"{bname:22s} "
        + "  ".join(f"{f} {v:7.0f} us" for f, v in res.items())
        + (f"  (bg one={one:.2f} ms, still busy={busy})" if bfn else ""),
        flush=True,
    )
