# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 ratio-2 pooling with bounded speculative partial-group snapshots."""

import torch
from torch import nn

from vllm.model_executor.layers.quantization.gguf_shared_protocol import _fail_if
from vllm.triton_utils import tl, tldevice, triton


@triton.jit(do_not_specialize=["N"])
def _begin(positions, committed_position, metadata, N, BLOCK: tl.constexpr):
    start = tl.load(positions).to(tl.int64)
    previous = tl.load(committed_position)
    pending = tl.load(metadata + 2)
    _fail_if(
        (start < 0)
        | (start + N > 1048576)
        | ((start != 0) & ((start != previous) | (pending != 0)))
    )
    i = tl.arange(0, BLOCK)
    p = tl.load(positions + i, i < N, 0)
    _fail_if(tl.sum(((i < N) & (p != start + i)).to(tl.int32), 0))
    tl.store(metadata, start)
    tl.store(metadata + 1, N)
    tl.store(metadata + 2, 1)


@triton.jit(do_not_specialize=["N"])
def _pool_c2(
    kv,
    score,
    positions,
    partial,
    candidates,
    norm,
    out,
    complete,
    N,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    start = tl.load(positions).to(tl.int64)
    valid = d < D
    current = tl.load(kv + t * D + d, valid, 0)
    current_score = tl.load(score + t * D + d, valid, 0)
    previous = tl.load(kv + (t - 1) * D + d, (t > 0) & valid, 0)
    previous_score = tl.load(score + (t - 1) * D + d, (t > 0) & valid, 0)
    previous = tl.where(t > 0, previous, tl.load(partial + d, valid, 0))
    previous_score = tl.where(t > 0, previous_score, tl.load(partial + D + d, valid, 0))
    done = (start + t + 1) % 2 == 0
    maximum = tl.maximum(previous_score, current_score)
    a = tldevice.exp(previous_score - maximum)
    b = tldevice.exp(current_score - maximum)
    # Preserve the official FP32 softmax/product/sum, then FP16 store before
    # RMSNorm. The two projections have not been rounded to FP16 beforehand.
    value = (
        (tl.div_rn(a, a + b) * previous + tl.div_rn(b, a + b) * current)
        .to(tl.float16)
        .to(tl.float32)
    )
    value = tl.where(done, value, 0)
    variance = tl.sum(tl.where(valid, value * value, 0), 0) / D
    value = value * tl.rsqrt(variance + EPS) * tl.load(norm + d, valid, 0)
    tl.store(out + t * D + d, value, valid)
    tl.store(complete + t, done)
    # Small verification batches retain every prefix. Prefill can only
    # commit its full input and therefore needs just its final partial.
    if N <= 6 or t == N - 1:
        row = tl.where(N <= 6, t + 1, 6)
        tl.store(candidates + row * 2 * D + d, tl.where(done, 0, current), valid)
        tl.store(
            candidates + (row * 2 + 1) * D + d, tl.where(done, 0, current_score), valid
        )
    if t == 0:
        old = tl.load(partial + d, valid, 0)
        old_score = tl.load(partial + D + d, valid, 0)
        tl.store(candidates + d, tl.where(start == 0, 0, old), valid)
        tl.store(candidates + D + d, tl.where(start == 0, 0, old_score), valid)


@triton.jit
def _commit_c2(
    partial,
    candidates,
    metadata,
    position,
    selector,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    count = tl.load(selector).to(tl.int64)
    length, start = tl.load(metadata + 1), tl.load(metadata)
    _fail_if(
        (tl.load(metadata + 2) != 1)
        | (count < 0)
        | (count > length)
        | ((length > 6) & (count != length))
    )
    row = tl.where(length <= 6, count, 6)
    d = tl.arange(0, BLOCK)
    for part in tl.static_range(2):
        value = tl.load(candidates + (row * 2 + part) * D + d, d < D, 0)
        tl.store(partial + part * D + d, value, d < D)
    tl.store(position, start + count)
    tl.store(metadata + 2, 0)


class V41C2State(nn.Module):
    """A ratio-2 source's incomplete group and seven candidate boundaries."""

    def __init__(self, norm_weight, eps=1e-20):
        super().__init__()
        if (
            not norm_weight.is_cuda
            or norm_weight.dtype != torch.float32
            or norm_weight.shape != (512,)
        ):
            raise ValueError("V4.1 C2 normalization requires FP32 width 512 on CUDA")
        self.register_buffer("norm_weight", norm_weight, persistent=False)
        device = norm_weight.device
        self.register_buffer(
            "partial",
            torch.zeros(2, 512, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "candidates",
            torch.empty(7, 2, 512, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "metadata",
            torch.zeros(3, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "position",
            torch.zeros(1, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.eps = eps

    def forward(self, kv, score, positions):
        n = kv.shape[0]
        if (
            kv.ndim != 2
            or kv.shape != score.shape
            or kv.shape[1] != 512
            or not 1 <= n <= 16384
            or positions.shape != (n,)
            or kv.dtype != torch.float32
            or score.dtype != torch.float32
            or positions.dtype not in (torch.int32, torch.int64)
            or any(
                t.device != self.partial.device or not t.is_contiguous()
                for t in (kv, score, positions)
            )
        ):
            raise ValueError("V4.1 C2 requires unrounded FP32 projections/scores")
        result = torch.empty_like(kv, dtype=torch.float16)
        complete = torch.empty(n, dtype=torch.bool, device=kv.device)
        _begin[(1,)](
            positions,
            self.position,
            self.metadata,
            N=n,
            BLOCK=triton.next_power_of_2(n),
        )
        _pool_c2[(n,)](
            kv,
            score,
            positions,
            self.partial,
            self.candidates,
            self.norm_weight,
            result,
            complete,
            N=n,
            D=512,
            EPS=self.eps,
            BLOCK=512,
            enable_fp_fusion=False,
        )
        return result, complete

    def commit(self, accepted_inputs):
        if (
            accepted_inputs.numel() != 1
            or accepted_inputs.dtype not in (torch.int32, torch.int64)
            or accepted_inputs.device != self.partial.device
        ):
            raise ValueError("C2 commit selector must be one CUDA integer")
        _commit_c2[(1,)](
            self.partial,
            self.candidates,
            self.metadata,
            self.position,
            accepted_inputs,
            D=512,
            BLOCK=512,
        )


# ---- VLLM_V41_BATCH: one C2 state per request slot -------------------------
# Same pooling, snapshots and commit rule as V41C2State for a batch: token t
# of request row rows[t] (first token bases[t]) uses slot slots[row].


@triton.jit
def _bbegin(
    positions,
    mask,
    query_start_loc,
    slots,
    committed,
    metadata,
    scratch,
    errors,
    TRAP: tl.constexpr,
):
    r = tl.program_id(0)
    first = tl.load(query_start_loc + r).to(tl.int64)
    n = tl.load(query_start_loc + r + 1).to(tl.int64) - first
    slot = tl.load(slots + r).to(tl.int64)
    # Triton: nested ifs on tensors (no short-circuit `and`).
    if (n > 0) & (slot != scratch):  # noqa: SIM102 (scratch = graph padding)
        if tl.load(mask + first):
            start = tl.load(positions + first).to(tl.int64)
            previous = tl.load(committed + slot)
            pending = tl.load(metadata + slot * 3 + 2)
            bad = (
                (start < 0)
                | (start + n > 1048576)
                | ((start != 0) & ((start != previous) | (pending != 0)))
            )
            if TRAP:
                _fail_if(bad)
                for i in range(0, n):
                    _fail_if(tl.load(positions + first + i) != start + i)
            else:
                gap = 0
                for b in range(0, n, 512):  # vectorized: prefill rows are long
                    t = b + tl.arange(0, 512)
                    pos = tl.load(positions + first + t, mask=t < n, other=0)
                    gap += tl.sum(((t < n) & (pos != start + t)).to(tl.int32), 0)
                if gap:  # noqa: SIM102 (the CAS must not run for good rows)
                    if tl.atomic_cas(errors, 0, 5) == 0:
                        tl.store(errors + 1, slot.to(tl.int32))
                        tl.store(errors + 2, start.to(tl.int32))
                        tl.store(errors + 5, n.to(tl.int32))
                        tl.store(errors + 6, r)
            if (not TRAP) & bad:  # noqa: SIM102 (record the first failure)
                if tl.atomic_cas(errors, 0, 3) == 0:
                    tl.store(errors + 1, slot.to(tl.int32))
                    tl.store(errors + 2, start.to(tl.int32))
                    tl.store(errors + 3, previous.to(tl.int32))
                    tl.store(errors + 4, pending.to(tl.int32))
                    tl.store(errors + 5, n.to(tl.int32))
                    tl.store(errors + 6, r)
            tl.store(metadata + slot * 3, start)
            tl.store(metadata + slot * 3 + 1, n)
            tl.store(metadata + slot * 3 + 2, 1)


@triton.jit
def _bpool_c2(
    kv,
    score,
    positions,
    mask,
    rows,
    bases,
    query_start_loc,
    slots,
    partial,
    candidates,
    norm,
    out,
    complete,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    valid = d < D
    live = tl.load(mask + t)
    row = tl.load(rows + t)
    base = tl.load(bases + t)
    slot = tl.load(slots + row).to(tl.int64)
    n = tl.load(query_start_loc + row + 1) - base
    local = t - base
    start = tl.load(positions + base).to(tl.int64)
    own = partial + slot * 2 * D
    current = tl.load(kv + t * D + d, valid, 0)
    current_score = tl.load(score + t * D + d, valid, 0)
    previous = tl.load(kv + (t - 1) * D + d, (local > 0) & valid, 0)
    previous_score = tl.load(score + (t - 1) * D + d, (local > 0) & valid, 0)
    previous = tl.where(local > 0, previous, tl.load(own + d, valid, 0))
    previous_score = tl.where(local > 0, previous_score, tl.load(own + D + d, valid, 0))
    done = (start + local + 1) % 2 == 0
    maximum = tl.maximum(previous_score, current_score)
    a = tldevice.exp(previous_score - maximum)
    b = tldevice.exp(current_score - maximum)
    value = (
        (tl.div_rn(a, a + b) * previous + tl.div_rn(b, a + b) * current)
        .to(tl.float16)
        .to(tl.float32)
    )
    value = tl.where(done, value, 0)
    variance = tl.sum(tl.where(valid, value * value, 0), 0) / D
    value = value * tl.rsqrt(variance + EPS) * tl.load(norm + d, valid, 0)
    tl.store(out + t * D + d, value, valid)
    tl.store(complete + t, done)
    snapshots = candidates + slot * 14 * D
    if live & ((n <= 6) | (local == n - 1)):
        crow = tl.where(n <= 6, local + 1, 6)
        tl.store(snapshots + crow * 2 * D + d, tl.where(done, 0, current), valid)
        tl.store(
            snapshots + (crow * 2 + 1) * D + d,
            tl.where(done, 0, current_score),
            valid,
        )
    if live & (local == 0):
        old = tl.load(own + d, valid, 0)
        old_score = tl.load(own + D + d, valid, 0)
        tl.store(snapshots + d, tl.where(start == 0, 0, old), valid)
        tl.store(snapshots + D + d, tl.where(start == 0, 0, old_score), valid)


@triton.jit
def _bcommit_c2(
    partial,
    candidates,
    metadata,
    position,
    selector,
    slots,
    errors,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TRAP: tl.constexpr,
):
    i = tl.program_id(0)
    count = tl.load(selector + i).to(tl.int64)
    if count >= 0:
        slot = tl.load(slots + i).to(tl.int64)
        length = tl.load(metadata + slot * 3 + 1)
        start = tl.load(metadata + slot * 3)
        pending = tl.load(metadata + slot * 3 + 2)
        bad = (pending != 1) | (count > length) | ((length > 6) & (count != length))
        if TRAP:
            _fail_if(bad)
        elif bad:  # VLLM_V41_BATCH_DEBUG: record the first failure, skip
            if tl.atomic_cas(errors, 0, 4) == 0:
                tl.store(errors + 1, slot.to(tl.int32))
                tl.store(errors + 2, start.to(tl.int32))
                tl.store(errors + 3, count.to(tl.int32))
                tl.store(errors + 4, pending.to(tl.int32))
                tl.store(errors + 5, length.to(tl.int32))
                tl.store(errors + 6, i)
            return
        crow = tl.where(length <= 6, count, 6)
        d = tl.arange(0, BLOCK)
        for part in tl.static_range(2):
            value = tl.load(
                candidates + slot * 14 * D + (crow * 2 + part) * D + d, d < D, 0
            )
            tl.store(partial + slot * 2 * D + part * D + d, value, d < D)
        tl.store(position + slot, start + count)
        tl.store(metadata + slot * 3 + 2, 0)


class V41C2StateBatch(nn.Module):
    """V41C2State with one partial group and seven snapshots per request slot."""

    def __init__(self, single, slots):
        super().__init__()
        device = single.partial.device
        self.norm_weight, self.eps = single.norm_weight, single.eps
        self.partial = torch.zeros(slots, 2, 512, dtype=torch.float32, device=device)
        self.candidates = torch.empty(
            slots, 7, 2, 512, dtype=torch.float32, device=device
        )
        self.metadata = torch.zeros(slots, 3, dtype=torch.int64, device=device)
        self.position = torch.zeros(slots, dtype=torch.int64, device=device)
        self.query_start_loc = torch.zeros(slots + 1, dtype=torch.int32, device=device)
        self.forward_slots = torch.zeros(slots, dtype=torch.int32, device=device)
        self.errors = torch.zeros(8, dtype=torch.int32, device=device)

    def check_errors(self):
        """VLLM_V41_BATCH_DEBUG: the first recorded C2 begin/commit failure."""
        if not hasattr(self, "_errors_host"):
            self._errors_host = torch.zeros(8, dtype=torch.int32).pin_memory()
            self._errors_event = torch.cuda.Event()
            self._errors_event.record()
        if not self._errors_event.query():
            return
        v = self._errors_host.tolist()
        self._errors_host.copy_(self.errors, non_blocking=True)
        self._errors_event.record()
        if v[0] == 3:
            raise RuntimeError(
                f"V4.1 C2 debug (begin): row {v[6]} slot {v[1]} starts at {v[2]} "
                f"({v[5]} tokens), committed {v[3]}, pending {v[4]}"
            )
        if v[0] == 5:
            raise RuntimeError(
                f"V4.1 C2 debug (begin): row {v[6]} slot {v[1]} positions are not "
                f"contiguous from {v[2]} ({v[5]} tokens)"
            )
        if v[0] == 4:
            raise RuntimeError(
                f"V4.1 C2 debug (commit): entry {v[6]} slot {v[1]} pending forward "
                f"starts at {v[2]} with {v[5]} tokens, pending flag {v[4]}, "
                f"commit count {v[3]}"
            )

    def reset(self, slot):
        self.partial[slot].zero_()
        self.metadata[slot].zero_()
        self.position[slot] = 0

    def forward(self, kv, score, positions, token_mask, layout, slots):
        n, reqs = kv.shape[0], layout.num_reqs
        if kv.shape != (n, 512) or kv.dtype != torch.float32 or score.shape != kv.shape:
            raise ValueError("V4.1 C2 requires unrounded FP32 projections/scores")
        result = torch.empty_like(kv, dtype=torch.float16)
        complete = torch.empty(n, dtype=torch.bool, device=kv.device)
        self.query_start_loc[: reqs + 1].copy_(layout.query_start_loc[: reqs + 1])
        self.forward_slots[:reqs].copy_(slots[:reqs])
        _bbegin[(reqs,)](
            positions,
            token_mask,
            self.query_start_loc,
            self.forward_slots,
            self.position,
            self.metadata,
            self.position.shape[0] - 1,
            self.errors,
            TRAP=False,  # recorded (check_errors), see deepseek_v41_engram
        )
        _bpool_c2[(n,)](
            kv,
            score,
            positions,
            token_mask,
            layout.rows,
            layout.bases,
            self.query_start_loc,
            self.forward_slots,
            self.partial,
            self.candidates,
            self.norm_weight,
            result,
            complete,
            D=512,
            EPS=self.eps,
            BLOCK=512,
            enable_fp_fusion=False,
        )
        return result, complete

    def commit(self, selector, slots):
        """selector[i] accepted inputs of the pending forward of slots[i]."""
        _bcommit_c2[(selector.numel(),)](
            self.partial,
            self.candidates,
            self.metadata,
            self.position,
            selector,
            slots,
            self.errors,
            D=512,
            BLOCK=512,
            TRAP=False,  # recorded (check_errors), see deepseek_v41_engram
        )
