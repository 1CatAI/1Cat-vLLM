# SPDX-License-Identifier: Apache-2.0
"""Request-grouped QSA scoring. Selection, expansion and attention are separate.

Only score columns below ``min(visible[row], columns)`` are defined; consumers
must mask using ``visible``. Invalid physical pages have a score of -inf.
Invalid/padded request IDs have non-positive visibility, preserving signed
division for negative positions. No persistent KV/state is changed.
"""

import math

import torch

from ._jit import prepare, provenance

_DEFAULT_DIVISOR = math.sqrt(128)


def qsa_score_reason(q, cache, table, requests, positions, lengths):
    """Static capability rejection without reading device metadata."""
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (7, 0):
        return "requires_SM70"
    if q.ndim != 3 or not 1 <= q.shape[0] <= 32 or q.shape[1:] != (4, 128):
        return "requires_M1_32_H4_D128"
    if cache.ndim != 4 or cache.shape[2:] != (1, 128) or not all(cache.shape[:2]):
        return "requires_paged_H1_D128_cache"
    if q.dtype != torch.float16 or cache.dtype != torch.float16:
        return "requires_FP16_query_and_cache"
    if table.ndim != 2 or not 1 <= table.shape[0] <= 32 or not table.shape[1]:
        return "requires_1_32_requests_and_nonempty_page_table"
    if any(t.device != q.device for t in (cache, table, requests, positions, lengths)):
        return "requires_same_device"
    if any(t.dtype != torch.int32 for t in (table, requests, lengths)):
        return "requires_int32_request_metadata"
    if positions.dtype not in (torch.int32, torch.int64):
        return "requires_integer_positions"
    if (
        requests.shape != (q.shape[0],)
        or positions.shape != requests.shape
        or lengths.shape != (table.shape[0],)
    ):
        return "invalid_metadata_extent"
    if any(not t.is_contiguous() for t in (requests, positions, lengths)):
        return "requires_contiguous_row_metadata"
    if any(s < 0 for t in (q, cache, table) for s in t.stride()):
        return "requires_nonnegative_strides"
    return None


@torch.library.custom_op(
    "flashinfer_sm70::qsa_scores",
    mutates_args=("scores", "visible", "grouped_rows", "counts"),
)
def _scores(
    scores: torch.Tensor,
    visible: torch.Tensor,
    grouped_rows: torch.Tensor,
    counts: torch.Tensor,
    q: torch.Tensor,
    cache: torch.Tensor,
    table: torch.Tensor,
    requests: torch.Tensor,
    positions: torch.Tensor,
    lengths: torch.Tensor,
    divisor: float,
    compress: int,
) -> None:
    prepare("qsa").qsa_scores(
        scores,
        visible,
        grouped_rows,
        counts,
        q,
        cache,
        table,
        requests,
        positions,
        lengths,
        divisor,
        compress,
    )


@_scores.register_fake
def _scores_fake(
    scores,
    visible,
    grouped_rows,
    counts,
    q,
    cache,
    table,
    requests,
    positions,
    lengths,
    divisor,
    compress,
):
    return None


class QsaScorePlan:
    """Bounded per-instance workspace; construct before capture, run without allocation.

    The plan is stream-ordered and cannot execute concurrently on two streams.
    Independent captures/instances must own separate plans. Input pointers are
    never cached; an eager call may bind a new cache or weights. Captured graphs
    retain their captured pointers and must be recaptured for address changes.
    """

    def __init__(self, max_rows, max_requests, max_columns, device):
        if not 1 <= max_rows <= 32 or not 1 <= max_requests <= 32 or max_columns < 0:
            raise ValueError(
                "QSA plan requires rows/requests in 1..32 and nonnegative columns"
            )
        device = torch.device(device)
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (7, 0):
            raise ValueError("QSA plan requires SM70")
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Create QsaScorePlan before CUDA graph capture")
            prepare("qsa")
        self.max_rows, self.max_requests, self.max_columns = (
            max_rows,
            max_requests,
            max_columns,
        )
        self.scores = torch.empty(
            (max_rows, max_columns), device=device, dtype=torch.float32
        )
        self.visible = torch.empty(max_rows, device=device, dtype=torch.int32)
        self.grouped_rows = torch.empty(
            (max_requests, max_rows), device=device, dtype=torch.int32
        )
        self.counts = torch.empty(max_requests, device=device, dtype=torch.int32)
        self.device = self.scores.device

    @property
    def workspace_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (self.scores, self.visible, self.grouped_rows, self.counts)
        )

    def run(
        self,
        q,
        cache,
        table,
        requests,
        positions,
        lengths,
        *,
        compress=4,
        columns=None,
        divisor=_DEFAULT_DIVISOR,
    ):
        columns = self.max_columns if columns is None else columns
        if q.ndim != 3 or q.shape[1:] != (4, 128) or q.shape[0] > self.max_rows:
            raise ValueError("QSA scorer requires M<=plan capacity, H4, D128")
        m = q.shape[0]
        if cache.ndim != 4 or cache.shape[2:] != (1, 128) or cache.shape[1] <= 0:
            raise ValueError("cache must be [pages, page_size, 1, 128]")
        if table.ndim != 2 or not 0 < table.shape[0] <= self.max_requests:
            raise ValueError("invalid request table or plan request capacity")
        if not 0 <= columns <= min(self.max_columns, table.shape[1] * cache.shape[1]):
            raise ValueError("score columns exceed plan/cache capacity")
        if compress <= 0 or not math.isfinite(divisor) or divisor <= 0:
            raise ValueError("compression and divisor must be positive and finite")
        for t in (q, cache, table, requests, positions, lengths):
            if t.device != self.device or any(s < 0 for s in t.stride()):
                raise ValueError(
                    "QSA tensors must share the plan device and use nonnegative strides"
                )
        if q.dtype != torch.float16 or cache.dtype != torch.float16:
            raise ValueError("QSA scorer requires FP16 query/cache")
        if any(t.dtype != torch.int32 for t in (table, requests, lengths)):
            raise ValueError("table, requests and lengths must be int32")
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("positions must be int32/int64")
        if (
            requests.shape != (m,)
            or positions.shape != (m,)
            or lengths.shape != (table.shape[0],)
        ):
            raise ValueError("QSA row/request metadata extent mismatch")
        if any(not t.is_contiguous() for t in (requests, positions, lengths)):
            raise ValueError("row metadata must be contiguous")
        scores, visible = self.scores[:m, :columns], self.visible[:m]
        _scores(
            scores,
            visible,
            self.grouped_rows,
            self.counts,
            q,
            cache,
            table,
            requests,
            positions,
            lengths,
            float(divisor),
            compress,
        )
        return scores, visible

    def provenance(self):
        return {
            **provenance("qsa"),
            "workspace_bytes": self.workspace_bytes,
            "status": "candidate; GPU correctness and complete-chain performance pending",
        }
