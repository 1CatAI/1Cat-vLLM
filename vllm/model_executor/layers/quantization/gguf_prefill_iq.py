# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-forward, stream-owned compact canonical prefill GEMM context.

Imported only on an enabled M>6 prefill. The caller retains sole ownership of
its canonical bank and all activation/peer/accumulation transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


class Codec(IntEnum):
    IQ1_M = 29
    IQ2_XXS = 16
    IQ2_XS = 17


@dataclass(frozen=True)
class IndexedGateUp:
    output: torch.Tensor
    input: torch.Tensor
    offsets: torch.Tensor
    weights: torch.Tensor
    stats: torch.Tensor
    codec: Codec
    indices: torch.Tensor


@dataclass(frozen=True)
class ContiguousDown:
    output: torch.Tensor
    input: torch.Tensor
    offsets: torch.Tensor
    weights: torch.Tensor
    stats: torch.Tensor
    codec: Codec


@dataclass(frozen=True)
class TileSelection:
    # IQ1_M GU/DN, IQ2_XXS GU/DN, IQ2_XS GU/DN; immutable after load.
    tiles: tuple[int, ...]
    grid: int

    @classmethod
    def parse(cls, text: str, grid: str) -> TileSelection:
        parts = text.split(",")
        if len(parts) != 6 or any(
            p not in ("0", "1", "2", "3", "4", "5") for p in parts
        ):
            raise ValueError("VLLM_V41_PREFILL_IQ_TILES requires six tile IDs 0..5")
        if grid not in ("0", "80", "160"):
            raise ValueError("VLLM_V41_PREFILL_IQ_GRID requires 0, 80 or 160")
        return cls(tuple(int(p) for p in parts), int(grid))

    def tile(self, request: IndexedGateUp | ContiguousDown) -> int:
        index = (Codec.IQ1_M, Codec.IQ2_XXS, Codec.IQ2_XS).index(request.codec)
        return self.tiles[2 * index + int(isinstance(request, ContiguousDown))]


_selection = TileSelection.parse(
    envs.VLLM_V41_PREFILL_IQ_TILES, envs.VLLM_V41_PREFILL_IQ_GRID
)
_library = Path(__file__).with_name("v41prefmoe.so")
if not _library.is_file():
    raise FileNotFoundError(f"Enabled prefill IQ requires {_library}")
torch.ops.load_library(str(_library))
_announced = False


class IqPrefill:
    def __init__(self, tokens: int, max_routes: int, device: torch.device) -> None:
        if tokens <= 6 or not 0 < max_routes <= 6 * 16384:
            raise ValueError("Prefill IQ requires M>6 and at most 16384 routed tokens")
        self.tokens = tokens
        self.plan = torch.empty(
            1 + 4 * (max_routes // 32 + 32), dtype=torch.int32, device=device
        )
        self.stream = torch.cuda.current_stream(device)
        self.launches = 0

    def run(self, request: IndexedGateUp | ContiguousDown) -> None:
        global _announced
        if torch.cuda.current_stream(request.input.device) != self.stream:
            raise RuntimeError("Prefill IQ plan stream ownership changed")
        indices = request.indices if isinstance(request, IndexedGateUp) else None
        torch.ops.v41prefmoe.gemm(
            request.output,
            request.input,
            request.offsets,
            request.weights,
            request.stats,
            self.plan,
            int(request.codec),
            self.tokens,
            _selection.tile(request),
            _selection.grid,
            indices,
        )
        self.launches += 1
        if not _announced:
            logger.info(
                "V4.1 prefill IQ GEMM: enabled SM70 grouped canonical; launches=1"
            )
            _announced = True

    def gemm(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        offsets: torch.Tensor,
        weights: torch.Tensor,
        stats: torch.Tensor,
        family: int,
        decoder: int,
        n: int,
        experts: int,
        group: int,
        gated: bool,
        tune: bool,
        indices: torch.Tensor | None,
    ) -> None:
        codec = Codec(decoder)
        if (
            family != 0
            or gated
            or n != output.shape[1]
            or experts != offsets.numel() - 1
        ):
            raise ValueError(
                "Prefill IQ requires ungated canonical lattice descriptors"
            )
        if group != (32 if codec == Codec.IQ2_XXS else 16):
            raise ValueError("Prefill IQ canonical group size mismatch")
        if tune:
            raise ValueError("Candidate selection must be frozen before serving")
        if indices is None:
            self.run(ContiguousDown(output, input, offsets, weights, stats, codec))
        else:
            self.run(
                IndexedGateUp(output, input, offsets, weights, stats, codec, indices)
            )
