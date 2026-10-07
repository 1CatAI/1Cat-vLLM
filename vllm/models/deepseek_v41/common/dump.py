# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Golden-dump hook (PORT_DESIGN §3.8; owner L-CORE). Off by default; zero cost when off.

With ``VLLM_DS41_DUMP_DIR`` set, every REAL forward step (dummy runs -- profiling, graph capture, warmup -- are
skipped via ForwardContext.is_dummy_run) whose index is in ``VLLM_DS41_DUMP_STEPS`` (default ``0-7``) writes

    $VLLM_DS41_DUMP_DIR/step{k:05d}/L{nn:02d}.safetensors   per dumped layer (``VLLM_DS41_DUMP_LAYERS``: "all",
                                                               "none" or ids/ranges like "0-3,14,20"; default all)
    $VLLM_DS41_DUMP_DIR/step{k:05d}/final.safetensors       final.stream_in, final.hc, final.h, logits (last stage)
    $VLLM_DS41_DUMP_DIR/step{k:05d}/meta_pp{r}.json         positions, token count, tensors written by PP stage r

so ``tools/ds41_ref/compare.py dump --golden CASE --dump $DIR/step00000`` compares a step op by op. Tensor names
are the §3.8 names. Each lane calls ``dump(layer, name, tensor, tp_dim=...)`` at its op; ``tp_dim`` all-gathers a
TP-sharded tensor along that dim (every TP rank must make the same calls), TP rank 0 of each stage writes.
Tensors keep their dtype (bf16 stream, fp16 activations, fp32 mixes) and only the first ``num_tokens`` rows of a
padded batch are kept. Requires eager execution (``--enforce-eager``): the D2H copies cannot run inside CUDA graphs.
Callers on hot paths guard with ``if dump.ENABLED:`` so a disabled hook costs one boolean test.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
from typing import Any

import torch

DIR_KNOB = "VLLM_DS41_DUMP_DIR"
LAYERS_KNOB = "VLLM_DS41_DUMP_LAYERS"
STEPS_KNOB = "VLLM_DS41_DUMP_STEPS"

_DIR = os.environ.get(DIR_KNOB) or None
ENABLED: bool = _DIR is not None
FINAL_FILE = "final.safetensors"


def parse_ids(spec: str, knob: str) -> frozenset[int] | None:
    """"all" -> None (everything), "none" -> empty, "0-3,14" -> {0,1,2,3,14}. Malformed specs raise."""
    spec = spec.strip().lower()
    if spec == "all":
        return None
    if spec in ("", "none"):
        return frozenset()
    ids: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        try:
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-", 1))
                if hi < lo:
                    raise ValueError
                ids.update(range(lo, hi + 1))
            else:
                ids.add(int(part))
        except ValueError as exc:
            raise ValueError(f"{knob}={spec!r}: expected 'all', 'none' or ids/ranges like '0-3,14'") from exc
    return frozenset(ids)


class _DumpState:
    def __init__(self) -> None:
        self.layers = parse_ids(os.environ.get(LAYERS_KNOB, "all"), LAYERS_KNOB)
        self.steps = parse_ids(os.environ.get(STEPS_KNOB, "0-7"), STEPS_KNOB)
        self.step = -1
        self.active = False
        self.num_tokens = 0
        self.files: dict[str, dict[str, torch.Tensor]] = {}
        self.meta: dict[str, Any] = {}
        self.pending_final: dict[str, torch.Tensor] = {}
        self.final_dir: Path | None = None

    def wants(self, layer: int | None) -> bool:
        return self.active and (layer is None or self.layers is None or layer in self.layers)


_STATE: _DumpState | None = _DumpState() if ENABLED else None


def _ranks() -> tuple[int, int, int]:
    from vllm.distributed import get_pp_group, get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size(), get_pp_group().rank_in_group


def _is_dummy() -> bool:
    from vllm.forward_context import get_forward_context, is_forward_context_available

    return is_forward_context_available() and get_forward_context().is_dummy_run


def _write(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    save_file(tensors, str(tmp))
    os.replace(tmp, path)


def _flush_final() -> None:
    state = _STATE
    if state is None or not state.pending_final:
        return
    assert state.final_dir is not None
    _write(state.final_dir / FINAL_FILE, state.pending_final)
    state.pending_final = {}


atexit.register(_flush_final)


def begin_step(positions: torch.Tensor, num_tokens: int) -> None:
    """Called by DeepseekV41Model.forward on entry (only when ENABLED)."""
    state = _STATE
    if state is None:
        return
    _flush_final()           # a previous last-stage step that never reached compute_logits
    if _is_dummy():
        state.active = False
        return
    state.step += 1
    state.active = state.steps is None or state.step in state.steps
    state.num_tokens = num_tokens
    state.files = {}
    tp_rank, _, pp_rank = _ranks()
    state.meta = {"step": state.step, "num_tokens": num_tokens, "pp_rank": pp_rank,
                  "positions": positions[:num_tokens].tolist() if state.active and tp_rank == 0 else None,
                  "tensors": {}}


def dump(layer: int | None, name: str, tensor: torch.Tensor, *, tp_dim: int | None = None) -> None:
    """Record ``tensor`` under the §3.8 ``name`` for ``layer`` (None = final.safetensors) in the current step."""
    state = _STATE
    if state is None or not state.wants(layer):
        return
    t = tensor.detach()
    if t.dim() > 0 and t.shape[0] >= state.num_tokens:
        t = t[: state.num_tokens]
    tp_rank, tp_size, _ = _ranks()
    if tp_dim is not None and tp_size > 1:
        from vllm.distributed import tensor_model_parallel_all_gather

        t = tensor_model_parallel_all_gather(t.contiguous(), dim=tp_dim)
    if tp_rank != 0:
        return
    cpu = t.to("cpu", copy=True).contiguous()
    fname = FINAL_FILE if layer is None else f"L{layer:02d}.safetensors"
    if layer is None:
        state.pending_final[name] = cpu
    else:
        state.files.setdefault(fname, {})[name] = cpu
    state.meta["tensors"][f"{fname}:{name}"] = {"shape": list(cpu.shape), "dtype": str(cpu.dtype)}


def end_forward() -> None:
    """Called by DeepseekV41Model.forward on exit: writes this stage's layer files (+ meta)."""
    state = _STATE
    if state is None or not state.active:
        return
    tp_rank, _, pp_rank = _ranks()
    if tp_rank != 0:
        return
    assert _DIR is not None
    step_dir = Path(_DIR) / f"step{state.step:05d}"
    for fname, tensors in state.files.items():
        _write(step_dir / fname, tensors)
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / f"meta_pp{pp_rank}.json").write_text(json.dumps(state.meta, indent=1))
    state.files = {}
    state.final_dir = step_dir


def end_logits() -> None:
    """Called by DeepseekV41ForCausalLM.compute_logits after dumping ``logits``: writes final.safetensors."""
    _flush_final()
