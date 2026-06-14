# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that compute logits from hidden_stats."""

import json
import os

import torch

from vllm import envs
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _parse_step_filter(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    steps: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"invalid top-token margin step range: {item}")
            steps.update(range(start, end + 1))
            continue
        step = int(item)
        if step < 0:
            raise ValueError(f"invalid top-token margin step: {item}")
        steps.add(step)
    return steps


def _parse_token_probe(raw: str | None) -> list[int]:
    if raw is None:
        return []
    token_ids: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        token_id = int(item)
        if token_id < 0:
            raise ValueError(f"invalid top-token margin probe token: {item}")
        token_ids.append(token_id)
    return token_ids


def _top_token_margin_dump_step(module: torch.nn.Module) -> int | None:
    out_dir = envs.VLLM_SM70_DUMP_TOP_TOKEN_MARGIN_DIR
    if not out_dir:
        return None
    enable_file = envs.VLLM_SM70_DUMP_TOP_TOKEN_MARGIN_ENABLE_FILE
    if enable_file and not os.path.exists(enable_file):
        return None
    step = int(getattr(module, "_sm70_top_token_margin_step", 0))
    module._sm70_top_token_margin_step = step + 1
    steps = _parse_step_filter(envs.VLLM_SM70_DUMP_TOP_TOKEN_MARGIN_STEPS)
    if steps is not None and step not in steps:
        return None
    reports = int(getattr(module, "_sm70_top_token_margin_reports", 0))
    max_reports = envs.VLLM_SM70_DUMP_TOP_TOKEN_MARGIN_MAX_REPORTS
    if max_reports > 0 and reports >= max_reports:
        return None
    module._sm70_top_token_margin_reports = reports + 1
    return step


def _write_top_token_margin_record(record: dict[str, object]) -> None:
    out_dir = envs.VLLM_SM70_DUMP_TOP_TOKEN_MARGIN_DIR
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    path = os.path.join(
        out_dir,
        f"top_token_margin_pid{os.getpid()}_cuda{device}.jsonl",
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _cuda_graph_capture_active() -> bool:
    if not torch.cuda.is_available():
        return False
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    if is_capturing is None:
        return False
    try:
        return bool(is_capturing())
    except RuntimeError:
        return False


def _maybe_sync_top1_all_gather(module: torch.nn.Module,
                                local_pair: torch.Tensor) -> None:
    raw_steps = envs.VLLM_SM70_SYNC_TOP1_ALLGATHER_STEPS
    if not raw_steps or not local_pair.is_cuda:
        return
    if torch.compiler.is_compiling() or _cuda_graph_capture_active():
        return

    step = int(getattr(module, "_sm70_top1_allgather_sync_step", 0)) + 1
    module._sm70_top1_allgather_sync_step = step
    target_steps = _parse_step_filter(raw_steps)
    if target_steps is not None and step not in target_steps:
        return

    mode = envs.VLLM_SM70_SYNC_TOP1_ALLGATHER_MODE.strip().lower()
    if mode == "d2h":
        _ = local_pair.detach().cpu()
    elif mode == "device":
        torch.cuda.synchronize(local_pair.device)
    else:
        torch.cuda.current_stream(local_pair.device).synchronize()


# --8<-- [start:logits_processor]
@PluggableLayer.register("logits_processor")
class LogitsProcessor(PluggableLayer):
    """Process logits and apply logits processors from sampling metadata.

    This layer does the following:
    1. Gather logits from model hidden_states.
    2. Scale logits if needed.
    3. Apply logits processors (if any).
    """

    # --8<-- [end:logits_processor]

    def __init__(
        self,
        vocab_size: int,
        org_vocab_size: int | None = None,
        scale: float = 1.0,
        logits_as_input: bool = False,
        soft_cap: float | None = None,
    ) -> None:
        """
        Args:
            scale: A scaling factor to apply to the logits.
        """
        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        # Whether the input is logits (default is hidden states).
        self.logits_as_input = logits_as_input
        # original vocabulary size (without LoRA).
        self.org_vocab_size = org_vocab_size or vocab_size
        # Soft cap the logits. Used in Gemma 2.
        self.soft_cap = soft_cap
        # Whether to use gather or all-gather to gather the logits.
        self.use_all_gather = current_platform.use_all_gather()

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.logits_as_input:
            logits = hidden_states
        else:
            # Get the logits for the next tokens.
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)
        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = torch.tanh(logits)
                logits = logits * self.soft_cap

            if self.scale != 1.0:
                logits *= self.scale
        return logits

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """gather/all-gather the logits tensor across model parallel group."""
        if self.use_all_gather:
            # Gather is not supported for some devices such as TPUs.
            # Use all-gather instead.
            # NOTE(woosuk): Here, the outputs of every device should not be None
            # because XLA requires strict SPMD among all devices. Every device
            # should execute the same operations after gathering the logits.
            logits = tensor_model_parallel_all_gather(logits)
        else:
            # None may be returned for rank > 0
            logits = tensor_model_parallel_gather(logits)
        return logits

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # Get the logits for the next tokens.
        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)

        # Gather logits for TP
        logits = self._gather_logits(logits)

        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    def get_top_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vocab-parallel argmax without all-gathering full logits.

        Each TP rank computes local argmax, then only the (value, index) pairs
        are gathered and reduced. Communication: O(batch * 2 * tp_size) vs
        O(batch * vocab_size).
        """
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local argmax reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )
        tp_size = get_tensor_model_parallel_world_size()

        local_top1 = lm_head.maybe_get_sm70_lm_head_top1(
            hidden_states, embedding_bias
        )
        if local_top1 is None:
            logits = lm_head.quant_method.apply(
                lm_head, hidden_states, bias=embedding_bias
            )
            if self.soft_cap is not None:
                logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
            if self.scale != 1.0:
                logits = logits * self.scale

            # Mask out padding entries beyond org_vocab_size on this shard.
            num_pad = lm_head.shard_indices.num_org_vocab_padding
            if num_pad > 0:
                logits[..., -num_pad:] = -float("inf")

            local_max_vals, local_max_indices = logits.max(dim=-1)

            # Convert shard-local indices to global vocab indices.
            vocab_start = lm_head.shard_indices.org_vocab_start_index
            global_indices = local_max_indices + vocab_start
        else:
            local_max_vals, global_indices = local_top1

        if tp_size == 1:
            self._maybe_dump_top_token_margin(
                lm_head, hidden_states, embedding_bias, global_indices
            )
            return global_indices

        # All-gather (value, index) pairs, then reduce to global argmax.
        # Use float32 to avoid bf16 precision loss on large vocab indices.
        local_pair = torch.stack(
            [local_max_vals.float(), global_indices.float()], dim=-1
        )
        _maybe_sync_top1_all_gather(self, local_pair)
        custom_top_tokens = self._maybe_custom_top1_argmax(local_pair)
        if custom_top_tokens is not None:
            self._maybe_dump_top_token_margin(
                lm_head, hidden_states, embedding_bias, custom_top_tokens
            )
            return custom_top_tokens

        # [batch, 2] -> [batch, 2 * tp_size]
        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        # [batch, tp_size, 2] where [:, :, 0]=values, [:, :, 1]=indices
        gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
        max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
        top_tokens = top_tokens.squeeze(-1).to(torch.int64)
        self._maybe_dump_top_token_margin(
            lm_head, hidden_states, embedding_bias, top_tokens
        )
        return top_tokens

    def _maybe_dump_top_token_margin(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None,
        selected_tokens: torch.Tensor,
    ) -> None:
        if torch.compiler.is_compiling():
            return
        step = _top_token_margin_dump_step(self)
        if step is None:
            return
        with torch.no_grad():
            logits = lm_head.quant_method.apply(
                lm_head, hidden_states, bias=embedding_bias
            )
            if self.soft_cap is not None:
                logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
            if self.scale != 1.0:
                logits = logits * self.scale
            num_pad = lm_head.shard_indices.num_org_vocab_padding
            if num_pad > 0:
                logits[..., -num_pad:] = -float("inf")
            logits = self._gather_logits(logits)
            if logits is None:
                return
            logits = logits[..., : self.org_vocab_size]
            top_k = min(8, logits.shape[-1])
            top_vals, top_indices = torch.topk(logits, k=top_k, dim=-1)
            selected_tokens = selected_tokens.to(logits.device).view(-1, 1)
            selected_vals = logits.gather(dim=-1, index=selected_tokens).squeeze(-1)
            margins = top_vals[:, 0] - top_vals[:, 1] if top_k > 1 else top_vals[:, 0]
            probe_token_ids = _parse_token_probe(
                os.getenv("VLLM_SM70_DUMP_TOP_TOKEN_MARGIN_PROBE_TOKENS")
            )
            if probe_token_ids:
                valid_probe_token_ids = [
                    token_id
                    for token_id in probe_token_ids
                    if token_id < logits.shape[-1]
                ]
                probe_index = torch.tensor(
                    valid_probe_token_ids,
                    dtype=torch.long,
                    device=logits.device,
                )
                probe_values = (
                    logits.index_select(dim=-1, index=probe_index)
                    .detach()
                    .float()
                    .cpu()
                    .tolist()
                )
            else:
                valid_probe_token_ids = []
                probe_values = []

        top_ids_list = top_indices.detach().cpu().tolist()
        top_vals_list = top_vals.detach().float().cpu().tolist()
        selected_list = selected_tokens.squeeze(-1).detach().cpu().tolist()
        selected_ranks = []
        for selected, top_ids in zip(selected_list, top_ids_list, strict=False):
            selected_ranks.append(
                top_ids.index(selected) if selected in top_ids else None
            )
        _write_top_token_margin_record(
            {
                "device": int(torch.cuda.current_device())
                if torch.cuda.is_available()
                else None,
                "decode_step": step,
                "num_tokens": int(hidden_states.shape[0]),
                "pid": int(os.getpid()),
                "selected_tokens": selected_list,
                "selected_top_ranks": selected_ranks,
                "selected_values": selected_vals.detach().float().cpu().tolist(),
                "top_ids": top_ids_list,
                "top_values": top_vals_list,
                "top1_top2_margins": margins.detach().float().cpu().tolist(),
                "probe_token_ids": valid_probe_token_ids,
                "probe_values": probe_values,
            }
        )

    def _maybe_custom_top1_argmax(
        self,
        local_pair: torch.Tensor,
    ) -> torch.Tensor | None:
        if not envs.VLLM_SM70_TOP1_CUSTOM_AR:
            return None
        if local_pair.numel() != 2 or local_pair.dtype != torch.float32:
            return None
        try:
            communicator = get_tp_group().device_communicator
            ca_comm = getattr(communicator, "ca_comm", None)
            if ca_comm is None:
                return None
            top_tokens = ca_comm.custom_top1_argmax(local_pair.reshape(-1))
        except Exception as exc:
            logger.warning_once(
                "SM70 custom top1 argmax failed; falling back to all_gather: %s",
                exc,
            )
            return None
        if top_tokens is None:
            return None
        logger.info_once("SM70 custom top1 argmax path enabled.")
        return top_tokens

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s
