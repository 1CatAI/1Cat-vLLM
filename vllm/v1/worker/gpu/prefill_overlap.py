# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Large prefill batches as two interleaved micro-batches on tensor-parallel
engines: while one half all-reduces on a side stream, the other half computes.

Both halves run the whole model on the same compute stream, layer by layer in
turn, so the second half's attention sees the first half's KV cache writes and
recurrent state of the same layer, exactly as a following prefill chunk would.
"""

import dataclasses
import threading
from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import (
    BatchDescriptor,
    ForwardContext,
    create_forward_context,
    override_forward_context,
)
from vllm.logger import init_logger
from vllm.runtime_resources import activate_runtime_resources
from vllm.utils.torch_utils import current_stream
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.ubatching import (
    MicroBatchAborted,
    cat_ubatch_outputs,
    make_ubatch_contexts,
)

logger = init_logger(__name__)

# Builder 0 serves the whole batch, whose metadata the speculator reuses.
HALF_BUILDER_IDS = (1, 2)
NUM_METADATA_BUILDERS = 1 + len(HALF_BUILDER_IDS)


def token_range_requests(query_start_loc: np.ndarray, start: int, end: int) -> slice:
    """The requests with tokens in ``[start, end)``."""
    first = int(np.searchsorted(query_start_loc, start, side="right")) - 1
    last = int(np.searchsorted(query_start_loc, end, side="left"))
    return slice(first, last)


def token_range_batch(input_batch: InputBatch, start: int, end: int) -> InputBatch:
    """Tokens ``[start, end)`` of a batch without draft tokens as a batch of
    their own, for attention metadata and the forward. A request cut at
    ``end`` ends there; one cut at ``start`` continues after the tokens before
    it. The range produces no logits; sampling uses the whole batch."""
    assert input_batch.num_draft_tokens == 0
    assert input_batch.dcp_local_seq_lens is None
    query_start_loc = input_batch.query_start_loc_np
    reqs = token_range_requests(query_start_loc, start, end)
    first, last = reqs.start, reqs.stop
    num_reqs = last - first
    num_tokens = end - start
    cut_before = np.maximum(start - query_start_loc[first:last], 0)
    cut_after = np.maximum(query_start_loc[first + 1 : last + 1] - end, 0)
    tokens_after_gpu = (input_batch.query_start_loc[first + 1 : last + 1] - end).clamp(
        min=0
    )
    is_incomplete_prefilling_np = cut_after > 0
    if input_batch.is_incomplete_prefilling_np is not None:
        is_incomplete_prefilling_np |= input_batch.is_incomplete_prefilling_np[reqs]
    cu_num_logits_np = np.zeros(num_reqs + 1, dtype=input_batch.cu_num_logits_np.dtype)

    def requests(values):
        return None if values is None else values[reqs]

    return dataclasses.replace(
        input_batch,
        req_ids=input_batch.req_ids[reqs],
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs,
        idx_mapping=input_batch.idx_mapping[reqs],
        idx_mapping_np=input_batch.idx_mapping_np[reqs],
        expanded_idx_mapping=input_batch.expanded_idx_mapping[:0],
        expanded_local_pos=input_batch.expanded_local_pos[:0],
        num_scheduled_tokens=(
            input_batch.num_scheduled_tokens[reqs] - cut_before - cut_after
        ),
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        num_draft_tokens_per_req=requests(input_batch.num_draft_tokens_per_req),
        query_start_loc=(input_batch.query_start_loc[first : last + 1] - start).clamp(
            0, num_tokens
        ),
        query_start_loc_np=np.clip(
            query_start_loc[first : last + 1] - start, 0, num_tokens
        ),
        seq_lens=input_batch.seq_lens[reqs] - tokens_after_gpu,
        seq_lens_cpu_upper_bound=(
            input_batch.seq_lens_cpu_upper_bound[reqs] - torch.from_numpy(cut_after)
        ),
        num_computed_tokens_np=input_batch.num_computed_tokens_np[reqs] + cut_before,
        prefill_len_np=input_batch.prefill_len_np[reqs],
        num_computed_prefill_tokens_np=(
            input_batch.num_computed_prefill_tokens_np[reqs] + cut_before
        ),
        is_prefilling_np=input_batch.is_prefilling_np[reqs],
        max_seq_len_np=requests(input_batch.max_seq_len_np),
        input_ids=input_batch.input_ids[start:end],
        positions=input_batch.positions[start:end],
        logits_indices=input_batch.logits_indices[:0],
        cu_num_logits=input_batch.cu_num_logits.new_zeros(num_reqs + 1),
        cu_num_logits_np=cu_num_logits_np,
        has_structured_output_reqs=False,
        prefix_anchor_lens=requests(input_batch.prefix_anchor_lens),
        is_incomplete_prefilling_np=is_incomplete_prefilling_np,
    )


def token_range_model_inputs(
    model_inputs: dict[str, Any], start: int, end: int
) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for name, value in model_inputs.items():
        if value is None:
            sliced[name] = None
        elif name == "positions":
            # 1D, or (3, num_tokens) with M-RoPE.
            sliced[name] = value[..., start:end]
        elif name in ("input_ids", "inputs_embeds"):
            sliced[name] = value[start:end]
        else:
            raise NotImplementedError(
                f"tp_prefill_overlap cannot split the model input {name!r}"
            )
    return sliced


class PrefillOverlap:
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.tensor_parallel_size < 2
            or parallel_config.pipeline_parallel_size != 1
            or parallel_config.data_parallel_size != 1
        ):
            raise ValueError(
                "kernel_config.tp_prefill_overlap_min_tokens needs tensor "
                "parallelism without pipeline or data parallelism."
            )
        if vllm_config.lora_config is not None or vllm_config.kv_transfer_config:
            raise ValueError(
                "kernel_config.tp_prefill_overlap_min_tokens excludes LoRA and "
                "KV transfer, whose per-batch state it cannot split."
            )
        if parallel_config.decode_context_parallel_size > 1:
            raise ValueError(
                "kernel_config.tp_prefill_overlap_min_tokens excludes decode "
                "context parallelism, whose sequence lengths it cannot split."
            )
        if vllm_config.model_config.is_encoder_decoder:
            raise ValueError(
                "kernel_config.tp_prefill_overlap_min_tokens excludes "
                "encoder-decoder models, whose encoder inputs it cannot split."
            )
        self.vllm_config = vllm_config
        self.min_tokens = vllm_config.kernel_config.tp_prefill_overlap_min_tokens
        self.device = device
        self.comm_stream = torch.cuda.Stream(device)
        # Two micro-batch threads and the caller.
        self.ready_barrier = threading.Barrier(3)

    @staticmethod
    def check_model(model: torch.nn.Module) -> None:
        """Reject models whose forward the two halves cannot share: MoE layers
        dispatch without the micro-batch hooks, and deepstack vision features
        live in a model buffer indexed from token 0 in each half (Qwen3-VL
        counts their levels in deepstack_num_level, Granite 4 Vision lists the
        layers that read them in _ds_layer_indices)."""
        # Imported here: the MoE layer pulls in the model executor.
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE

        for module in model.modules():
            if isinstance(module, FusedMoE):
                raise ValueError(
                    "kernel_config.tp_prefill_overlap_min_tokens supports "
                    "dense models only; this model has MoE layers."
                )
            if getattr(module, "deepstack_num_level", 0) > 0 or getattr(
                module, "_ds_layer_indices", None
            ):
                raise ValueError(
                    "kernel_config.tp_prefill_overlap_min_tokens cannot split "
                    "the deepstack visual features of this model."
                )

    def split_token(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        model_state: ModelState,
    ) -> int | None:
        """Where the batch splits into two micro-batches, or None to keep it."""
        if (
            cudagraph_mode != CUDAGraphMode.NONE
            or input_batch.num_tokens < self.min_tokens
            or input_batch.num_draft_tokens > 0
            or input_batch.num_tokens_after_padding != input_batch.num_tokens
        ):
            return None
        split = model_state.align_ubatch_split(input_batch, input_batch.num_tokens // 2)
        return split if split < input_batch.num_tokens else None

    def forward_context(
        self,
        attn_metadata: dict[str, Any],
        slot_mapping: dict[str, torch.Tensor],
        num_tokens: int,
        skip_compiled: bool,
    ) -> ForwardContext:
        return create_forward_context(
            attn_metadata,
            self.vllm_config,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
            slot_mapping=slot_mapping,
            skip_compiled=skip_compiled,
        )

    def run(
        self,
        model: torch.nn.Module,
        model_inputs: list[dict[str, Any]],
        forward_contexts: list[ForwardContext],
    ) -> Any:
        """Run the halves in turn on the caller's stream; each all-reduce runs
        on the comm stream while the other half computes."""
        logger.info_once(
            "TP prefill overlap: prefill batches of at least %d tokens run as "
            "two interleaved halves.",
            self.min_tokens,
        )
        contexts = make_ubatch_contexts(
            num_micro_batches=len(model_inputs),
            compute_stream=current_stream(),
            comm_stream=self.comm_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
            overlap_collectives=True,
        )
        outputs: list[Any] = [None] * len(contexts)
        errors: list[BaseException] = []

        def run_half(ctx, inputs):
            try:
                # New threads start on device 0 and its default stream.
                torch.accelerator.set_device_index(self.device)
                torch.accelerator.set_stream(ctx.compute_stream)
                with (
                    torch.inference_mode(),
                    ctx,
                    activate_runtime_resources(ctx.forward_context.runtime_resources),
                ):
                    outputs[ctx.id] = model(**inputs)
            except BaseException as error:
                errors.append(error)
                raise

        # The threads switch the global forward context; restore it afterwards.
        with override_forward_context(None):
            threads = [
                threading.Thread(target=run_half, args=(ctx, inputs))
                for ctx, inputs in zip(contexts, model_inputs)
            ]
            for thread in threads:
                thread.start()
            self.ready_barrier.wait()
            contexts[0].cpu_wait_event.set()
            for thread in threads:
                thread.join()
        if errors:
            # The other half stops with MicroBatchAborted; report the cause.
            raise next(
                (e for e in errors if not isinstance(e, MicroBatchAborted)),
                errors[0],
            )
        return cat_ubatch_outputs(outputs)
