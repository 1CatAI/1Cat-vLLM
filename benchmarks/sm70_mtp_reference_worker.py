# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic reference worker for main-policy MTP projection comparison.

Never use its requests as the default speed result. The reference retains
main's FP32-policy vendor shared projection and the checkpoint draft head.
"""

from vllm.v1.worker.gpu_worker import Worker


class ReferenceWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv, sm70_mtp_head

        sm70_fp16_gemv._shared_batch_runtime_ok = lambda x: False
        sm70_mtp_head.prepare_mtp_qpn8_head = lambda head: None
        super().load_model(load_dummy_weights=load_dummy_weights)


class HeadCandidateWorker(ReferenceWorker):
    def load_model(self, *, load_dummy_weights=False):
        # Load the ordinary shared target head first, then install a view only
        # on the proposer. Failed numerical candidates never become defaults.
        super().load_model(load_dummy_weights=load_dummy_weights)
        from vllm.models.qwen4_exp.nvidia.sm70_mtp_head import MTPQPN8Head

        model = self.model_runner.speculator.model
        model._sm70_draft_head = MTPQPN8Head(model.lm_head)


class OutputCandidateWorker(ReferenceWorker):
    def load_model(self, *, load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        from vllm.model_executor.layers.quantization.sm70_online_qpn8 import (
            prepare_channel_qpn8_weight,
        )

        seen = set()
        for model in (self.model_runner.model, self.model_runner.speculator.model):
            for layer in model.modules():
                if id(layer) in seen:
                    continue
                seen.add(id(layer))
                weight = getattr(layer, "weight", None)
                prefix = str(getattr(layer, "prefix", ""))
                if (
                    weight is not None
                    and weight.is_cuda
                    and tuple(weight.shape) == (2560, 1536)
                    and prefix.endswith((".linear_attn.out_proj", ".self_attn.o_proj"))
                ):
                    codes, scales = prepare_channel_qpn8_weight(weight)
                    layer.register_buffer(
                        "_sm70_qwen38_output_codes", codes, persistent=False
                    )
                    layer.register_buffer(
                        "_sm70_qwen38_output_scales", scales, persistent=False
                    )


class ShortlistCandidateWorker(Worker):
    """Benchmark-only shortlist, installed before normal graph capture."""

    def load_model(self, *, load_dummy_weights=False):
        import json
        from pathlib import Path

        super().load_model(load_dummy_weights=load_dummy_weights)
        from vllm.models.qwen4_exp.nvidia.sm70_mtp_head import prepare_mtp_qpn8_head

        model = self.model_runner.speculator.model
        if model._sm70_draft_head is None:
            model._sm70_draft_head = prepare_mtp_qpn8_head(model.lm_head)
        if model._sm70_draft_head is None:
            raise RuntimeError("Shortlist candidate requires admitted QPN8 draft head")
        config = self.vllm_config.model_config.hf_config
        path = Path(config.sm70_mtp_draft_vocab_file)
        model._sm70_draft_head.prepare_shortlist(
            json.loads(path.read_text())["token_ids"]
        )


class RestorationControlWorker(Worker):
    """Matched checkpoint-head/zero-split HC control; all other routes retained."""

    def load_model(self, *, load_dummy_weights=False):
        from vllm import _custom_ops as ops
        from vllm.models.qwen4_exp.nvidia import sm70_mtp_head

        sm70_mtp_head.prepare_mtp_qpn8_head = lambda head: None
        original_hc = ops.sm70_qwen38_hc_batch

        def zero_split(*args, **kwargs):
            if len(args) > 13:
                args = (*args[:13], 0)
            else:
                kwargs["cta_split_warps"] = 0
            return original_hc(*args, **kwargs)

        ops.sm70_qwen38_hc_batch = zero_split
        super().load_model(load_dummy_weights=load_dummy_weights)


class SharedChainCandidateWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        from vllm.models.qwen4_exp.nvidia.sm70_mtp_structural import (
            prepare_shared_chain_probe,
        )

        target = prepare_shared_chain_probe(self.model_runner.model)
        draft = prepare_shared_chain_probe(self.model_runner.speculator.model)
        if target != 48 or draft != 1:
            raise RuntimeError(
                f"Shared-chain preparation missed layers: {target}/{draft}"
            )


class DraftExpertQPN8CandidateWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        from vllm.models.qwen4_exp.nvidia.sm70_mtp_structural import (
            prepare_draft_expert_qpn8_probe,
        )

        prepared = prepare_draft_expert_qpn8_probe(self.model_runner.speculator.model)
        if prepared != 1:
            raise RuntimeError(
                f"Draft QPN8 expert preparation missed layer: {prepared}"
            )
