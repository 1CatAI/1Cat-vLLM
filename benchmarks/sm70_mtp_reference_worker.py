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
