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


class DraftFCGatherControlWorker(Worker):
    """Retain two draft FC gathers with all other defaults and startup fixes."""

    def load_model(self, *, load_dummy_weights=False):
        from vllm.models.qwen4_exp.nvidia import sm70_mtp_fc

        sm70_mtp_fc.maybe_combine_fc = lambda *args: None
        super().load_model(load_dummy_weights=load_dummy_weights)
