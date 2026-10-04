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
