# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dump CPU stacks during a stuck startup; disable the timer before inference."""

import faulthandler

from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.worker_base import CompilationTimes


class StartupStackWorker(Worker):
    def compile_or_warm_up_model(self) -> CompilationTimes:
        faulthandler.dump_traceback_later(90, repeat=True)
        try:
            return super().compile_or_warm_up_model()
        finally:
            faulthandler.cancel_dump_traceback_later()
