# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.worker.gpu.spec_decode.dspark.block_drafter import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import Inkling2DSparkSpeculator

__all__ = ["DSparkSpeculator", "Inkling2DSparkSpeculator"]
