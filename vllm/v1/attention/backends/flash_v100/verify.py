# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility import for the feature-owned verification executor."""

import sys

from vllm.v1.attention.backends.flash_v100.spec import verifier

sys.modules[__name__] = verifier
