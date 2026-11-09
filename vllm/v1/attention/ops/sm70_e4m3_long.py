# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility alias for the grouped-family long operator."""

import sys

from vllm.v1.attention.ops import sm70_grouped_long as _owner

sys.modules[__name__] = _owner
