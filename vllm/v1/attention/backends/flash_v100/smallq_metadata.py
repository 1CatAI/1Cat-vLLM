# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility alias for speculative small-query metadata."""

import sys

from vllm.v1.attention.backends.flash_v100.spec import smallq_metadata as _owner

sys.modules[__name__] = _owner
