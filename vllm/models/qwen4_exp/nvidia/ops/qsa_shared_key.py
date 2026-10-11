# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility imports for the shared attention QSA scoring provider."""

from vllm.v1.attention.ops.sm70_qsa_shared_key import (
    load_operator as load_operator,
)
from vllm.v1.attention.ops.sm70_qsa_shared_key import (
    maybe_shared_key_scores as maybe_shared_key_scores,
)
from vllm.v1.attention.ops.sm70_qsa_shared_key import (
    shared_key_reason as shared_key_reason,
)
