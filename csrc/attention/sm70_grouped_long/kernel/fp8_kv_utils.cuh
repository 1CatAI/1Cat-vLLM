// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

// Share the codec with the vendored extension; do not duplicate converters.
#include "../../../../flash-attention-v100/kernel/kv_codec.cuh"
