# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Load the CPU-only package without initializing vLLM's CUDA runtime."""

import importlib.util
import sys
from pathlib import Path

PACKAGE = (
    Path(__file__).resolve().parents[3]
    / "vllm/distributed/device_communicators/topology_all_reduce"
)
NAME = "_vllm_topology_all_reduce_tests"
if NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        NAME, PACKAGE / "__init__.py", submodule_search_locations=[str(PACKAGE)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = module
    spec.loader.exec_module(module)
