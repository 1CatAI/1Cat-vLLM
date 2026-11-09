# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional native-wheel attachment for the explicit benchmark overlay mode.

Activated only when this directory is explicitly added to PYTHONPATH. It does
not change kernels, dispatch, precision, or an installed environment.
"""

import os
from pathlib import Path

if directory := os.environ.get("ONECAT_VLLM_EXTENSION_DIR"):
    import vllm

    extension_dir = Path(directory).expanduser().resolve()
    if not (extension_dir / "_C.abi3.so").is_file():
        raise RuntimeError(f"Missing native SM70 extension in {extension_dir}")
    if str(extension_dir) not in vllm.__path__:
        vllm.__path__.append(str(extension_dir))
