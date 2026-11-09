# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 engine startup self-cleanup (opt21 infrastructure).

Runs once in the engine's parent process at startup, *before* workers spawn and
before any torch.compile / CUDA-graph capture.  Its behaviour follows the
compile-cache reuse toggle (``VLLM_DISABLE_COMPILE_CACHE``):

* reuse DISABLED  -> reclaim the early compile-cache tree and any stale
  ``/dev/shm`` offload leftovers, so a fresh run does not inherit state.
* reuse ENABLED   -> leave every cache untouched.

Every step is best-effort: it never raises into engine startup, and it only
touches shared resources while no engine is resident (all GPUs quiet), which is
also what makes reclaiming ``/dev/shm`` safe per the project's iron rules.
"""

import glob
import os
import subprocess

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# A GPU with at least this many MiB resident counts as "busy" (an engine is
# running); shared caches and /dev/shm must then be left alone.
_GPU_BUSY_MIB = 2000

_COMPILE_CACHE_SUBDIR = "torch_compile_cache"
_SHM_GLOB = "/dev/shm/vllm_offload_*.mmap"


def _gpu_quiet() -> bool:
    """True when every visible GPU shows negligible residency."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if out.returncode != 0:
        return False
    try:
        used = [int(x) for x in out.stdout.split()]
    except ValueError:
        return False
    return bool(used) and all(mib < _GPU_BUSY_MIB for mib in used)


def _path_in_use(path: str) -> bool:
    """True if any process currently has ``path`` open or mapped."""
    try:
        out = subprocess.run(
            ["lsof", "-t", "--", path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.stdout.strip():
            return True
        # lsof exits 1 when nothing matches; anything else is unusable output,
        # in which case fall through to the /proc scan.
        if out.returncode in (0, 1):
            return False
    except (OSError, subprocess.SubprocessError):
        pass
    base = os.path.basename(os.path.realpath(path))
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/maps", encoding="utf-8",
                      errors="ignore") as fh:
                for line in fh:
                    if base in line:
                        return True
        except OSError:
            continue
    return False


def _clean_compile_cache() -> None:
    cache_dir = os.path.join(envs.VLLM_CACHE_ROOT, _COMPILE_CACHE_SUBDIR)
    if not os.path.isdir(cache_dir):
        return
    import shutil

    try:
        shutil.rmtree(cache_dir)
    except OSError as exc:
        logger.warning_once("SM70 startup cleanup: cannot remove %s (%s)",
                            cache_dir, exc)
        return
    logger.info_once("SM70 startup cleanup reclaimed compile cache %s",
                     cache_dir)


def _clean_shm_leftovers() -> None:
    removed = 0
    for path in glob.glob(_SHM_GLOB):
        if _path_in_use(path):
            continue
        try:
            os.unlink(path)
            removed += 1
        except OSError:
            pass
    if removed:
        logger.info_once(
            "SM70 startup cleanup reclaimed %d stale /dev/shm file(s)", removed)


def sm70_startup_cleanup() -> None:
    """Best-effort startup cleanup; never raises into engine startup."""
    if not envs.VLLM_SM70_STARTUP_CACHE_CLEANUP:
        return
    if not envs.VLLM_DISABLE_COMPILE_CACHE:
        logger.info_once(
            "SM70 startup cleanup skipped: compile-cache reuse is enabled.")
        return
    if not _gpu_quiet():
        logger.info_once(
            "SM70 startup cleanup skipped: a GPU is still resident.")
        return
    try:
        _clean_compile_cache()
        _clean_shm_leftovers()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning_once("SM70 startup cleanup failed: %s", exc)
