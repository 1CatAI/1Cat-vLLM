# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Import/export native tuning tables in the same policy domain as execution."""

import hashlib
import json
import tempfile
from pathlib import Path

import pybase64 as base64

from vllm._sm70.policy import NativeBindings
from vllm.config import get_current_vllm_config_or_none
from vllm.config.sm70_native import NATIVE_FIELDS

_MAGIC = b"SM70POLICY1\n"


def _prepared_policies():
    cfg = get_current_vllm_config_or_none()
    if cfg is None:
        return ()
    kernel = cfg.kernel_config
    policies = [
        getattr(kernel, "sm70_" + family).native
        for family in ("awq", "fp8", "nvfp4", "gguf")
    ]
    policies.append(kernel.sm70_mxfp4)
    policies.extend(
        getattr(kernel.sm70_moe, family).native
        for family in ("awq", "fp8", "nvfp4", "mxfp4")
    )
    return tuple(policy for policy in policies if policy.values)


def _policies():
    values = [(), *(policy.values for policy in _prepared_policies())]
    return tuple(dict.fromkeys(NativeBindings(v).values for v in values))


def _accepts_legacy_seed(values):
    # Untagged historical tables preserve legacy startup defaults. A typed
    # calculation override must use its matching partition or warm up afresh.
    return not any(
        policy.values == values
        and any(
            policy.sources.get(field) == "configuration"
            for field, _alias, _families, diagnostic in NATIVE_FIELDS
            if not diagnostic
        )
        for policy in _prepared_policies()
    )


def _key(values):
    if not values:
        return "legacy"
    calculation = tuple(
        value for entry, value in zip(NATIVE_FIELDS, values) if not entry[3]
    )
    return hashlib.sha256(json.dumps(calculation).encode()).hexdigest()


def import_cache(device_hint, path: str) -> int:
    policies = _policies()
    partition_path = Path(path + ".policy-v1")
    source = (
        partition_path if policies != ((),) and partition_path.exists() else Path(path)
    )
    payload = source.read_bytes()
    records = 0
    if not payload.startswith(_MAGIC):
        # An existing unpartitioned LUT remains an explicitly imported seed.
        # Subsequent tuning writes to each policy's own native cache.
        for values in policies:
            if not _accepts_legacy_seed(values):
                continue
            records += int(
                NativeBindings(values).sm70_gemm_import_cache(device_hint, path)
            )
        return records
    partitions = json.loads(payload[len(_MAGIC) :])
    with tempfile.TemporaryDirectory(prefix="vllm-sm70-policy-import-") as directory:
        for values in policies:
            encoded = partitions.get(_key(values))
            if encoded is None:
                continue
            native_path = Path(directory) / "table.bin"
            native_path.write_bytes(base64.b64decode(encoded, validate=True))
            records += int(
                NativeBindings(values).sm70_gemm_import_cache(
                    device_hint, str(native_path)
                )
            )
    return records


def export_cache(device_hint, path: str) -> int:
    policies = _policies()
    if policies == ((),):
        return int(NativeBindings().sm70_gemm_export_cache(device_hint, path))
    records = 0
    partitions = {}
    with tempfile.TemporaryDirectory(prefix="vllm-sm70-policy-export-") as directory:
        for values in policies:
            native_path = Path(directory) / "table.bin"
            records += int(
                NativeBindings(values).sm70_gemm_export_cache(
                    device_hint, str(native_path)
                )
            )
            partitions[_key(values)] = base64.b64encode(
                native_path.read_bytes()
            ).decode("ascii")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Keep the historical raw LUT readable by older binaries/tools.
    target.write_bytes(base64.b64decode(partitions["legacy"]))
    Path(path + ".policy-v1").write_bytes(
        _MAGIC + json.dumps(partitions, sort_keys=True).encode()
    )
    return records


def packed_cache_bytes(path: Path) -> bytes:
    partitions = Path(str(path) + ".policy-v1")
    return (partitions if partitions.exists() else path).read_bytes()
