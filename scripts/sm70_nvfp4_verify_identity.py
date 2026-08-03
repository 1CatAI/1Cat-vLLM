#!/usr/bin/env python3
"""Verify public 1.2.2 _C hash + presence of NVFP4 scale normalize helpers."""
from __future__ import annotations

import hashlib
import importlib.metadata as m
import pathlib
import sys

WANT_C = "ffc6271aaf25d96fe690d0f899544801cc1e39486905c49ad0090cb2cbe7a147"
WANT_WHEEL = "8a628983ad9d675559910372643220c418b307ddc7fd52ac65a7f5fbcb104bc6"


def main() -> int:
    try:
        ver = m.version("1cat-vllm")
    except Exception:
        ver = None
    print(f"1cat-vllm version: {ver}")
    if ver and ver != "1.2.2":
        print("WARN: documented wheel hashes are for 1.2.2; re-check for other releases")

    import vllm

    base = pathlib.Path(vllm.__file__).parent
    c = base / "_C.abi3.so"
    if not c.exists():
        print(f"MISSING {c}")
        return 2
    h = hashlib.sha256(c.read_bytes()).hexdigest()
    print(f"_C.abi3.so: {h}")
    print(f"  match_public_1.2.2: {h == WANT_C}")
    print(f"  expected_wheel_sha256: {WANT_WHEEL}")

    paths = [
        base / "model_executor/layers/quantization/sm70_turbomind.py",
        base
        / "model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py",
        base
        / "model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py",
    ]
    ok = h == WANT_C
    for p in paths:
        if not p.exists():
            print(f"MISSING {p}")
            ok = False
            continue
        text = p.read_text(errors="replace")
        has = "normalize_nvfp4_global_scale_for_sm70" in text or "normalize_nvfp4_global" in text
        print(f"{p.name}: normalize_helper={has} sha256={hashlib.sha256(p.read_bytes()).hexdigest()[:16]}...")
        if not has:
            ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
