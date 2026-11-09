# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU check of K128 reader cursors against independent original GGUF bytes."""

import argparse
import json
import runpy
from pathlib import Path

import gguf
import numpy as np


def check_iq3(raw, packed, k):
    nb = k // 256
    original = raw.reshape(raw.shape[0], nb, 110)
    records = packed.reshape(raw.shape[0] // 32, nb * 3520)
    checked = 0
    for tile in range(records.shape[0]):
        for col in range(32):
            for warp in range(8):
                first_part = warp * (k // 1024)
                payload = first_part * 1536 + col * 16
                tail = nb * 3072 + first_part * 128 + col * 4
                dp = nb * 3328 + (first_part // 2) * 192 + col * 2
                sp = nb * 3328 + (first_part // 2) * 192 + 64 + col * 4
                half = first_part & 1
                cache = None
                for step in range(k // 1024):
                    data = records[tile]
                    if step == 0 or half == 0:
                        cache = (data[dp : dp + 2].copy(), data[sp : sp + 4].copy())
                    part = first_part + step
                    source = original[tile * 32 + col, part // 2]
                    np.testing.assert_array_equal(cache[0], source[:2])
                    np.testing.assert_array_equal(cache[1], source[106:110])
                    words = np.concatenate(
                        [
                            data[payload + i * 512 : payload + i * 512 + 16]
                            for i in range(3)
                        ]
                        + [data[tail : tail + 4]]
                    )
                    bits = np.unpackbits(words, bitorder="little").reshape(32, 13)
                    ids = (bits.astype(np.uint32) << np.arange(13)).sum(axis=1)
                    octets = np.arange(16) + half * 16
                    high = source[66 + octets // 4]
                    first = source[2 + 2 * octets].astype(np.uint32) | (
                        ((high >> (2 * (octets % 4))) & 1).astype(np.uint32) << 8
                    )
                    second = source[3 + 2 * octets].astype(np.uint32) | (
                        ((high >> (2 * (octets % 4) + 1)) & 1).astype(np.uint32) << 8
                    )
                    signs = source[74 + octets].astype(np.uint32)
                    expected = np.stack(
                        ((first << 4) | (signs & 15), (second << 4) | (signs >> 4)),
                        axis=-1,
                    ).reshape(32)
                    np.testing.assert_array_equal(ids, expected)
                    payload += 1536
                    tail += 128
                    dp += half * 192
                    sp += half * 192
                    half ^= 1
                    checked += 1
    return checked


def check_iq4(raw, packed, k):
    nb = k // 256
    original = raw.reshape(raw.shape[0], nb, 136)
    records = packed.reshape(raw.shape[0] // 32, nb * 4352)
    checked = 0
    for tile in range(records.shape[0]):
        for col in range(32):
            for warp in range(8):
                first_part = warp * (k // 1024)
                payload = first_part * 2048 + col * 16
                block = first_part // 2
                dp = nb * 4096 + block * 64 + col * 2
                hp = nb * 4160 + block * 64 + col * 2
                lp = nb * 4224 + block * 128 + col * 4
                half = first_part & 1
                cache = None
                for step in range(k // 1024):
                    data = records[tile]
                    if step == 0 or half == 0:
                        cache = tuple(
                            data[p : p + width].copy()
                            for p, width in ((dp, 2), (hp, 2), (lp, 4))
                        )
                    part = first_part + step
                    source = original[tile * 32 + col, part // 2]
                    for current, expected in zip(
                        cache, (source[:2], source[2:4], source[4:8])
                    ):
                        np.testing.assert_array_equal(current, expected)
                    for group in range(4):
                        packet = data[
                            payload + group * 512 : payload + group * 512 + 16
                        ]
                        codes = np.empty(32, dtype=np.uint8)
                        codes[::2], codes[1::2] = packet & 15, packet >> 4
                        qs = source[
                            8 + (half * 4 + group) * 16 : 8
                            + (half * 4 + group + 1) * 16
                        ]
                        np.testing.assert_array_equal(
                            codes, np.concatenate((qs & 15, qs >> 4))
                        )
                    payload += 2048
                    dp += half * 64
                    hp += half * 64
                    lp += half * 128
                    half ^= 1
                    checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    iq4 = runpy.run_path(
        str(root / "vllm/model_executor/layers/quantization/gguf_iq4_native.py")
    )["pack_iq4_xs_records"]
    iq3 = runpy.run_path(
        str(root / "vllm/model_executor/layers/quantization/gguf_iq3_records.py")
    )["signed_index_records"]
    tensors = {t.name: t for t in gguf.GGUFReader(str(args.model)).tensors}
    rows = []
    for layer in (39, 42):
        for role in ("gate", "up"):
            tensor = tensors[f"blk.{layer}.ffn_{role}.weight"]
            typ = int(tensor.tensor_type)
            raw = tensor.data[:64].copy()
            k = int(tensor.shape[0])
            packed = (iq3 if typ == 21 else iq4)(raw)
            count = (check_iq3 if typ == 21 else check_iq4)(raw, packed, k)
            row = {
                "layer": layer,
                "role": role,
                "type": typ,
                "N": 64,
                "K": k,
                "K128_records_checked": count,
                "record_byte_count": packed.nbytes,
                "mismatches": 0,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"complete": True, "cpu_only": True, "rows": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
