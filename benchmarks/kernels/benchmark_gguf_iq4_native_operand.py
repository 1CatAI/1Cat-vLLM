# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU operand oracle only: original-byte IQ4_XS Float/Half reconstruction."""

import argparse
import json
import runpy
from pathlib import Path

import gguf
import numpy as np
import torch


def stress_source():
    # Every original d is crossed with every six-bit scale code and all
    # sixteen LUT indices, including both nibble planes in each K32 group.
    blocks = np.zeros((512, 2, 136), dtype=np.uint8)
    ds = np.repeat(
        np.array(
            [0.0, -0.0, 2**-24, -(2**-24), 0.01337, -0.01337, 65504, -65504], "<f2"
        ),
        64,
    )
    blocks[..., :2] = (
        np.broadcast_to(ds[:, None, None], (512, 2, 1)).copy().view(np.uint8)
    )
    codes = np.broadcast_to(
        (np.arange(512, dtype=np.uint16) % 64)[:, None, None], (512, 2, 8)
    )
    hi = np.sum(((codes >> 4) & 3) << (2 * np.arange(8)), axis=-1, dtype=np.uint16)
    blocks[..., 2:4] = hi[..., None].astype("<u2").view(np.uint8)
    blocks[..., 4:8] = ((codes[..., ::2] & 15) | ((codes[..., 1::2] & 15) << 4)).astype(
        np.uint8
    )
    indices = np.arange(16, dtype=np.uint8)
    blocks[..., 8:] = np.tile(indices | (indices << np.uint8(4)), 8)
    return blocks.reshape(512, 272)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stress-only", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    api = runpy.run_path(
        str(source / "vllm/model_executor/layers/quantization/gguf_iq4_native.py")
    )
    torch.ops.load_library(str(args.library))
    device = torch.cuda.get_device_properties(0)
    assert (device.major, device.minor) == (7, 0)
    cases = []
    if not args.stress_only:
        reader = gguf.GGUFReader(str(args.model))
        tensors = {t.name: t for t in reader.tensors}
        for layer, role in ((39, "gate"), (42, "up")):
            tensor = tensors[f"blk.{layer}.ffn_{role}.weight"]
            assert int(tensor.tensor_type) == 23
            n, k = (int(v) for v in reversed(tensor.shape))
            assert n % 4 == 0 and n // 4 % 32 == 0
            cases.append(
                (f"layer{layer}_{role}_tp4_rank0", tensor.data[: n // 4].copy(), k)
            )
    cases.append(("all_scale_codes_signed_zeros_extreme_d", stress_source(), 512))
    rows = []
    for name, original, k in cases:
        n = original.shape[0]
        records = api["pack_iq4_xs_records"](original)
        np.testing.assert_array_equal(
            api["unpack_iq4_xs_records"](records, n, k), original
        )
        official = gguf.quants.dequantize(original, gguf.GGMLQuantizationType.IQ4_XS)
        packed = torch.from_numpy(records).cuda()
        for dtype, np_dtype, bits in (
            (torch.float32, np.float32, np.uint32),
            (torch.float16, np.float16, np.uint16),
        ):
            output = torch.empty((n, k), dtype=dtype, device="cuda")
            torch.ops.gguf_iq4_native_oracle_research.decode(output, packed)
            actual = output.cpu().numpy()
            with np.errstate(over="ignore"):
                reference = official.astype(np_dtype)
            mismatch = int(np.count_nonzero(actual.view(bits) != reference.view(bits)))
            finite = np.isfinite(actual) & np.isfinite(reference)
            row = {
                "case": name,
                "N": n,
                "K": k,
                "source_bytes": original.nbytes,
                "record_bytes": records.nbytes,
                "dtype": str(dtype),
                "elements": n * k,
                "bit_mismatches": mismatch,
                "max_abs_error_finite": float(
                    np.abs(actual[finite] - reference[finite]).max()
                ),
                "expected_nonfinite": int(np.count_nonzero(~np.isfinite(reference))),
                "actual_nonfinite": int(np.count_nonzero(~np.isfinite(actual))),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            assert mismatch == 0
            if name.startswith("layer"):
                assert row["expected_nonfinite"] == row["actual_nonfinite"] == 0
        del packed, output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "complete": True,
                "operand_only": True,
                "gpu": device.name,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
