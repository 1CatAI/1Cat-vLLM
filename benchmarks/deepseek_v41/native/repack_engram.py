# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repack a V4.1 Engram table into 264-byte rows (256 B E4M3 + 8 B E8M0).

Usage: repack_engram.py SRC_DIR LAYER OUT_DIR. Writes OUT_DIR/engram-L.rows and
OUT_DIR/engram-L.json (rows, row_bytes, row_bias, exponent stats, crc).
"""

import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from st_inventory import headers

ROWS = 1 << 21  # 2M rows per chunk: 512 MiB codes + 16 MiB scales


def main():
    src, layer, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    inv = headers(src)
    w = inv[f"layers.{layer}.engram.embed.weight"]
    s = inv[f"layers.{layer}.engram.embed.scale"]
    n, width = w["shape"]
    assert (width, s["shape"]) == (256, [n, 8]), (w, s)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"engram-{layer:02d}.rows")
    hist = np.zeros(256, np.int64)
    nan_codes = 0
    crc = 0
    t = time.time()
    with (
        open(os.path.join(src, w["file"]), "rb") as fw,
        open(os.path.join(src, s["file"]), "rb") as fs,
        open(path + ".part", "wb") as out,
    ):
        for start in range(0, n, ROWS):
            rows = min(ROWS, n - start)
            fw.seek(w["offset"] + start * 256)
            codes = np.frombuffer(fw.read(rows * 256), np.uint8).reshape(rows, 256)
            fs.seek(s["offset"] + start * 8)
            scales = np.frombuffer(fs.read(rows * 8), np.uint8).reshape(rows, 8)
            nan_codes += int(np.count_nonzero((codes & 0x7F) == 0x7F))
            hist += np.bincount(scales.ravel(), minlength=256)
            packed = np.empty((rows, 264), np.uint8)
            packed[:, :256] = codes
            packed[:, 256:] = scales
            b = packed.tobytes()
            crc = zlib.crc32(b, crc)
            out.write(b)
            if start // ROWS % 16 == 0:
                done = (start + rows) / n
                print(
                    f"layer {layer}: {done:.1%} {(start + rows) * 264 / 1e9 / (time.time() - t):.2f} GB/s",
                    flush=True,
                )
    os.replace(path + ".part", path)
    used = np.nonzero(hist)[0]
    smin, smax = int(used.min()), int(used.max())
    # Largest |value| = 448 * 2^(smax-127); keep it at 448 * 2^6 in FP16.
    bias = 6 - (smax - 127)
    meta = dict(
        layer=layer,
        rows=n,
        row_bytes=264,
        code_bytes=256,
        scale_bytes=8,
        row_bias=bias,
        scale_min=smin,
        scale_max=smax,
        nan_codes=nan_codes,
        scale_hist={str(i): int(hist[i]) for i in used},
        crc32=crc,
        source=dict(weight=w, scale=s),
    )
    json.dump(
        meta, open(os.path.join(out_dir, f"engram-{layer:02d}.json"), "w"), indent=1
    )
    print(
        "REPACK_DONE",
        layer,
        json.dumps(
            {
                k: meta[k]
                for k in ("rows", "row_bias", "scale_min", "scale_max", "nan_codes")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
