# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E8M0 exponent ranges of every FP8/MXFP4 scale tensor in local layer files."""

import glob
import json
import struct
import sys

import numpy as np

for fn in sorted(glob.glob(sys.argv[1] + "/layer-*.safetensors")) + glob.glob(
    sys.argv[1] + "/mtp.safetensors"
):
    f = open(fn, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    h = json.loads(f.read(n))
    base = 8 + n
    worst = {}
    for k, v in h.items():
        if k == "__metadata__" or v["dtype"] != "F8_E8M0":
            continue
        a, b = v["data_offsets"]
        f.seek(base + a)
        s = np.frombuffer(f.read(b - a), np.uint8)
        kind = (
            "expert" if ".experts." in k and "shared" not in k else k.split(".", 2)[2]
        )
        lo, hi = int(s.min()) - 127, int(s.max()) - 127
        if kind == "expert":
            e = s.reshape(-1)
            w = worst.setdefault("expert", [99, -99, 0])
            w[0] = min(w[0], lo)
            w[1] = max(w[1], hi)
            w[2] = max(w[2], hi - lo)
        else:
            worst[kind] = [lo, hi]
    out = {
        k: v
        for k, v in worst.items()
        if k == "expert" or v[0] < -14 or v[1] > 7 or "engram" in k
    }
    print(fn.split("/")[-1], json.dumps(out))
