# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Header-only inventory of a safetensors checkpoint (stdlib only)."""

import collections
import glob
import json
import os
import struct
import sys

import regex as re


def headers(src):
    out = {}
    for fn in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        with open(fn, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        base = 8 + n
        for k, v in h.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            out[k] = dict(
                file=os.path.basename(fn),
                offset=base + a,
                nbytes=b - a,
                dtype=v["dtype"],
                shape=v["shape"],
            )
    return out


if __name__ == "__main__":
    src = sys.argv[1]
    inv = headers(src)
    json.dump(inv, open(sys.argv[2], "w"))
    groups = collections.Counter()
    for k, v in inv.items():
        if k.startswith("layers."):
            g = "layers.N." + re.sub(r"\d+", "N", k.split(".", 2)[2])
        elif k.startswith("mtp."):
            g = "mtp.N." + re.sub(r"\d+", "N", k.split(".", 2)[2])
        else:
            g = re.sub(r"\d+", "N", k)
        groups[(g, v["dtype"])] += 1
    nonlayer = sorted(
        {
            re.sub(r"\.\d+\.", ".N.", k)
            for k in inv
            if not k.startswith(("layers.", "mtp."))
        }
    )
    print("tensors", len(inv), "bytes", sum(v["nbytes"] for v in inv.values()))
    print(
        "NONLAYER prefixes:",
        sorted({k.split(".")[0] for k in inv if not k.startswith(("layers.", "mtp."))}),
    )
    for (g, d), c in sorted(groups.items()):
        if "experts.N." in g and not g.endswith(("w1.weight", "w1.scale")):
            continue
        if g.startswith(("vision", "visual", "vit")):
            continue
        print(f"{c:6d} {d:8s} {g}")
