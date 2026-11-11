# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy one PP stage's tensors (raw bytes) into per-layer safetensors files.

stdlib only. Usage: extract_stage.py SRC DST FIRST LAST [--mtp]
Writes DST/layer-XX.safetensors, DST/globals.safetensors, optional
DST/mtp.safetensors and DST/model.safetensors.index.json, plus small files.
"""

import json
import os
import shutil
import struct
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from st_inventory import headers

CHUNK = 64 << 20
GLOBALS = ("embed.", "head.", "norm.")


def write_file(path, names, inv, src, log):
    meta, off = {}, 0
    for n in names:
        v = inv[n]
        meta[n] = dict(
            dtype=v["dtype"], shape=v["shape"], data_offsets=[off, off + v["nbytes"]]
        )
        off += v["nbytes"]
    hdr = json.dumps(meta, separators=(",", ":")).encode()
    hdr += b" " * ((8 - len(hdr) % 8) % 8)
    crcs = {}
    tmp = path + ".part"
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(hdr)))
        out.write(hdr)
        files = {}
        for n in names:
            v = inv[n]
            f = files.get(v["file"]) or files.setdefault(
                v["file"], open(os.path.join(src, v["file"]), "rb")
            )
            f.seek(v["offset"])
            left = v["nbytes"]
            c = 0
            while left:
                b = f.read(min(CHUNK, left))
                if not b:
                    raise OSError(f"short read {n}")
                c = zlib.crc32(b, c)
                out.write(b)
                left -= len(b)
            crcs[n] = c
        for f in files.values():
            f.close()
    os.replace(tmp, path)
    log.write(json.dumps(dict(file=os.path.basename(path), bytes=off, crc=crcs)) + "\n")
    log.flush()


def main():
    src, dst, first, last = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    mtp = "--mtp" in sys.argv
    os.makedirs(dst, exist_ok=True)
    inv = headers(src)
    for fn in os.listdir(src):
        p = os.path.join(src, fn)
        if os.path.isfile(p) and not fn.endswith(".safetensors"):
            shutil.copy2(p, dst)
        elif os.path.isdir(p) and fn in ("encoding", "inference"):
            shutil.copytree(p, os.path.join(dst, fn), dirs_exist_ok=True)
    plan = [("globals.safetensors", sorted(n for n in inv if n.startswith(GLOBALS)))]
    for L in range(first, last + 1):
        pre = f"layers.{L}."
        plan.append(
            (
                f"layer-{L:02d}.safetensors",
                [n for n in inv if n.startswith(pre) and ".engram.embed." not in n],
            )
        )
    if mtp:
        plan.append(("mtp.safetensors", [n for n in inv if n.startswith("mtp.")]))
    wmap = {}
    with open(os.path.join(dst, "extract.log"), "a") as log:
        for fn, names in plan:
            for n in names:
                wmap[n] = fn
            if os.path.exists(os.path.join(dst, fn)):
                continue
            t = time.time()
            write_file(os.path.join(dst, fn), names, inv, src, log)
            gb = sum(inv[n]["nbytes"] for n in names) / 1e9
            print(f"{fn} {gb:.2f} GB {gb / (time.time() - t):.2f} GB/s", flush=True)
    json.dump(
        dict(
            metadata=dict(total_size=sum(inv[n]["nbytes"] for n in wmap)),
            weight_map=wmap,
        ),
        open(os.path.join(dst, "model.safetensors.index.json"), "w"),
        indent=1,
    )
    print("EXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
