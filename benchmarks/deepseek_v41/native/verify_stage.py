# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check a stage directory written by extract_stage.py against its extract.log.

Every tensor's bytes are read back from the device (page cache dropped first)
and compared with the CRC32 recorded while copying from the checkpoint, so a
storage fault that left misplaced or stale pages shows up before serving.
Usage: verify_stage.py STAGE_DIR. Exit status 1 if any tensor differs.
"""

import json
import os
import struct
import sys
import zlib


def main():
    stage = sys.argv[1]
    records = [json.loads(line) for line in open(os.path.join(stage, "extract.log"))]
    bad_files = 0
    for record in records:
        path = os.path.join(stage, record["file"])
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
            base = 8 + n
            bad = []
            for name, want in record["crc"].items():
                a, b = header[name]["data_offsets"]
                f.seek(base + a)
                left, crc = b - a, 0
                while left:
                    chunk = f.read(min(64 << 20, left))
                    crc = zlib.crc32(chunk, crc)
                    left -= len(chunk)
                if crc != want:
                    bad.append(name)
        bad_files += bool(bad)
        print(record["file"], "OK" if not bad else f"BAD {bad[:3]}", flush=True)
    print("VERIFY_DONE bad_files", bad_files)
    sys.exit(1 if bad_files else 0)


if __name__ == "__main__":
    main()
