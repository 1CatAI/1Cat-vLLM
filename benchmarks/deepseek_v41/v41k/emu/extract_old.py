# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write old_extract.inc: the old v41k kernels the emulator compares against.

Cuts line ranges out of v41k.cu at a base commit (default 8a7bd7b, the
EXPERT_FAST branch point), checks each range's first line, and applies the
marked edits that make the host build follow the old SASS: Group3::apply
returns d * xd * S, and each accumulation is FFMA(that, 0.125, acc).

    python benchmarks/deepseek_v41/v41k/emu/extract_old.py [base]
"""

import os
import subprocess
import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "8a7bd7b"
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "benchmarks/deepseek_v41/v41k/v41k.cu"
# (first, last, first line starts with)
RANGES = [
    (19, 36, "template <int Type>"),  # block_bytes, build_table
    (565, 578, "__device__ __forceinline__ void ld3("),  # ld3, ld16
    (583, 672, "template <int Type>"),  # Group3, book_words
    (674, 686, "__device__ __forceinline__ int find_distinct("),
    (1264, 1288, "__device__ __forceinline__ void load_q8("),  # + q8_warp_aligned
    (1290, 1392, "#ifndef V41K_GU_MINB"),  # gate_up_em5
    (1657, 1750, "// Row-split down (G3)"),  # down_rows
]
EDITS = [
    (
        "    return d * xd * float(s0 * m0 + s1 * m1) * 0.125f;",
        (
            "    return d * xd * float(s0 * m0 + s1 * m1);  // EDIT 1: the x0.125 is"
            " the FFMA below (old SASS)"
        ),
    ),
    (
        "        gs[c] += dg.apply(q, d);",
        (
            "        gs[c] = __builtin_fmaf(dg.apply(q, d), 0.125f, gs[c]);  // EDIT 2:"
            " FFMA(x, 0.125, acc) as in the old SASS"
        ),
    ),
    (
        "        us[c] += du.apply(q, d);",
        "        us[c] = __builtin_fmaf(du.apply(q, d), 0.125f, us[c]);  // EDIT 3",
    ),
    (
        "            dot[c] += w.apply(qv, d);",
        (
            "            dot[c] = __builtin_fmaf(w.apply(qv, d), 0.125f, dot[c]);"
            "  // EDIT 4: FFMA(x, 0.125, dot) as in the old SASS"
        ),
    ),
]


def main():
    root = subprocess.run(
        ["git", "-C", HERE, "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lines = subprocess.run(
        ["git", "-C", root, "show", f"{BASE}:{SRC}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\n")
    out = [f"// Extracted verbatim from v41k.cu (base {BASE}) by extract_old.py;"]
    out.append("// the marked EDIT lines are the only changes.")
    out.append("namespace v41k {")
    for first, last, head in RANGES:
        chunk = lines[first - 1 : last]
        assert chunk[0].startswith(head), (first, chunk[0])
        out += chunk
    out.append("}  // namespace v41k")
    text = "\n".join(out) + "\n"
    for old, new in EDITS:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    with open(os.path.join(HERE, "old_extract.inc"), "w") as f:
        f.write(text)
    print("wrote", os.path.join(HERE, "old_extract.inc"))


if __name__ == "__main__":
    main()
