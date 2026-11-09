# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extract immutable image/video/audio references from the pinned H3 example."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from vllm.model_executor.models.minimax_h3.config import BASE_REVISION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.model_root / "assets/ref2va.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to prepare reference fixtures")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, options in (
        ("first.png", ["-frames:v", "1"]),
        ("reference.mp4", ["-t", "4", "-c:v", "libx264", "-crf", "18", "-c:a", "aac"]),
        ("reference.wav", ["-t", "4", "-vn", "-ar", "32000", "-c:a", "pcm_f32le"]),
    ):
        subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-i", str(source), *options, str(out / name)],
            check=True,
        )
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-sseof",
            "-0.1",
            "-i",
            str(source),
            "-frames:v",
            "1",
            str(out / "last.png"),
        ],
        check=True,
    )
    files = []
    for path in (source, *sorted(out.iterdir())):
        if not path.is_file() or path.suffix == ".json":
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files.append(
            {"filename": path.name, "sha256": digest, "bytes": path.stat().st_size}
        )
    (out / "manifest.json").write_text(
        json.dumps({"revision": BASE_REVISION, "files": files}, indent=2)
    )


if __name__ == "__main__":
    main()
