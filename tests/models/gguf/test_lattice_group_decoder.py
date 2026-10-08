# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the production integer decoder on CPU against official GGUF decoding."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import gguf
import numpy as np
import pytest
import regex as re

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / "csrc/sm70_turbomind/ops"
BOOKS = ROOT / "csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm"


@pytest.fixture(scope="module")
def decoder(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    assert compiler is not None
    library = tmp_path_factory.mktemp("lattice-group") / "probe.so"
    subprocess.run(
        [
            compiler,
            "-O2",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-Wno-unknown-pragmas",
            "-I",
            str(OPS),
            str(ROOT / "benchmarks/csrc/sm70_lattice_group_host_probe.cpp"),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    lib = ctypes.CDLL(str(library))
    lib.lattice_groups.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        *[ctypes.c_void_p] * 5,
    ]
    lib.lattice_groups.restype = ctypes.c_int
    return lib.lattice_groups


def native_book(source_type):
    source = (BOOKS / "lattice_codebooks.h").read_text()
    pattern = rf"lattice_grid_{source_type}\[\] = \{{(.*?)\}};"
    body = re.search(pattern, source, re.S)
    assert body is not None
    data = np.array([int(v) for v in re.findall(r"\d+", body[1])], dtype=np.uint8)
    return data.view("<u4") ^ np.uint32(0x80808080)


@pytest.mark.parametrize("source_type", [18, 21, 22])
@pytest.mark.parametrize("bank_aware", [False, True])
@pytest.mark.parametrize("pattern", ["random", "zero", "ones"])
def test_source_words_and_subscales_match_official(
    decoder, source_type, bank_aware, pattern
):
    qtype = gguf.GGMLQuantizationType(source_type)
    _, size = gguf.GGML_QUANT_SIZES[qtype]
    rng = np.random.default_rng(source_type)
    blocks = rng.integers(0, 256, (257, size), dtype=np.uint8)
    if pattern != "random":
        blocks.fill(0 if pattern == "zero" else 255)
    # Cover both signs, subnormals, zero, and finite FP16 exponent boundaries.
    bits = np.array(
        [
            0,
            1,
            0x3FF,
            0x400,
            0x3555,
            0x3C00,
            0x7BFF,
            0x8000,
            0x8001,
            0x83FF,
            0x8400,
            0xBC00,
        ],
        dtype="<u2",
    )
    blocks[:, :2] = np.resize(bits, blocks.shape[0]).view(np.uint8).reshape(-1, 2)
    book = native_book(source_type)
    if bank_aware and source_type == 22:
        book = np.concatenate((book[::2], book[1::2]))
    masks = np.array(
        [sum(255 << (i * 8) for i in range(4) if s & (1 << i)) for s in range(16)],
        dtype=np.uint32,
    )
    words = np.empty((blocks.shape[0], 8, 32), dtype=np.int8)
    scales = np.empty((blocks.shape[0], 8), dtype="<u2")
    subscales = np.empty((blocks.shape[0], 8, 2), dtype=np.uint8)
    arrays = (book, masks, words, scales, subscales)
    status = decoder(
        source_type,
        bank_aware,
        blocks.ctypes.data,
        blocks.shape[0],
        *[a.ctypes.data for a in arrays],
    )
    assert status == 0
    d = scales.view("<f2").astype(np.float32)[..., None]
    multipliers = np.repeat(subscales.astype(np.float32), 16, axis=-1)
    if source_type != 22:
        multipliers[..., 16:] = multipliers[..., :16]
    factor = 0.25 if source_type == 18 else 0.125 if source_type == 22 else 1
    actual = words.astype(np.float32) * multipliers * d * factor
    expected = gguf.quants.dequantize(blocks, qtype).reshape(-1, 8, 32)
    np.testing.assert_array_equal(actual, expected)
