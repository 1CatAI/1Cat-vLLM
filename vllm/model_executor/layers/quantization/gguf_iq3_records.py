# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless 13-bit signed-book indices, aligned16B records and a4B tail."""

import numpy as np


def signed_index_records(raw):
    n, rb = raw.shape
    nb = rb // 110
    b = raw.reshape(n // 32, 32, nb, 110).transpose(0, 2, 1, 3).copy()
    o = np.arange(32)
    h = b[..., 66 + o // 4]
    first = b[..., 2 + 2 * o].astype(np.uint32) | (
        ((h >> (2 * (o % 4))) & 1).astype(np.uint32) << 8
    )
    second = b[..., 3 + 2 * o].astype(np.uint32) | (
        ((h >> (2 * (o % 4) + 1)) & 1).astype(np.uint32) << 8
    )
    signs = b[..., 74:106].astype(np.uint32)
    ids = (
        np.stack(((first << 4) | (signs & 15), (second << 4) | (signs >> 4)), axis=-1)
        .reshape(n // 32, nb, 32, 2, 32)
        .transpose(0, 1, 3, 2, 4)
    )
    bits = ((ids[..., None] >> np.arange(13, dtype=np.uint32)) & 1).astype(np.uint8)
    packets = np.packbits(
        bits.reshape(n // 32, nb, 2, 32, 416), axis=-1, bitorder="little"
    )
    main = (
        packets[..., :48]
        .reshape(n // 32, nb, 2, 32, 3, 16)
        .transpose(0, 1, 2, 4, 3, 5)
        .copy()
        .reshape(n // 32, nb * 3072)
    )
    tail = packets[..., 48:].copy().reshape(n // 32, nb * 256)
    meta = np.concatenate(
        (
            b[..., :2].reshape(n // 32, nb, 64),
            b[..., 106:110].reshape(n // 32, nb, 128),
        ),
        axis=-1,
    ).reshape(n // 32, nb * 192)
    result = np.concatenate((main, tail, meta), axis=1).reshape(-1)
    assert result.nbytes == raw.nbytes
    # Inverse reads output, independently reconstructing every original byte.
    v = result.reshape(n // 32, nb * 3520)
    recovered_main = (
        v[:, : nb * 3072]
        .reshape(n // 32, nb, 2, 3, 32, 16)
        .transpose(0, 1, 2, 4, 3, 5)
        .reshape(n // 32, nb, 2, 32, 48)
    )
    recovered_tail = v[:, nb * 3072 : nb * 3328].reshape(n // 32, nb, 2, 32, 4)
    recovered = np.concatenate((recovered_main, recovered_tail), axis=-1)
    rb_bits = np.unpackbits(recovered, axis=-1, bitorder="little").reshape(
        n // 32, nb, 2, 32, 32, 13
    )
    ri = (
        (rb_bits.astype(np.uint32) << np.arange(13, dtype=np.uint32))
        .sum(axis=-1, dtype=np.uint32)
        .transpose(0, 1, 3, 2, 4)
        .reshape(n // 32, nb, 32, 32, 2)
    )
    idx = ri >> 4
    sg = ri & 15
    rec = np.empty_like(b)
    rec[..., 2:66] = (idx & 255).astype(np.uint8).reshape(n // 32, nb, 32, 64)
    rec[..., 74:106] = (sg[..., 0] | (sg[..., 1] << 4)).astype(np.uint8)
    high = (idx >> 8).reshape(n // 32, nb, 32, 8, 8)
    rec[..., 66:74] = (
        (high << np.arange(8, dtype=np.uint32))
        .sum(axis=-1, dtype=np.uint32)
        .astype(np.uint8)
    )
    rm = v[:, nb * 3328 :].reshape(n // 32, nb, 192)
    rec[..., :2] = rm[..., :64].reshape(n // 32, nb, 32, 2)
    rec[..., 106:110] = rm[..., 64:].reshape(n // 32, nb, 32, 4)
    assert np.array_equal(rec, b)
    return result
