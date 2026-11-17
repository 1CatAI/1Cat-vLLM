# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pillow input errors follow the client-error path for each image entry."""

from io import BytesIO

import pybase64 as base64
import pytest
from PIL import Image

from vllm.entrypoints.utils import create_error_response
from vllm.multimodal.media.image import ImageMediaIO


def _load(image_io, data, route, tmp_path):
    if route == "bytes":
        return image_io.load_bytes(data)
    if route == "base64":
        return image_io.load_base64("image/png", base64.b64encode(data).decode())
    path = tmp_path / "input.png"
    path.write_bytes(data)
    return image_io.load_file(path)


@pytest.mark.parametrize("route", ["bytes", "base64", "file"])
def test_broken_png_is_client_error(route, tmp_path):
    data = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000004000000040806000000a9f19e"
        "7e000000144944415478da63fccfc0c0c0c0c0c0c0c0c0c000000e0001ff02ab"
        "ce36890000000049454e44ae426082"
    )
    with pytest.raises(ValueError, match="Failed to load image") as caught:
        _load(ImageMediaIO(), data, route, tmp_path)
    assert isinstance(caught.value.__cause__, SyntaxError)
    assert create_error_response(caught.value).error.code == 400


@pytest.mark.parametrize("route", ["bytes", "base64", "file"])
def test_pillow_pixel_limit_is_client_error(route, tmp_path, monkeypatch):
    buffer = BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="PNG")
    # Pillow rejects above twice its limit; no large allocation is needed.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ValueError, match="Failed to load image") as caught:
        _load(ImageMediaIO(), buffer.getvalue(), route, tmp_path)
    assert isinstance(caught.value.__cause__, Image.DecompressionBombError)
    assert create_error_response(caught.value).error.code == 400


@pytest.mark.parametrize("route", ["bytes", "base64", "file"])
def test_valid_image_pixels_are_preserved(route, tmp_path):
    buffer = BytesIO()
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(buffer, format="PNG")
    result = _load(ImageMediaIO(), buffer.getvalue(), route, tmp_path).media
    assert result.mode == "RGB"
    assert result.size == (4, 4)
    assert list(result.getdata()) == [(1, 2, 3)] * 16
