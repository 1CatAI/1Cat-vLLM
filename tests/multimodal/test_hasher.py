# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import struct
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.image import normalize_image
from vllm.multimodal.media import ImageMediaIO, MediaWithBytes

pytestmark = pytest.mark.cpu_test

ASSETS_DIR = Path(__file__).parent / "assets"
assert ASSETS_DIR.exists()


def test_hash_single_item_different_shape():
    x1 = torch.zeros(())
    x2 = torch.zeros((1,))

    hasher = MultiModalHasher
    assert hasher.hash_kwargs(x=x1) != hasher.hash_kwargs(x=x2)


def test_hash_key_order_invariant():
    x = torch.zeros((5, 10))
    y = torch.ones((5, 10))

    hasher = MultiModalHasher
    assert hasher.hash_kwargs(x=x, y=y) == hasher.hash_kwargs(y=y, x=x)


# NOTE: Images that are the same visually are allowed to have the same hash
@pytest.mark.parametrize("mode_pair", [("1", "L"), ("RGBA", "CMYK")])
def test_hash_collision_image_mode(mode_pair):
    mode1, mode2 = mode_pair
    image1 = Image.new(mode1, size=(10, 10), color=1)
    image2 = Image.new(mode2, size=(10, 10), color=1)

    hasher = MultiModalHasher
    assert hasher.hash_kwargs(image=image1) != hasher.hash_kwargs(image=image2)


def test_hash_collision_image_palette():
    # These images differ only in Image.palette._palette
    image1 = Image.open(ASSETS_DIR / "image1.png")
    image2 = Image.open(ASSETS_DIR / "image2.png")

    hasher = MultiModalHasher
    assert hasher.hash_kwargs(image=image1) != hasher.hash_kwargs(image=image2)


def test_hash_collision_image_transpose():
    image1 = Image.new("1", size=(10, 20))
    ImageDraw.Draw(image1).line([(0, 0), (10, 0)])

    image2 = Image.new("1", size=(20, 10))
    ImageDraw.Draw(image2).line([(0, 0), (0, 10)])

    hasher = MultiModalHasher
    assert hasher.hash_kwargs(image=image1) != hasher.hash_kwargs(image=image2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hash_collision_tensor_shape(dtype):
    # The hash should be different though the data is the same when flattened
    arr1 = torch.zeros((5, 10, 20, 3), dtype=dtype)
    arr2 = torch.zeros((10, 20, 5, 3), dtype=dtype)

    hasher = MultiModalHasher
    assert hasher.hash_kwargs(data=arr1) != hasher.hash_kwargs(data=arr2)


def test_hash_collision_array_shape():
    # The hash should be different though the data is the same when flattened
    arr1 = np.zeros((5, 10, 20, 3))
    arr2 = np.zeros((10, 20, 5, 3))

    hasher = MultiModalHasher
    assert hasher.hash_kwargs(data=arr1) != hasher.hash_kwargs(data=arr2)


def test_hash_non_contiguous_array():
    arr = np.arange(24).reshape(4, 6).T
    assert not arr.flags.c_contiguous

    arr_c = np.ascontiguousarray(arr)
    assert arr_c.flags.c_contiguous

    hasher = MultiModalHasher
    # Both should be hashable and produce the same hashes
    assert hasher.hash_kwargs(data=arr) == hasher.hash_kwargs(data=arr_c)


def test_hash_image_exif_id():
    # Test that EXIF ImageId tag can be used to store UUID
    # and the hasher will use that instead of the image data.
    image1 = image2 = Image.new("1", size=(10, 20))
    id = uuid.uuid4()
    image1.getexif()[Image.ExifTags.Base.ImageID] = id
    image2 = Image.open(ASSETS_DIR / "image1.png")
    image2.getexif()[Image.ExifTags.Base.ImageID] = "Not a UUID"
    image2a = Image.open(ASSETS_DIR / "image1.png")

    hasher = MultiModalHasher
    # first image has UUID in ImageID, so it should hash to that UUID
    assert hasher.hash_kwargs(image=image1) == hasher.hash_kwargs(image=id.bytes)
    # second image has non-UUID in ImageID, so it should hash to the image data
    assert hasher.hash_kwargs(image=image2) == hasher.hash_kwargs(image=image2a)


def test_hash_image_malformed_exif():
    # Test that images with malformed EXIF headers (e.g. invalid TIFF header)
    # do not raise an unhandled exception during hashing and fall back to image data.
    buf = BytesIO()
    Image.new("RGB", (64, 48)).save(buf, "JPEG")
    jpg = buf.getvalue()
    rest = jpg[2:]
    rest = rest[2 + struct.unpack(">H", rest[2:4])[0] :]
    payload = b"Exif\x00\x00XXXX\x00\x00\x00\x08" + bytes(32)
    data = b"\xff\xd8\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload + rest

    image = normalize_image(Image.open(BytesIO(data)))
    image.load()

    hasher = MultiModalHasher
    # Should hash without raising SyntaxError or any other exception
    hash_val = hasher.hash_kwargs(image=image)
    assert isinstance(hash_val, str) and len(hash_val) > 0

    # Also verify MediaWithBytes wrapping the image with malformed EXIF
    media_item = MediaWithBytes(image, data)
    hash_media = hasher.hash_kwargs(image=media_item)
    assert isinstance(hash_media, str) and len(hash_media) > 0


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "P"])
def test_hash_loaded_png_with_malformed_exif(mode):
    """The chat API path (load_bytes, then hashing) survives a PNG whose eXIf
    chunk has no valid TIFF header, whatever mode conversion it takes, and
    hashes the original bytes."""
    buf = BytesIO()
    Image.new(mode, (16, 8)).save(buf, "PNG", exif=b"Exif\x00\x00" + bytes(8))
    data = buf.getvalue()

    media = ImageMediaIO().load_bytes(data)

    # Same original bytes around an image without EXIF give the same hash.
    reference = MediaWithBytes(Image.new("RGB", (1, 1)), data)
    hasher = MultiModalHasher
    assert hasher.hash_kwargs(image=media) == hasher.hash_kwargs(image=reference)
