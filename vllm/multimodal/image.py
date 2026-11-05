# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib

from PIL import Image, ImageOps


def normalize_image(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation, tolerating malformed optional metadata."""
    with contextlib.suppress(Exception):
        image = ImageOps.exif_transpose(image)
    return image


def rescale_image_size(
    image: Image.Image, size_factor: float, transpose: int = -1
) -> Image.Image:
    """Rescale the dimensions of an image by a constant factor."""
    new_width = int(image.width * size_factor)
    new_height = int(image.height * size_factor)
    image = image.resize((new_width, new_height))
    if transpose >= 0:
        image = image.transpose(Image.Transpose(transpose))
    return image


def rgba_to_rgb(
    image: Image.Image,
    background_color: tuple[int, int, int] | list[int] = (255, 255, 255),
) -> Image.Image:
    """Convert an RGBA image to RGB with filled background color."""
    assert image.mode == "RGBA"
    converted = Image.new("RGB", image.size, background_color)
    converted.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return converted


def convert_image_mode(
    image: Image.Image,
    to_mode: str,
    background_color: tuple[int, int, int] | list[int] = (255, 255, 255),
) -> Image.Image:
    # Test transparency before the same-mode shortcut: RGB PNGs can carry tRNS.
    if to_mode == "RGB" and (
        image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info
    ):
        return rgba_to_rgb(image.convert("RGBA"), background_color)
    if image.mode == to_mode:
        return image
    return image.convert(to_mode)
