# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import numpy as np
import pybase64
import pytest
from PIL import Image

from vllm.multimodal.media import ImageMediaIO, MediaConnector

pytestmark = pytest.mark.cpu_test


@pytest.fixture(scope="module")
def image_http():
    bodies: dict[str, bytes] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(bodies[self.path])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", bodies
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def load_image(data, source, tmp_path, image_http, background=(255, 255, 255)):
    if source == "bytes":
        return ImageMediaIO(rgba_background_color=background).load_bytes(data)
    connector = MediaConnector(
        allowed_local_media_path=str(tmp_path),
        media_io_kwargs={"image": {"rgba_background_color": background}},
    )
    if source == "file":
        path = tmp_path / "input.png"
        path.write_bytes(data)
        url = path.as_uri()
    elif source == "data":
        url = "data:image/png;base64," + pybase64.b64encode(data).decode()
    else:
        address, bodies = image_http
        bodies["/input.png"] = data
        url = address + "/input.png"
    return connector.fetch_image(url)


@pytest.mark.parametrize("source", ["bytes", "file", "data", "http"])
@pytest.mark.parametrize("orientation", range(2, 9))
def test_exif_pixels_and_dimensions(orientation, source, tmp_path, image_http):
    image = Image.fromarray(np.arange(18, dtype=np.uint8).reshape(2, 3, 3))
    operations = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }
    exif = Image.Exif()
    exif[274] = orientation
    stream = BytesIO()
    image.save(stream, "PNG", exif=exif)
    actual = load_image(stream.getvalue(), source, tmp_path, image_http)
    expected = image.transpose(operations[orientation])
    assert actual.size == expected.size
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert actual.original_bytes == stream.getvalue()
    assert actual.getexif().get(274, 1) == 1


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_ordinary_image_pixels_unchanged(mode):
    image = Image.new(mode, (7, 5), 255)
    stream = BytesIO()
    image.save(stream, "PNG")
    from vllm.multimodal.image import convert_image_mode

    expected = convert_image_mode(image, "RGB")
    np.testing.assert_array_equal(
        np.asarray(ImageMediaIO().load_bytes(stream.getvalue())), np.asarray(expected)
    )
