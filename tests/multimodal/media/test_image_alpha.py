# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any

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
@pytest.mark.parametrize("mode", ["RGBA", "LA", "P", "RGB", "L"])
@pytest.mark.parametrize("background", [(255, 255, 255), (12, 34, 56)])
def test_alpha_matches_qwen_white_composite(
    mode, background, source, tmp_path, image_http
):
    image = Image.new(mode, (3, 1))
    options: dict[str, Any] = {}
    if mode == "RGBA":
        image.putdata([(20, 40, 60, a) for a in (0, 128, 255)])
    elif mode == "LA":
        image.putdata([(40, a) for a in (0, 128, 255)])
    elif mode == "P":
        image.putpalette([20, 40, 60, 70, 80, 90, 100, 110, 120] + [0] * 759)
        image.putdata([0, 1, 2])
        options["transparency"] = bytes([0, 128, 255])
    elif mode == "RGB":
        image.putdata([(20, 40, 60), (70, 80, 90), (100, 110, 120)])
        options["transparency"] = (20, 40, 60)
    else:
        image.putdata([20, 40, 60])
        options["transparency"] = 20
    stream = BytesIO()
    image.save(stream, "PNG", **options)
    with Image.open(BytesIO(stream.getvalue())) as decoded:
        expected = Image.alpha_composite(
            Image.new("RGBA", decoded.size, (*background, 255)),
            decoded.convert("RGBA"),
        ).convert("RGB")
    actual = load_image(stream.getvalue(), source, tmp_path, image_http, background)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


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
