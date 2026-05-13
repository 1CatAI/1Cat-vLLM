#!/usr/bin/env python3
"""Small authenticated reverse proxy for an OpenAI-compatible local server."""

from __future__ import annotations

import argparse
import http.client
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "OpenAIAuthProxy/0.1"
    target = urlparse("http://127.0.0.1:8000")
    api_key = ""

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        self._handle_proxy()

    def do_POST(self) -> None:
        self._handle_proxy()

    def do_HEAD(self) -> None:
        self._handle_proxy()

    def _handle_proxy(self) -> None:
        if not self._authorized():
            body = b'{"error":{"message":"Unauthorized","type":"auth_error"}}\n'
            self.send_response(401)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
            return

        body_len = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(body_len) if body_len else None
        target_path = self.path
        conn = http.client.HTTPConnection(
            self.target.hostname or "127.0.0.1",
            self.target.port or 80,
            timeout=3600,
        )
        headers = {}
        for key, value in self.headers.items():
            lower_key = key.lower()
            if lower_key in HOP_BY_HOP_HEADERS or lower_key == "host":
                continue
            if lower_key == "authorization":
                continue
            headers[key] = value
        headers["Host"] = self.target.netloc

        try:
            conn.request(self.command, target_path, body=body, headers=headers)
            response = conn.getresponse()
            response_headers = {
                key.lower(): value for key, value in response.getheaders()
            }
            is_stream = response_headers.get("content-type", "").startswith(
                "text/event-stream"
            )
            self.send_response(response.status, response.reason)
            self._send_cors_headers()
            for key, value in response.getheaders():
                lower_key = key.lower()
                if lower_key in HOP_BY_HOP_HEADERS:
                    continue
                if lower_key == "content-length" and is_stream:
                    continue
                self.send_header(key, value)
            if is_stream:
                self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            if self.command != "HEAD":
                if is_stream:
                    while True:
                        chunk = response.readline()
                        if not chunk:
                            break
                        self._write_http_chunk(chunk)
                        if chunk.strip() == b"data: [DONE]":
                            break
                    self._write_http_chunk(b"")
                else:
                    while True:
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
            self._force_close_client()
        except BrokenPipeError:
            raise
        except Exception as exc:
            body = (
                '{"error":{"message":"proxy upstream error: '
                + str(exc).replace('"', "'")
                + '","type":"proxy_error"}}\n'
            ).encode()
            try:
                self.send_response(502)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)
            except BrokenPipeError:
                pass
        finally:
            conn.close()

    def _authorized(self) -> bool:
        if not self.api_key:
            return False
        auth = self.headers.get("Authorization") or ""
        return auth == f"Bearer {self.api_key}"

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Authorization, Content-Type, OpenAI-Organization, OpenAI-Project",
        )
        self.send_header("Access-Control-Allow-Methods", "GET, POST, HEAD, OPTIONS")

    def _force_close_client(self) -> None:
        try:
            self.connection.shutdown(2)
        except OSError:
            pass
        try:
            self.connection.close()
        except OSError:
            pass

    def _write_http_chunk(self, chunk: bytes) -> None:
        if chunk:
            self.wfile.write(("%x\r\n" % len(chunk)).encode("ascii"))
            self.wfile.write(chunk)
            self.wfile.write(b"\r\n")
        else:
            self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=18000)
    parser.add_argument("--target", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--api-key-env",
        default="PUBLIC_VLLM_API_KEY",
        help="environment variable that contains the required bearer token",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is required")

    ProxyHandler.target = urlparse(args.target)
    ProxyHandler.api_key = api_key
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), ProxyHandler)
    print(
        "proxy listening on "
        f"{args.listen_host}:{args.listen_port}, target={args.target}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
