# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal TCP forwarder: 0.0.0.0:LISTEN -> TARGET_HOST:TARGET_PORT (API exposure)."""

import asyncio
import sys

LISTEN, HOST, PORT = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])


async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        writer.close()


async def handle(client_reader, client_writer):
    try:
        server_reader, server_writer = await asyncio.open_connection(HOST, PORT)
    except Exception:
        client_writer.close()
        return
    await asyncio.gather(
        pipe(client_reader, server_writer), pipe(server_reader, client_writer)
    )


async def main():
    server = await asyncio.start_server(
        handle, sys.argv[4] if len(sys.argv) > 4 else "127.0.0.1", LISTEN
    )
    async with server:
        await server.serve_forever()


asyncio.run(main())
