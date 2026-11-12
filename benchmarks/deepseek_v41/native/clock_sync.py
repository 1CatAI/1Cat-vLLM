# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Clock offsets of the other nodes relative to this one (IPoIB TCP ping-pong).

server: clock_sync.py serve PORT ; client: clock_sync.py probe HOST PORT N
offset = remote_time - local_midpoint, from the minimum-RTT exchange.
"""

import socket
import struct
import sys
import time

if sys.argv[1] == "serve":
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", int(sys.argv[2])))
    s.listen(1)
    c, _ = s.accept()
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    while True:
        b = c.recv(8)
        if len(b) < 8:
            break
        c.sendall(struct.pack("<q", time.time_ns()))
else:
    host, port, n = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    c = socket.create_connection((host, port))
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    best = None
    for _ in range(n):
        t0 = time.time_ns()
        c.sendall(b"x" * 8)
        r = struct.unpack("<q", c.recv(8))[0]
        t1 = time.time_ns()
        rtt = t1 - t0
        if best is None or rtt < best[0]:
            best = (rtt, r - (t0 + t1) // 2)
    print(f"{host} rtt_min_us={best[0] / 1e3:.1f} offset_us={best[1] / 1e3:.1f}")
