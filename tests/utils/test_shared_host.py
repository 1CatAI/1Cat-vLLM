# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import array
import fcntl
import mmap
import multiprocessing as mp
import os
import socket
import struct

import pytest

from vllm.utils.shared_host import SharedHostArena

pytestmark = pytest.mark.skipif(
    not hasattr(os, "memfd_create"), reason="shared host arena requires Linux"
)


def _child(connection):
    with connection, SharedHostArena.receive(connection) as arena:
        assert arena.buffer[:4] == b"seed"
        arena.buffer[4:8] = b"peer"
        connection.send(struct.pack("!QQ", arena.identity.device, arena.identity.inode))


def test_spawned_process_maps_same_object_and_updates_parent():
    context = mp.get_context("spawn")
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    with parent, child, SharedHostArena.create(4097) as arena:
        assert arena.size == 2 * mmap.PAGESIZE
        arena.buffer[:4] = b"seed"
        process = context.Process(target=_child, args=(child,))
        process.start()
        try:
            parent.settimeout(15)
            arena.send(parent)
            identity = struct.unpack("!QQ", parent.recv(16))
            assert identity == (arena.identity.device, arena.identity.inode)
            process.join(15)
            assert process.exitcode == 0
            assert arena.buffer[4:8] == b"peer"
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)


def test_size_seal_prevents_peer_truncating_a_live_mapping():
    with SharedHostArena.create(mmap.PAGESIZE) as arena:
        with pytest.raises(OSError):
            os.ftruncate(arena._fd, 0)
        with pytest.raises(OSError):
            os.ftruncate(arena._fd, arena.size * 2)
        # The pool remains writable because GPU evictions write into it.
        arena.buffer[0] = 23
        assert arena.buffer[0] == 23


def test_close_rejects_live_export_and_can_be_retried():
    arena = SharedHostArena.create(4096)
    view = memoryview(arena.buffer)
    with pytest.raises(BufferError):
        arena.close()
    view.release()
    arena.close()
    arena.close()
    with (
        socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection,
        pytest.raises(ValueError, match="closed"),
    ):
        arena.send(connection)


@pytest.mark.parametrize("version,wrong_identity", [(99, False), (1, True)])
def test_invalid_handle_closes_received_descriptor(version, wrong_identity):
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    with parent, child, SharedHostArena.create(4096) as arena:
        before = len(os.listdir("/proc/self/fd"))
        parent.sendmsg(
            [
                struct.pack(
                    "!IQQQ",
                    version,
                    arena.size,
                    arena.identity.device,
                    arena.identity.inode + int(wrong_identity),
                )
            ],
            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [arena._fd]))],
        )
        with pytest.raises(ValueError):
            SharedHostArena.receive(child)
        assert len(os.listdir("/proc/self/fd")) == before


def test_unsealed_file_is_rejected():
    fd = os.memfd_create("unsealed", os.MFD_ALLOW_SEALING)
    os.ftruncate(fd, mmap.PAGESIZE)
    with pytest.raises(ValueError, match="not sealed"):
        SharedHostArena(fd)
    with pytest.raises(OSError):
        fcntl.fcntl(fd, fcntl.F_GETFD)


def test_stream_socket_is_not_accepted_as_atomic_handle_transport():
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    with (
        parent,
        child,
        SharedHostArena.create(4096) as arena,
        pytest.raises(ValueError, match="SEQPACKET"),
    ):
        arena.send(parent)


def _registration_child(connection):
    with connection, SharedHostArena.receive(connection) as arena:
        connection.send(b"ready")
        with arena.registration_guard():
            arena.buffer[0] = 71
            connection.send(b"acquired")


def test_registration_record_lock_excludes_received_descriptor_process():
    context = mp.get_context("spawn")
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    with parent, child, SharedHostArena.create(mmap.PAGESIZE) as arena:
        process = context.Process(target=_registration_child, args=(child,))
        process.start()
        try:
            parent.settimeout(15)
            with arena.registration_guard():
                arena.send(parent)
                assert parent.recv(16) == b"ready"
                parent.settimeout(0.1)
                with pytest.raises(TimeoutError):
                    parent.recv(16)
                assert arena.buffer[0] == 0
            parent.settimeout(15)
            assert parent.recv(16) == b"acquired"
            process.join(15)
            assert process.exitcode == 0
            assert arena.buffer[0] == 71
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)


def test_closed_arena_cannot_take_registration_lock():
    arena = SharedHostArena.create(mmap.PAGESIZE)
    arena.close()
    with pytest.raises(ValueError, match="closed"), arena.registration_guard():
        pass
