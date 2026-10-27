# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One Linux memory object mapped by several spawned worker processes.

Handles describe file identity and length, never another process's pointers.
CUDA registration is deliberately owned by each consumer, outside this class.
"""

import array
import ctypes
import fcntl
import mmap
import os
import socket
import struct
from contextlib import contextmanager
from dataclasses import dataclass

_HANDLE = struct.Struct("!IQQQ")
_VERSION = 1


@dataclass(frozen=True)
class SharedHostIdentity:
    size: int
    device: int
    inode: int


class SharedHostArena:
    """A writable MAP_SHARED memfd, with its size sealed for its lifetime.

    The caller must release exported buffers and CUDA registrations before
    closing the mapping. A failed close leaves the object open for retry.
    """

    def __init__(self, fd: int, expected: SharedHostIdentity | None = None):
        self._fd = fd
        try:
            info = os.fstat(fd)
            self.identity = SharedHostIdentity(info.st_size, info.st_dev, info.st_ino)
            if expected is not None and self.identity != expected:
                raise ValueError("Shared host file identity does not match its handle")
            if info.st_size <= 0 or info.st_size % mmap.PAGESIZE:
                raise ValueError("Shared host arena must have a page-aligned size")
            required = fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK
            if fcntl.fcntl(fd, fcntl.F_GET_SEALS) & required != required:
                raise ValueError("Shared host arena size is not sealed")
            self.buffer = mmap.mmap(
                fd,
                info.st_size,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        except BaseException:
            os.close(fd)
            self._fd = -1
            raise

    @classmethod
    def create(cls, size: int, name: str = "vllm-shared-host") -> "SharedHostArena":
        if size <= 0:
            raise ValueError("Shared host arena size must be positive")
        if not hasattr(os, "memfd_create"):
            raise RuntimeError("Shared host arenas require Linux memfd_create")
        size = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
        fd = os.memfd_create(name, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        try:
            os.ftruncate(fd, size)
            fcntl.fcntl(
                fd,
                fcntl.F_ADD_SEALS,
                fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL,
            )
        except BaseException:
            os.close(fd)
            raise
        return cls(fd)

    @property
    def size(self) -> int:
        return self.identity.size

    @property
    def address(self) -> int:
        """This process's CPU address, not a transferable handle."""
        return ctypes.addressof(ctypes.c_char.from_buffer(self.buffer))

    @contextmanager
    def registration_guard(self):
        """Serialize startup registration of this object across processes.

        SCM_RIGHTS descriptors share an open file description, so flock
        would treat them as the same owner. POSIX record locks instead have
        process ownership and actually exclude the other spawned ranks.
        This guard must not span an interprocess collective or a GPU wait.
        """
        if self._fd < 0:
            raise ValueError("Shared host arena is closed")
        fcntl.lockf(self._fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.lockf(self._fd, fcntl.LOCK_UN)

    def send(self, connection: socket.socket) -> None:
        self._check_socket(connection)
        if self._fd < 0:
            raise ValueError("Shared host arena is closed")
        handle = _HANDLE.pack(
            _VERSION, self.size, self.identity.device, self.identity.inode
        )
        sent = connection.sendmsg(
            [handle],
            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [self._fd]))],
        )
        if sent != len(handle):
            raise OSError("Incomplete shared host handle transfer")

    @classmethod
    def receive(cls, connection: socket.socket) -> "SharedHostArena":
        cls._check_socket(connection)
        item_size = array.array("i").itemsize
        data, messages, flags, _ = connection.recvmsg(
            _HANDLE.size,
            socket.CMSG_SPACE(item_size),
            socket.MSG_CMSG_CLOEXEC,
        )
        descriptors: list[int] = []
        for level, kind, value in messages:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                received = array.array("i")
                received.frombytes(value[: len(value) // item_size * item_size])
                descriptors.extend(received)
        try:
            if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                raise ValueError("Truncated shared host handle")
            if len(data) != _HANDLE.size or len(descriptors) != 1:
                raise ValueError("Expected exactly one shared host handle")
            version, size, device, inode = _HANDLE.unpack(data)
            if version != _VERSION:
                raise ValueError("Unsupported shared host handle version")
        except BaseException:
            for fd in descriptors:
                os.close(fd)
            raise
        # __init__ takes ownership of the received descriptor, including failure.
        return cls(descriptors[0], SharedHostIdentity(size, device, inode))

    @staticmethod
    def _check_socket(connection: socket.socket) -> None:
        if (
            connection.family != socket.AF_UNIX
            or connection.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
            != socket.SOCK_SEQPACKET
        ):
            raise ValueError("Shared host handles require a UNIX SEQPACKET socket")

    def close(self) -> None:
        if self._fd >= 0:
            self.buffer.close()
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> "SharedHostArena":
        return self

    def __exit__(self, *_):
        self.close()
