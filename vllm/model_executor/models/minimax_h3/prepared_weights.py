# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Leased, exact CPU weight snapshots built directly in their final storage."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import regex as re
import torch

from .residency import MMapHostWeights, PinnedModuleStager, set_tensor_storage

FORMAT = 1


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _targets(module):
    return dict(list(module.named_parameters()) + list(module.named_buffers()))


def _schema(module):
    return {
        name: {"shape": list(t.shape), "dtype": str(t.dtype)}
        for name, t in _targets(module).items()
    }


def checkpoint_identity(paths):
    """Track installed immutable artifacts; replacing/editing a file invalidates."""
    result = []
    for value in paths:
        root = Path(value).resolve()
        files = (
            [root]
            if root.is_file()
            else sorted(
                p for p in root.iterdir() if p.suffix in {".json", ".safetensors"}
            )
        )
        if not files:
            raise ValueError(f"No checkpoint files at {root}")
        for path in files:
            stat = path.stat()
            result.append(
                (
                    str(path),
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                )
            )
    return result


def preparation_key(paths, *, component, rank, world_size, options):
    source = Path(__file__).parent
    files = sorted(source.glob("*.py")) + [
        source.parents[1] / "parameter.py",
        source.parents[1] / "layers/linear.py",
        source.parents[1] / "layers/sm70_diffusion.py",
    ]
    source_hash = hashlib.sha256()
    for path in files:
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    value = {
        "format": FORMAT,
        "component": component,
        "rank": rank,
        "world_size": world_size,
        "files": checkpoint_identity(paths),
        "source": source_hash.hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": str(torch.get_default_dtype()),
        "options": options,
    }
    return hashlib.sha256(_json(value)).hexdigest()


class PreparedWeights:
    """Only a completed, checksummed entry can be attached by another worker.

    Each worker holds its entry lease for the lifetime of its host masters.
    Eviction never touches an active entry or original model artifacts. Builds
    use shared writable mappings; publication reattaches private mappings so
    adapters or accidental CPU writes cannot change the reusable snapshot.
    """

    def __init__(
        self,
        root,
        key,
        module,
        *,
        limit_bytes=128 * 2**30,
        reserve_bytes=2**30,
        progress=None,
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("Invalid prepared weight cache identity")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.entry = self.root / ("entry-" + key)
        self.key = key
        self.schema = _schema(module)
        self.limit = limit_bytes
        self.reserve = reserve_bytes
        self.progress = progress or (lambda *args: None)
        self.lease = (self.root / (key + ".lock")).open("a+b")
        self.files = set()
        self.maps = []
        self.owned = set()
        self.readonly = False
        self.bytes_reserved = 0
        self.fallback = MMapHostWeights(self.root.parent / "h3-host")

    @contextmanager
    def capacity(self):
        with (self.root / "capacity.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _entries(self):
        return [
            p
            for p in self.root.iterdir()
            if re.fullmatch(r"entry-[0-9a-f]{64}", p.name)
            and p.is_dir()
            and not p.is_symlink()
        ]

    def _reserved(self, path):
        try:
            value = json.loads((path / "reservation.json").read_bytes())["bytes"]
            if type(value) is not int or value < 0:
                raise ValueError("Invalid cache reservation")
            return value
        except (OSError, ValueError, KeyError, TypeError):
            return sum(p.stat().st_size for p in path.iterdir() if p.is_file())

    def _reservation(self):
        (self.entry / "reservation.json").write_bytes(
            _json({"bytes": self.bytes_reserved})
        )

    def _make_room(self, extra):
        def enough():
            return (
                sum(self._reserved(p) for p in self._entries()) + extra <= self.limit
                and shutil.disk_usage(self.root).free >= extra + self.reserve
            )

        for path in sorted(self._entries(), key=lambda p: p.stat().st_mtime_ns):
            if enough():
                return
            if path == self.entry:
                continue
            with (self.root / (path.name[6:] + ".lock")).open("a+b") as lease:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                shutil.rmtree(path)
        if not enough():
            raise RuntimeError(
                "Not enough space for prepared H3 weights; "
                "active caches and model files were retained"
            )

    def _digest(self, path, done, total):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 2**20):
                digest.update(chunk)
                done += len(chunk)
                self.progress("checking_prepared_weights", done, total, "bytes")
        return digest.hexdigest(), done

    def restore(self, module):
        fcntl.flock(self.lease, fcntl.LOCK_SH)
        if self._restore(module):
            return True
        fcntl.flock(self.lease, fcntl.LOCK_UN)
        fcntl.flock(self.lease, fcntl.LOCK_EX)
        # A previous builder may have published while we waited for the lease.
        if self._restore(module):
            fcntl.flock(self.lease, fcntl.LOCK_SH)
            return True
        with self.capacity():
            if self.entry.is_symlink():
                raise ValueError("Prepared weight entries must not be symlinks")
            if self.entry.exists():
                shutil.rmtree(self.entry)
            self.entry.mkdir(mode=0o700)
            self._reservation()
        return False

    def _restore(self, module, *, verify=True):
        try:
            if self.entry.is_symlink() or any(
                (self.entry / p).is_symlink() for p in ("ready.json", "manifest.json")
            ):
                return False
            ready = json.loads((self.entry / "ready.json").read_bytes())
            payload = (self.entry / "manifest.json").read_bytes()
            if hashlib.sha256(payload).hexdigest() != ready["sha256"]:
                return False
            manifest = json.loads(payload)
            if (
                manifest["format"] != FORMAT
                or manifest["key"] != self.key
                or manifest["schema"] != self.schema
            ):
                return False
            targets = _targets(module)
            names = set()
            maps, bindings = [], []
            done = 0
            total = sum(g["bytes"] for g in manifest["groups"])
            self.progress("checking_prepared_weights", 0, total, "bytes")
            for group in manifest["groups"]:
                size = group["bytes"]
                if not isinstance(size, int) or size < 0:
                    return False
                if size:
                    filename = group["file"]
                    if not re.fullmatch(r"weights-[a-z0-9_]+\.bin", filename):
                        return False
                    path = self.entry / filename
                    if path.is_symlink() or path.stat().st_size != size:
                        return False
                    if verify:
                        digest, done = self._digest(path, done, total)
                        if digest != group["sha256"]:
                            return False
                    raw = torch.from_file(
                        str(path), shared=False, size=size, dtype=torch.uint8
                    )
                else:
                    raw = torch.empty(0, dtype=torch.uint8)
                maps.append(raw)
                for binding in group["bindings"]:
                    name = binding["name"]
                    if name not in targets or name in names:
                        return False
                    target = targets[name]
                    dtype = target.dtype
                    if binding["dtype"] != str(dtype):
                        return False
                    shape, stride, offset = (
                        binding["shape"],
                        binding["stride"],
                        binding["offset"],
                    )
                    if (
                        len(shape) != len(stride)
                        or len(shape) > 16
                        or any(
                            not isinstance(x, int) or x < 0
                            for x in [*shape, *stride, offset]
                        )
                        or math.prod(shape) != target.numel()
                    ):
                        return False
                    extent = (
                        0
                        if not math.prod(shape)
                        else offset
                        + sum((d - 1) * s for d, s in zip(shape, stride))
                        + 1
                    )
                    if extent * dtype.itemsize > size:
                        return False
                    value = torch.empty(0, dtype=dtype).set_(
                        raw.untyped_storage(), offset, shape, stride
                    )
                    bindings.append((target, value))
                    names.add(name)
            if names != set(targets):
                return False
            # Do not partially bind a failed/corrupt cache entry.
            for target, value in bindings:
                set_tensor_storage(target, value)
            self.maps = maps
            self.owned = {
                (m.untyped_storage().data_ptr(), m.untyped_storage().nbytes())
                for m in maps
            }
            self.readonly = True
            os.utime(self.entry, None)
            self.progress("reusing_prepared_weights", total, total, "bytes")
            return True
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, OverflowError):
            return False

    def snapshot(self, source, *, preserve=True):
        storage = source.untyped_storage()
        if (storage.data_ptr(), storage.nbytes()) in self.owned:
            return source.detach()
        if self.readonly:
            # Ordinary LoRA sidecars are installed after the reusable base.
            return self.fallback.snapshot(source, preserve=preserve)
        if source.device.type != "cpu" or source.dtype != torch.uint8:
            raise ValueError("Prepared backing requires CPU storage bytes")
        filename = storage.filename
        if filename and Path(filename).parent == self.entry:
            return source.detach()
        if not source.numel():
            return source.detach()
        with self.capacity():
            self._make_room(source.numel())
            fd, filename = tempfile.mkstemp(
                prefix="weights-", suffix=".bin", dir=self.entry
            )
            try:
                os.posix_fallocate(fd, 0, source.numel())
                raw = torch.from_file(
                    filename, shared=True, size=source.numel(), dtype=torch.uint8
                )
                self.bytes_reserved += source.numel()
                self.files.add(Path(filename))
                self._reservation()
            except BaseException:
                Path(filename).unlink(missing_ok=True)
                raise
            finally:
                os.close(fd)
        if preserve:
            raw.copy_(source)
        return raw

    def release_unused(self, module):
        live = {
            Path(t.untyped_storage().filename)
            for t in _targets(module).values()
            if t.untyped_storage().filename
        }
        with self.capacity():
            for path in self.files - live:
                self.bytes_reserved -= path.stat().st_size
                path.unlink()
            self.files.intersection_update(live)
            self._reservation()

    def publish(self, module):
        if self.readonly:
            return
        PinnedModuleStager.map_cpu_weights(module, self)
        self.release_unused(module)
        groups = {}
        for name, tensor in _targets(module).items():
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key not in groups:
                groups[key] = {
                    "file": Path(storage.filename).name if storage.nbytes() else None,
                    "bytes": storage.nbytes(),
                    "bindings": [],
                }
            groups[key]["bindings"].append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "stride": list(tensor.stride()),
                    "offset": tensor.storage_offset(),
                    "dtype": str(tensor.dtype),
                }
            )
        done, total = 0, sum(g["bytes"] for g in groups.values())
        for group in groups.values():
            if group["bytes"]:
                path = self.entry / group["file"]
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
                group["sha256"], done = self._digest(path, done, total)
        payload = _json(
            {
                "format": FORMAT,
                "key": self.key,
                "schema": self.schema,
                "groups": list(groups.values()),
            }
        )
        for name, content in [
            ("manifest.json", payload),
            ("ready.json", _json({"sha256": hashlib.sha256(payload).hexdigest()})),
        ]:
            temporary = self.entry / (name + ".tmp")
            with temporary.open("wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.entry / name)
        fd = os.open(self.entry, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        if not self._restore(module, verify=False):
            raise RuntimeError("Published prepared weights could not be restored")
        fcntl.flock(self.lease, fcntl.LOCK_SH)

    def close(self):
        self.lease.close()
