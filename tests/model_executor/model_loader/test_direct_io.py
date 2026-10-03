# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from vllm.model_executor.model_loader import direct_io
from vllm.model_executor.model_loader.default_loader import (
    _pipeline_stage_layer_range,
)
from vllm.model_executor.model_loader.direct_io import (
    decoder_layer_filter,
    direct_io_weights,
)
from vllm.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
)
from vllm.model_executor.models.interfaces import SupportsMultiModal


def _checkpoint(path) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    tensors = {
        "model.language_model.layers.0.w": torch.randn(33, 17),
        "model.language_model.layers.1.w": torch.randn(64, 64).half(),
        "model.language_model.layers.2.w": torch.randn(5, 7).bfloat16(),
        "layers.3.ffn.w": torch.randn(9, 11).to(torch.float8_e4m3fn),
        "layers.3.ffn.scale": torch.randn(3, 4).to(torch.float8_e8m0fnu),
        "mtp.layers.0.w": torch.randint(0, 255, (129,), dtype=torch.uint8),
        "model.visual.blocks.0.w": torch.randint(-5, 5, (7, 3), dtype=torch.int8),
        "model.language_model.embed_tokens.weight": torch.arange(12).reshape(3, 4),
        "empty": torch.empty(0, 4),
    }
    save_file(tensors, str(path))
    return tensors


def _reference(path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt") as f:
        return {name: f.get_tensor(name) for name in f.keys()}  # noqa: SIM118


def _assert_same(got: dict[str, torch.Tensor], want: dict[str, torch.Tensor]):
    assert got.keys() == want.keys()
    for name, tensor in want.items():
        assert got[name].dtype == tensor.dtype, name
        assert got[name].shape == tensor.shape, name
        assert torch.equal(got[name].view(torch.uint8), tensor.view(torch.uint8))


@pytest.mark.parametrize("small_runs", [False, True], ids=["one_run", "many_runs"])
def test_direct_io_matches_safe_open(tmp_path, monkeypatch, small_runs):
    path = tmp_path / "shard.safetensors"
    _checkpoint(path)
    if small_runs:
        # Force a separate O_DIRECT request for almost every tensor.
        monkeypatch.setattr(direct_io, "_MAX_RUN", 1)
        monkeypatch.setattr(direct_io, "_MAX_GAP", 0)

    got = dict(direct_io_weights(str(path), lambda name: True))

    _assert_same(got, _reference(path))


def test_direct_io_reads_only_kept_tensors(tmp_path):
    path = tmp_path / "shard.safetensors"
    _checkpoint(path)
    other_stage = decoder_layer_filter(1, 3)

    got = dict(direct_io_weights(str(path), lambda name: not other_stage(name)))

    owned_by_other_stages = {
        "model.language_model.layers.0.w",
        "layers.3.ffn.w",
        "layers.3.ffn.scale",
    }
    want = {
        name: tensor
        for name, tensor in _reference(path).items()
        if name not in owned_by_other_stages
    }
    _assert_same(got, want)


def test_decoder_layer_filter_leaves_other_stacks_alone():
    outside = decoder_layer_filter(4, 8)
    assert outside("layers.2.attn.w")
    assert outside("model.layers.9.mlp.w")
    assert outside("model.language_model.layers.8.w")
    assert not outside("model.language_model.layers.4.w")
    assert not outside("layers.7.ffn.w")
    # MTP heads and vision towers number their own layers.
    assert not outside("mtp.layers.0.w")
    assert not outside("mtp.0.layers.1.w")
    assert not outside("model.visual.layers.0.w")
    assert not outside("model.language_model.embed_tokens.weight")


class _Stack(nn.Module):
    def __init__(self, start: int, end: int, total: int):
        super().__init__()
        self.start_layer, self.end_layer = start, end
        self.layers = nn.ModuleList(nn.Identity() for _ in range(total))


@pytest.mark.parametrize(
    "start,end,expected", [(0, 10, None), (3, 7, (3, 7)), (0, 4, (0, 4))]
)
def test_pipeline_stage_layer_range(start, end, expected):
    model = nn.Sequential(_Stack(start, end, 10))
    assert _pipeline_stage_layer_range(model) == expected


def test_pipeline_stage_layer_range_without_decoder_stack():
    assert _pipeline_stage_layer_range(nn.Linear(2, 2)) is None


def _file_backed(tensor: torch.Tensor) -> bool:
    address = tensor.data_ptr()
    with open("/proc/self/maps") as mappings:
        for line in mappings:
            fields = line.split(maxsplit=5)
            start, end = (int(value, 16) for value in fields[0].split("-"))
            if start <= address < end:
                return (
                    len(fields) == 6
                    and fields[5].startswith("/")
                    and not (fields[5].startswith("/dev/zero"))
                )
    return False


def test_only_decoder_layers_are_read_directly(tmp_path):
    # Decoder layers go through O_DIRECT unless the model wants them mapped;
    # embeddings, vision and MTP tensors stay mapped. Their pages are released
    # after each one is consumed, which must not change the data read later.
    path = tmp_path / "shard.safetensors"
    _checkpoint(path)
    requested = {"model.language_model.layers.1.w"}
    mapped = requested | {
        "mtp.layers.0.w",
        "model.visual.blocks.0.w",
        "model.language_model.embed_tokens.weight",
    }

    got = dict(
        safetensors_weights_iterator(
            [str(path)],
            use_tqdm_on_load=False,
            safetensors_load_strategy="direct",
            map_weight=requested.__contains__,
        )
    )

    _assert_same(got, _reference(path))
    for name, tensor in got.items():
        if tensor.numel():
            assert _file_backed(tensor) == (name in mapped), name


def _sharing_rank(rank, init_file, path, out_dir, mode):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    open_fds: set[int] = set()
    real_open, real_close = direct_io.os.open, direct_io.os.close

    def tracked_open(*args):
        fd = real_open(*args)
        open_fds.add(fd)
        return fd

    def tracked_close(fd):
        open_fds.discard(fd)
        real_close(fd)

    direct_io.os.open = tracked_open
    direct_io.os.close = tracked_close
    if rank == 1:
        # The follower receives every run and must not read the disk.
        def no_disk(*args):
            raise AssertionError("the follower read the disk")

        direct_io._read_run = no_disk
        if mode == "follower_cannot_buffer":

            def no_memory(*args):
                raise OSError(12, "Cannot allocate memory")

            direct_io._run_buffer = no_memory
    elif mode == "leader_cannot_open":

        def no_open(*args):
            raise OSError(22, "O_DIRECT not supported")

        direct_io.os.open = no_open
    elif mode == "leader_cannot_read":

        def eof(*args):
            raise EOFError("truncated shard")

        direct_io._read_run = eof
    sharing = direct_io.RunSharing(dist.group.WORLD, 0, rank == 0)
    keep = (
        (lambda name: "mtp" in name)
        if mode == "different_tensors" and rank == 1
        else (lambda name: True)
    )
    try:
        got = {
            name: tensor.clone()
            for name, tensor in direct_io_weights(path, keep, sharing)
        }
        torch.save(got, f"{out_dir}/rank{rank}.pt")
    except RuntimeError as error:
        torch.save({"error": str(error)}, f"{out_dir}/rank{rank}.pt")
    finally:
        torch.save(len(open_fds), f"{out_dir}/open_fds{rank}.pt")
        dist.destroy_process_group()


def _run_sharing(tmp_path, mode):
    path = tmp_path / "shard.safetensors"
    _checkpoint(path)
    mp.spawn(
        _sharing_rank,
        args=(str(tmp_path / "init"), str(path), str(tmp_path), mode),
        nprocs=2,
    )
    for rank in (0, 1):
        assert torch.load(tmp_path / f"open_fds{rank}.pt") == 0, "leaked shard fd"
    return path, [torch.load(tmp_path / f"rank{rank}.pt") for rank in (0, 1)]


def test_run_sharing_reads_once_per_group(tmp_path):
    path, results = _run_sharing(tmp_path, "all")
    for result in results:
        _assert_same(result, _reference(path))


@pytest.mark.parametrize(
    "mode,message",
    [
        ("different_tensors", "select different tensors"),
        ("leader_cannot_open", "select different tensors"),
        ("leader_cannot_read", "could not read or buffer"),
        ("follower_cannot_buffer", "could not read or buffer"),
    ],
)
def test_run_sharing_fails_together(tmp_path, mode, message):
    # No rank may be left waiting for a broadcast that never comes.
    _, results = _run_sharing(tmp_path, mode)
    for result in results:
        assert message in result["error"]


def test_run_of_only_empty_tensors_at_an_aligned_offset(tmp_path):
    path = tmp_path / "shard.safetensors"
    save_file(
        {
            "model.layers.0.a": torch.zeros(4096 - 80, dtype=torch.uint8),
            "model.layers.0.e": torch.empty(0, 4),
        },
        str(path),
    )

    got = dict(direct_io_weights(str(path), lambda name: name.endswith(".e")))

    assert got["model.layers.0.e"].shape == (0, 4)


def test_release_keeps_a_neighbours_in_place_write(tmp_path):
    # Only whole pages inside a released tensor are dropped: a small tensor
    # sharing a page with its neighbour keeps what the consumer wrote to it.
    path = tmp_path / "shard.safetensors"
    save_file(
        {"a.w": torch.zeros(16), "b.w": torch.arange(4096, dtype=torch.float32)},
        str(path),
    )
    kept = {}
    for name, tensor in direct_io.released_mapped_weights(str(path), lambda n: True):
        if name == "a.w":
            tensor[3] = 7.0
        kept[name] = tensor

    assert kept["a.w"][3] == 7.0
    assert torch.equal(kept["b.w"], torch.arange(4096, dtype=torch.float32))


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.start_layer, self.end_layer = 16, 32
        self.layers = nn.ModuleList(nn.Identity() for _ in range(32))


class _Multimodal(nn.Module, SupportsMultiModal):
    def __init__(self):
        super().__init__()
        self.audio_tower = _Encoder()
        self.language_model = nn.Sequential(_Stack(14, 28, 28))

    def get_language_model(self):
        return self.language_model


def test_stage_range_comes_from_the_language_model():
    # An encoder tower registered first carries its own pipeline range.
    assert _pipeline_stage_layer_range(_Multimodal()) == (14, 28)


def test_ambiguous_stage_ranges_raise():
    model = nn.Sequential(_Encoder(), _Stack(14, 28, 28))
    with pytest.raises(ValueError, match="several pipeline-partial"):
        _pipeline_stage_layer_range(model)


@pytest.mark.parametrize("failure", [None, "storage", "dtype"])
def test_auto_direct_admission_preserves_checkpoint_bytes(
    tmp_path, monkeypatch, failure
):
    from vllm.model_executor.model_loader import weight_utils

    path = tmp_path / "auto.safetensors"
    _checkpoint(path)
    monkeypatch.setattr(weight_utils, "_get_available_ram_bytes", lambda: 0)
    monkeypatch.setattr(weight_utils, "_get_fs_type", lambda files: "ext4")
    if failure == "storage":
        monkeypatch.setattr(
            weight_utils,
            "direct_io_capability",
            lambda path: (False, "storage does not support aligned reads"),
        )
    if failure == "dtype":
        read_header = direct_io._read_header

        def unsupported(fd):
            header, start = read_header(fd)
            header["layers.3.ffn.w"]["dtype"] = "UNSUPPORTED"
            return header, start

        monkeypatch.setattr(direct_io, "_read_header", unsupported)
    got = dict(safetensors_weights_iterator([str(path)], use_tqdm_on_load=False))
    _assert_same(got, _reference(path))
    assert _file_backed(got["layers.3.ffn.w"]) is (failure is not None)


def test_explicit_lazy_does_not_probe_or_enable_direct_io(tmp_path, monkeypatch):
    from vllm.model_executor.model_loader import weight_utils

    path = tmp_path / "lazy.safetensors"
    _checkpoint(path)
    monkeypatch.setattr(weight_utils, "_get_available_ram_bytes", lambda: 0)

    def unexpected(path):
        raise AssertionError("explicit lazy loading must retain its precedence")

    monkeypatch.setattr(weight_utils, "direct_io_capability", unexpected)
    got = dict(
        safetensors_weights_iterator(
            [str(path)], use_tqdm_on_load=False, safetensors_load_strategy="lazy"
        )
    )
    _assert_same(got, _reference(path))
    assert _file_backed(got["layers.3.ffn.w"])


def test_cgroup_parent_pressure_limits_are_respected(tmp_path):
    root = tmp_path / "groups"
    child = root / "child"
    child.mkdir(parents=True)
    membership = tmp_path / "membership"
    membership.write_text("0::/child\n")
    for node, used, maximum, high in (
        (root, 50, 256, 128),
        (child, 10, 512, 256),
    ):
        (node / "memory.current").write_text(str(used))
        (node / "memory.max").write_text(str(maximum))
        (node / "memory.high").write_text(str(high))
    assert direct_io.cgroup_available_bytes(root, membership) == 78
    (root / "memory.high").write_text("max")
    assert direct_io.cgroup_available_bytes(root, membership) == 206
    (root / "memory.current").write_text("300")
    assert direct_io.cgroup_available_bytes(root, membership) == 0


def _auto_rank(rank, init_file, path, out_dir):
    import json

    from vllm.model_executor.model_loader import weight_utils

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    weight_utils._get_available_ram_bytes = lambda: 0
    weight_utils._get_fs_type = lambda files: "ext4"
    weight_utils.direct_io_capability = lambda path: (rank == 0, "unsupported storage")
    weight_utils._direct_io_sharing = lambda ids: direct_io.RunSharing(
        dist.group.WORLD, 0, rank == 0
    )
    try:
        got = dict(
            weight_utils.safetensors_weights_iterator([path], use_tqdm_on_load=False)
        )
        _assert_same(got, _reference(path))
        with open(f"{out_dir}/{rank}.json", "w") as result:
            json.dump({"mapped": _file_backed(got["layers.3.ffn.w"])}, result)
    finally:
        dist.destroy_process_group()


def test_auto_direct_requires_tensor_parallel_consensus(tmp_path):
    import json

    path = tmp_path / "shared.safetensors"
    _checkpoint(path)
    mp.spawn(
        _auto_rank, args=(str(tmp_path / "init"), str(path), str(tmp_path)), nprocs=2
    )
    for rank in range(2):
        assert json.loads((tmp_path / f"{rank}.json").read_text())["mapped"]
