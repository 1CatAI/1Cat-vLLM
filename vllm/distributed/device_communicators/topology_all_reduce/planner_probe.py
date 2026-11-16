# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Driver-only topology discovery for the general planner; no CUDA contexts."""

import subprocess

import regex as re

from .collective_planner import Topology, from_probe


def matrix(text):
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    rows = {}
    header = None
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        gpu = [int(v[3:]) for v in fields if re.fullmatch(r"GPU\d+", v)]
        if len(gpu) > 1 or len(gpu) == 1 and len(fields) == 1:
            header = gpu
            continue
        if re.fullmatch(r"GPU\d+", fields[0]) and len(fields) > 1:
            rows[int(fields[0][3:])] = fields[1:]
    nodes = header or list(rows)
    if not 1 <= len(nodes) <= 64 or set(nodes) != set(rows):
        raise ValueError("Unknown/truncated physical GPU matrix")
    if any(len(rows[node]) < len(nodes) for node in nodes):
        raise ValueError("Truncated GPU row")
    return nodes, {(a, b): rows[a][j] for a in nodes for j, b in enumerate(nodes)}


def from_text(topology, read, write, hardware="", profiles=None):
    nodes, labels = matrix(topology)
    rnodes, reads = matrix(read)
    wnodes, writes = matrix(write)
    if nodes != rnodes or nodes != wnodes:
        raise ValueError("Driver matrices disagree")
    normalized = "\t" + "\t".join(f"GPU{i}" for i in range(len(nodes))) + "\n"
    normalized += "\n".join(
        "GPU" + str(i) + "\t" + "\t".join(labels[a, b] for b in nodes)
        for i, a in enumerate(nodes)
    )
    record = dict(
        topology=normalized,
        peer_read=[[reads[a, b] for b in nodes] for a in nodes],
        peer_write=[[writes[a, b] for b in nodes] for a in nodes],
    )
    converted = None
    if profiles:
        converted = dict(
            profiles,
            edges=[
                dict(e, source=nodes.index(e["source"]), dest=nodes.index(e["dest"]))
                for e in profiles.get("edges", [])
                if e["source"] in nodes and e["dest"] in nodes
            ],
        )
    base = from_probe(record, converted, identity=dict(hardware=hardware))
    edges = [
        dict(e, source=nodes[e["source"]], dest=nodes[e["dest"]]) for e in base.edges
    ]
    # Initial PCIe throughput follows negotiated generation/width, rather than
    # carrying an x8 Gen3 estimate onto every other machine.
    rates = {
        1: 250.0,
        2: 500.0,
        3: 984.6153846,
        4: 1969.230769,
        5: 3938.461538,
        6: 7562.5,
    }
    ports = {}
    for line in hardware.splitlines():
        fields = [v.strip() for v in line.split(",")]
        try:
            if len(fields) >= 6:
                ports[int(fields[0])] = rates[int(fields[-2])] * int(fields[-1]) * 0.74
        except (ValueError, KeyError):
            continue
    overrides = {(e["source"], e["dest"]): e for e in (profiles or {}).get("edges", [])}
    for edge in edges:
        a, b = edge["source"], edge["dest"]
        if (
            not edge["transport"].startswith("NV")
            and a in ports
            and b in ports
            and "bytes_per_us" not in overrides.get((a, b), {})
        ):
            edge["bytes_per_us"] = min(ports[a], ports[b])
    for i, node in enumerate(nodes):
        for key in (f"pcie-tx-{i}", f"pcie-rx-{i}"):
            if (
                node in ports
                and key in base.resources
                and key not in (profiles or {}).get("resources", {})
            ):
                base.resources[key]["bytes_per_us"] = ports[node]
    return Topology(
        nodes,
        edges,
        base.resources,
        dict(base.identity, driver_topology=topology, peer_read=read, peer_write=write),
    )


def discover(profiles=None):
    def query(*args):
        return subprocess.check_output(["nvidia-smi", *args], text=True, timeout=15)

    return from_text(
        query("topo", "-m"),
        query("topo", "-p2p", "r"),
        query("topo", "-p2p", "w"),
        query(
            "--query-gpu=index,uuid,pci.bus_id,driver_version,pcie.link.gen.current,pcie.link.width.current",
            "--format=csv,noheader",
        ),
        profiles,
    )


def refresh(previous):
    """Reprobe before using a cached decision; discard stale physical profiles."""
    live = discover()
    if previous is None:
        return live
    same_hardware = previous.identity.get("hardware") == live.identity["hardware"]
    explicit_unbound = (
        not previous.identity.get("hardware")
        and previous.identity.get("cost_origin") != "explicit-initial-estimates"
    )
    if live.labels == previous.labels and (same_hardware or explicit_unbound):
        # Preserve explicit current-link costs/capacity models, but incorporate
        # fresh driver graph and hardware identity in the selection key.
        identity = dict(previous.identity, **live.identity)
        identity["cost_origin"] = previous.identity.get(
            "cost_origin", live.identity["cost_origin"]
        )
        return Topology(live.nodes, previous.edges, previous.resources, identity)
    return live


def visible_devices():
    """Map actual CUDA ordinals by UUID, including reordered UUID environments."""
    import torch

    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True,
        timeout=15,
    )
    by_uuid = {
        uuid.strip().lower().removeprefix("gpu-"): int(index)
        for index, uuid in (line.split(",", 1) for line in raw.splitlines())
    }
    result = []
    for logical in range(torch.accelerator.device_count()):
        uuid = (
            str(torch.cuda.get_device_properties(logical).uuid)
            .lower()
            .removeprefix("gpu-")
        )
        if uuid not in by_uuid:
            raise ValueError(
                "CUDA UUID does not map to a physical GPU; MIG requires its own backend"
            )
        result.append(by_uuid[uuid])
    return result
