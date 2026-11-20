from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Any, Iterable, Mapping

from program.nccl_traffic import per_tx_byte
from program.link_energy import resolve_rule, line_rate_bps

def _nonnegative(value: Any, name: str) -> float:
    parsed = float(value or 0.0)
    if not isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and >= 0 (got {value!r})")
    return parsed

def _normalized_topology(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "").replace("-", "")

@dataclass(frozen=True)
class CollectiveCommunicationEnergy:
    operation_id: str
    dimension_index: int
    collective_type: str
    collective_size_bytes: int
    transmitted_bytes: int
    chunks: int
    active_link_count: int
    serialization_time_ns: float
    network_transfers: int
    completion_time_ns: float
    link_dynamic_j: float
    active_network_j: float
    nccl_hbm_bytes: float = 0.0
    nccl_hbm_j: float = 0.0
    nccl_l2_bytes: float = 0.0
    nccl_l2_j: float = 0.0
    reduction_flops: float = 0.0
    reduction_j: float = 0.0
    astra_reduction_memory_bytes: float = 0.0

    custom_hbm_bytes: float = 0.0
    custom_hbm_j: float = 0.0
    custom_l2_bytes: float = 0.0
    custom_l2_j: float = 0.0
    custom_reduction_flops: float = 0.0
    custom_reduction_j: float = 0.0

    nccl_protocol = ""
    nccl_protocol_source = ""
    wire_byte_multiplier = 1.0
    nccl_l2_bypass_bytes = 0.0

    @property
    def nccl_memory_j(self) -> float:
        return self.nccl_hbm_j + self.nccl_l2_j + self.reduction_j

    @property
    def total_j(self) -> float:
        return self.link_dynamic_j + self.active_network_j + self.nccl_memory_j + self.custom_hbm_j + self.custom_l2_j + self.custom_reduction_j

    def scaled(self, factor: float) -> "CollectiveCommunicationEnergy":
        factor = _nonnegative(factor, "collective scale factor")
        suffix = "" if factor == 1.0 else f"*{factor:g}"
        return type(self)(
            operation_id=f"{self.operation_id}{suffix}",
            dimension_index=self.dimension_index,
            collective_type=self.collective_type,
            collective_size_bytes=self.collective_size_bytes,
            transmitted_bytes=self.transmitted_bytes * factor,
            chunks=self.chunks * factor,
            active_link_count=self.active_link_count,
            serialization_time_ns=self.serialization_time_ns * factor,
            network_transfers=self.network_transfers * factor,
            completion_time_ns=self.completion_time_ns * factor,
            link_dynamic_j=self.link_dynamic_j * factor,
            active_network_j=self.active_network_j * factor,
            nccl_hbm_bytes=self.nccl_hbm_bytes * factor,
            nccl_hbm_j=self.nccl_hbm_j * factor,
            nccl_l2_bytes=self.nccl_l2_bytes * factor,
            nccl_l2_j=self.nccl_l2_j * factor,
            reduction_flops=self.reduction_flops * factor,
            reduction_j=self.reduction_j * factor,
            astra_reduction_memory_bytes=self.astra_reduction_memory_bytes * factor,
            custom_hbm_bytes=self.custom_hbm_bytes * factor,
            custom_hbm_j=self.custom_hbm_j * factor,
            custom_l2_bytes=self.custom_l2_bytes * factor,
            custom_l2_j=self.custom_l2_j * factor,
            custom_reduction_flops=self.custom_reduction_flops * factor,
            custom_reduction_j=self.custom_reduction_j * factor,
            **({"nccl_l2_bypass_bytes": self.nccl_l2_bypass_bytes * factor} if self.nccl_l2_bypass_bytes else {}),
            **(dict(nccl_protocol=self.nccl_protocol,
                    nccl_protocol_source=self.nccl_protocol_source,
                    wire_byte_multiplier=self.wire_byte_multiplier) if self.nccl_protocol else {}),
        )

    def without_execution_repetition(self) -> "CollectiveCommunicationEnergy":
        operation_id, separator, suffix = self.operation_id.rpartition("*")
        if not separator:
            return self
        try:
            repetitions = float(suffix)
        except ValueError:
            return self
        if repetitions <= 0.0:
            return self
        return replace(self.scaled(1.0 / repetitions), operation_id=operation_id)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "operation_id": self.operation_id,
            "dimension_index": self.dimension_index,
            "collective_type": self.collective_type,
            "collective_size_bytes": self.collective_size_bytes,
            "transmitted_bytes": self.transmitted_bytes,
            "chunks": self.chunks,
            "active_link_count": self.active_link_count,
            "serialization_time_ns": self.serialization_time_ns,
            "network_transfers": self.network_transfers,
            "completion_time_ns": self.completion_time_ns,
            "link_dynamic_j": self.link_dynamic_j,
            "active_network_j": self.active_network_j,
            "nccl_hbm_bytes": self.nccl_hbm_bytes,
            "nccl_hbm_j": self.nccl_hbm_j,
            "nccl_l2_bytes": self.nccl_l2_bytes,
            "nccl_l2_j": self.nccl_l2_j,
            "reduction_flops": self.reduction_flops,
            "reduction_j": self.reduction_j,
            "astra_reduction_memory_bytes": self.astra_reduction_memory_bytes,
            "total_j": self.total_j,
        }
        if self.nccl_protocol:
            result.update(nccl_protocol=self.nccl_protocol,
                          nccl_protocol_source=self.nccl_protocol_source,
                          wire_byte_multiplier=self.wire_byte_multiplier)
        for name in ("custom_hbm_bytes", "custom_hbm_j", "custom_l2_bytes", "custom_l2_j", "custom_reduction_flops", "custom_reduction_j",
                     "nccl_l2_bypass_bytes"):
            if getattr(self, name):
                result[name] = getattr(self, name)
        return result

@dataclass(frozen=True)
class ProtocolCommunicationEnergy(CollectiveCommunicationEnergy):
    nccl_protocol: str = ""
    nccl_protocol_source: str = ""
    wire_byte_multiplier: float = 1.0

@dataclass(frozen=True)
class L2BypassCommunicationEnergy(CollectiveCommunicationEnergy):
    nccl_l2_bypass_bytes: float = 0.0

@dataclass(frozen=True)
class L2BypassProtocolCommunicationEnergy(ProtocolCommunicationEnergy):
    nccl_l2_bypass_bytes: float = 0.0

_L2_BYPASS_ROWS = {CollectiveCommunicationEnergy: L2BypassCommunicationEnergy,
                   ProtocolCommunicationEnergy: L2BypassProtocolCommunicationEnergy}

def collective_energy(cls, **values):
    if "nccl_l2_bypass_bytes" in values:
        cls = _L2_BYPASS_ROWS[cls]
    return cls(**values)

@dataclass(frozen=True)
class CommunicationEnergyResult:
    collectives: tuple[CollectiveCommunicationEnergy, ...] = ()

    @property
    def transmitted_bytes(self) -> int:
        return sum(item.transmitted_bytes for item in self.collectives)

    @property
    def link_dynamic_j(self) -> float:
        return sum(item.link_dynamic_j for item in self.collectives)

    @property
    def active_network_j(self) -> float:
        return sum(item.active_network_j for item in self.collectives)

    def _sum(self, name: str) -> float:
        return sum(getattr(item, name) for item in self.collectives)

    nccl_hbm_bytes = property(lambda self: self._sum("nccl_hbm_bytes"))
    nccl_hbm_j = property(lambda self: self._sum("nccl_hbm_j"))
    nccl_l2_bytes = property(lambda self: self._sum("nccl_l2_bytes"))
    nccl_l2_j = property(lambda self: self._sum("nccl_l2_j"))
    reduction_flops = property(lambda self: self._sum("reduction_flops"))
    reduction_j = property(lambda self: self._sum("reduction_j"))
    astra_reduction_memory_bytes = property(lambda self: self._sum("astra_reduction_memory_bytes"))
    custom_hbm_bytes = property(lambda self: self._sum("custom_hbm_bytes"))
    custom_hbm_j = property(lambda self: self._sum("custom_hbm_j"))
    custom_l2_bytes = property(lambda self: self._sum("custom_l2_bytes"))
    custom_l2_j = property(lambda self: self._sum("custom_l2_j"))
    custom_reduction_flops = property(lambda self: self._sum("custom_reduction_flops"))
    custom_reduction_j = property(lambda self: self._sum("custom_reduction_j"))
    nccl_l2_bypass_bytes = property(lambda self: self._sum("nccl_l2_bypass_bytes"))

    @property
    def nccl_memory_j(self) -> float:
        return self.nccl_hbm_j + self.nccl_l2_j + self.reduction_j

    @property
    def total_j(self) -> float:
        return self.link_dynamic_j + self.active_network_j + self.nccl_memory_j + self.custom_hbm_j + self.custom_l2_j + self.custom_reduction_j

    def scaled(self, factor: float) -> "CommunicationEnergyResult":
        factor = _nonnegative(factor, "communication scale factor")
        return CommunicationEnergyResult(
            tuple(item.scaled(factor) for item in self.collectives)
        )

    def without_execution_repetition(self) -> "CommunicationEnergyResult":
        return CommunicationEnergyResult(
            tuple(item.without_execution_repetition() for item in self.collectives)
        )

    def __add__(self, other: "CommunicationEnergyResult") -> "CommunicationEnergyResult":
        if not isinstance(other, CommunicationEnergyResult):
            return NotImplemented
        return CommunicationEnergyResult(self.collectives + other.collectives)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "energy_scope": "system",
            "transmitted_bytes": self.transmitted_bytes,
            "link_dynamic_j": self.link_dynamic_j,
            "active_network_j": self.active_network_j,
            "nccl_hbm_bytes": self.nccl_hbm_bytes,
            "nccl_hbm_j": self.nccl_hbm_j,
            "nccl_l2_bytes": self.nccl_l2_bytes,
            "nccl_l2_j": self.nccl_l2_j,
            "reduction_flops": self.reduction_flops,
            "reduction_j": self.reduction_j,
            "astra_reduction_memory_bytes": self.astra_reduction_memory_bytes,
            "total_j": self.total_j,
            "by_collective": [item.to_dict() for item in self.collectives],
        }
        for name in ("custom_hbm_bytes", "custom_hbm_j", "custom_l2_bytes", "custom_l2_j", "custom_reduction_flops", "custom_reduction_j",
                     "nccl_l2_bypass_bytes"):
            if getattr(self, name):
                result[name] = getattr(self, name)
        return result

def nccl_memory_energy(hw_config: Any, collective_type: str, participants: int, transmitted_bytes: float) -> dict:
    hbm_per, l2_per, flops_per = per_tx_byte(collective_type, participants)
    tech = hw_config.tech_config
    tx = float(transmitted_bytes)
    hbm, l2, flops = tx * hbm_per, tx * l2_per, tx * flops_per
    bypass = {}
    from sim_settings import l2_bypass
    if l2_bypass(hw_config):
        hbm, l2, bypass = l2, 0.0, {"nccl_l2_bypass_bytes": l2}
    return dict(
        nccl_hbm_bytes=hbm,
        nccl_hbm_j=hbm * _nonnegative(tech.DRAM.dynamic_energy_per_byte_value, "DRAM energy per byte"),
        nccl_l2_bytes=l2,
        nccl_l2_j=l2 * _nonnegative(tech.SRAML2.dynamic_energy_per_byte_value, "L2 energy per byte"),
        reduction_flops=flops,
        reduction_j=flops * _nonnegative(tech.core.vector_energy, "vector energy per FLOP"),
        **bypass,
    )

def link_dynamic_from_program_transfers(
    transfers: Iterable[Any], hw_config: Any, dimension_index: int, *, axis: str,
    durations: Mapping[int, float] | None = None,
) -> CommunicationEnergyResult:
    from program.ir import CollectiveType, TransferOp

    dimension = _dimension(hw_config, dimension_index)
    energy_per_bit = _nonnegative(
        getattr(dimension, "energy_per_bit", 0.0), "link energy per bit"
    )
    normalized_axis = str(axis).strip().lower()
    priced = []
    for op in transfers:
        if not isinstance(op, TransferOp):
            continue
        if (
            op.comm_type is not CollectiveType.PIPELINE
            or op.size_bytes <= 0
            or op.src_device == op.dst_device
            or str(op.interconnect or "").strip().lower() != normalized_axis
        ):
            continue
        rule = resolve_rule(hw_config, dimension)
        seconds = 0.0
        link_j = op.size_bytes * 8.0 * energy_per_bit
        if rule == "line_rate_while_active":
            if durations is None or op.uid not in durations:
                raise ValueError("line_rate_while_active requires the timing function's P2P duration")
            seconds = _nonnegative(durations[op.uid], "P2P active time")
            link_j = energy_per_bit * line_rate_bps(dimension) * seconds * 2
        priced.append(
            collective_energy(
                CollectiveCommunicationEnergy,
                operation_id=f"graph_link_dynamic_only:{op.uid}",
                dimension_index=dimension_index,
                collective_type="P2P",
                collective_size_bytes=op.size_bytes,
                transmitted_bytes=op.size_bytes,
                chunks=0,
                active_link_count=1,
                serialization_time_ns=0.0,
                network_transfers=1,
                completion_time_ns=seconds * 1e9,
                link_dynamic_j=link_j,
                active_network_j=0.0,
                **nccl_memory_energy(hw_config, "P2P", 2, op.size_bytes),
            )
        )
    return CommunicationEnergyResult(tuple(priced))

def protocol_energy_from_program(program: Any, hw_config: Any) -> CommunicationEnergyResult:
    from program.ir import CollectiveOp, ComputeOp
    from program.collective_launch import is_nccl_call
    from program.vllm_allreduce import op_choice
    from program.allreduce_curve import is_measured_call
    from program.nccl_latency import protocol_for_call, resolve_dimension
    from program.pipeline_collectives import FOLDED_SECONDS
    from timing_model import CollectiveType

    groups = {}
    supported = (CollectiveType.ALL_REDUCE, CollectiveType.ALL_GATHER, CollectiveType.REDUCE_SCATTER)
    for owner in program.ops:
        calls = (owner,) if isinstance(owner, CollectiveOp) else owner.embedded_collectives if isinstance(owner, ComputeOp) else ()
        for op in calls:
            if (is_measured_call(program, op) or op.coll not in supported or not is_nccl_call(op.coll, participants=op.participants,
                    size_bytes=op.size_bytes, algorithm=op_choice(program, op).algorithm)):
                continue
            label = getattr(op, 'label', None) or getattr(op, 'comm_key', None) or op.name
            key = (label, op.coll, op.interconnect, op.participants, op.size_bytes)
            if key not in groups:
                groups[key] = [op, 0]
            folded = isinstance(owner, CollectiveOp) and owner.uid in program.meta.misc.get(FOLDED_SECONDS, {})
            whole_group = folded or (isinstance(owner, ComputeOp) and not program.meta.misc.get('vtp_per_rank'))
            divisor = 1 if whole_group else op.participants
            groups[key][1] += max(1, int(program.dp_count)) / divisor
    rows = []
    for op, multiplicity in groups.values():
        model = protocol_for_call(program, op)
        _, dim = resolve_dimension(hw_config, op.interconnect)
        index = list(hw_config.network_layout.dimensions).index(dim)
        n = int(op.participants)
        steps = model["nsteps"]

        collective_bytes = float(op.size_bytes) * (n if op.coll is CollectiveType.ALL_GATHER else 1)
        payload = collective_bytes * steps
        wire = payload * model["wire_multiplier"]
        duration = model["latency_seconds"] + model["transfer_seconds"]
        bit_j = _nonnegative(getattr(dim, "energy_per_bit", 0), "link energy per bit")
        link_j = wire * 8 * bit_j
        if resolve_rule(hw_config, dim) == "line_rate_while_active":
            link_j = line_rate_bps(dim) * n * duration * bit_j
        system = _nonnegative(getattr(dim, "system_power_w", 0), "system network power")
        static = _nonnegative(getattr(dim, "static_power_per_link_w", 0), "static power per link")
        active_j = (system * min(1., n / max(1, int(dim.size))) if system else static * n) * duration
        row = collective_energy(
            ProtocolCommunicationEnergy,
            operation_id=f"nccl_protocol:{getattr(op, 'label', None) or getattr(op, 'comm_key', None) or op.name}",
            dimension_index=index, collective_type=op.coll.name,
            collective_size_bytes=op.size_bytes, transmitted_bytes=wire,
            chunks=0, active_link_count=n,
            serialization_time_ns=model["transfer_seconds"] * 1e9,
            network_transfers=steps, completion_time_ns=duration * 1e9,
            link_dynamic_j=link_j, active_network_j=active_j,
            nccl_protocol=model["protocol"], nccl_protocol_source=model["source"],
            wire_byte_multiplier=model["wire_multiplier"],
            **nccl_memory_energy(hw_config, op.coll.name, n, payload),
        )
        rows.append(row.scaled(multiplicity))
    return CommunicationEnergyResult(tuple(rows))

def _dimension(hw_config: Any, dimension_index: int) -> Any:
    dimensions = tuple(hw_config.network_layout.dimensions)
    if dimension_index < 0 or dimension_index >= len(dimensions):
        raise ValueError(
            f"AstraSim reported network dimension {dimension_index}, but the "
            f"hardware configuration defines {len(dimensions)} dimensions"
        )
    return dimensions[dimension_index]

_COLLECTIVE_OPS = {
    "ALL_REDUCE": "all-reduce",
    "ALL_GATHER": "all-gather",
    "REDUCE_SCATTER": "reduce-scatter",
    "ALL_TO_ALL": "all-to-all",
}

def _collective_algorithm(hw_config: Any, dimension: Any, collective_type: str) -> str:
    if (str(collective_type or "").upper() == "ALL_REDUCE" and getattr(hw_config, "vllm_nccl_ring", False)
            and "tp" in getattr(dimension, "parallelisms", ())):
        return "ring"
    op = _COLLECTIVE_OPS.get(str(collective_type or "").upper())
    if op is None:
        return ""
    from sim_settings import nccl_latency_rule
    if nccl_latency_rule(hw_config) in ("tuning_formula", "tuning_protocol"):
        from astrasim_lib.config_generation import _normalize_topology_name, resolve_collective_for_dimension

        return resolve_collective_for_dimension(
            hw_config, dimension, _normalize_topology_name(dimension.topology_type),
            op, "auto", int(getattr(dimension, "size", 1)),
            force_nccl_ring=True,
        )
    overrides = getattr(dimension, "collective_override", None) or {}
    override = overrides.get(op) or overrides.get(op.replace("-", "_"))
    if override:
        return str(override).lower()
    default = "auto"
    backend = getattr(getattr(hw_config, "execution_backend", None), "astrasim", None)
    collectives = getattr(backend, "collectives", None)
    if collectives is not None:
        default = str(getattr(collectives, op.replace("-", "_"), "auto") or "auto")

    from astrasim_lib.config_generation import _normalize_topology_name, choose_collective

    topology = _normalize_topology_name(getattr(dimension, "topology_type", "") or "")
    return str(choose_collective(default, topology, op)).lower()

def _network_transfer_count(
    dimension: Any, collective_type: str, algorithm: str = "", participants_override: int | None = None
) -> int:
    topology = _normalized_topology(getattr(dimension, "topology_type", ""))
    participants = max(1, int(participants_override if participants_override is not None else (getattr(dimension, "size", 1) or 1)))
    collective = str(collective_type or "").upper()

    if algorithm == "ring":
        topology = "ring"
    elif algorithm == "direct":
        topology = "fullyconnected"
    if topology == "ring":
        if collective == "ALL_REDUCE":
            return max(1, 2 * (participants - 1))
        if collective in {"ALL_GATHER", "REDUCE_SCATTER"}:
            return max(1, participants - 1)
    if topology == "fullyconnected" and collective == "ALL_REDUCE":
        return 2
    return 1

def _zero_moe_normalize_traffic(stats: Mapping[str, Any]) -> dict[str, Any]:
    sidecar = stats.get("zero_moe_operation_groups")

    if not isinstance(sidecar, list):
        raise ValueError("ZeRO+MoE communication energy requires operation-group sidecar")
    raw_operations = {}
    for entry in stats.get("operations", ()) or ():
        if not isinstance(entry, Mapping):
            continue
        oid = str(entry.get("operation_id"))
        if oid in raw_operations and raw_operations[oid] != entry:
            raise ValueError(f"conflicting operation metadata for {oid}")
        raw_operations[oid] = entry
    by_id = {}
    groups = {}
    for entry in sidecar:
        if not isinstance(entry, Mapping):
            raise ValueError("invalid operation-group sidecar entry")
        oid = str(entry["operation_id"])
        rank = int(entry["rank"])
        members = tuple(sorted(int(value) for value in entry["members"]))
        group = str(entry["group_id"])
        ordinal = int(entry["ordinal"])
        kind = str(entry["collective_type"]).upper()
        size = int(entry["collective_size_bytes"])
        if oid in by_id or not members or rank not in members or len(set(members)) != len(members):
            raise ValueError(f"invalid or duplicated sidecar operation {oid}")
        identity = (group, ordinal)
        state = groups.setdefault(identity, {"members": members, "kind": kind, "size": size, "ranks": set()})
        if (state["members"], state["kind"], state["size"]) != (members, kind, size) or rank in state["ranks"]:
            raise ValueError(f"inconsistent collective identity {identity}")
        state["ranks"].add(rank)
        meta = raw_operations.get(oid)
        if meta is not None and (str(meta.get("collective_type", "")).upper() != kind
                                 or int(meta.get("collective_size_bytes", 0)) != size):
            raise ValueError(f"trace metadata disagrees with sidecar for operation {oid}")
        by_id[oid] = f"group:{group}:invocation:{ordinal}"
    for identity, state in groups.items():
        if state["ranks"] != set(state["members"]):
            raise ValueError(f"incomplete collective participants for {identity}")

    def logical_id(oid: Any) -> str:
        key = str(oid)
        if key in by_id:
            return by_id[key]

        meta = raw_operations.get(key)
        if meta is not None and str(meta.get("collective_type", "")).upper() == "P2P":
            return f"p2p:{key}"
        raise ValueError(f"unmapped ZeRO+MoE traffic operation {key}")

    normalized = dict(stats)
    normalized["operations"] = [
        {"operation_id": f"group:{group}:invocation:{ordinal}",
         "collective_type": state["kind"], "collective_size_bytes": state["size"]}
        for (group, ordinal), state in groups.items()
    ] + [dict(entry, operation_id=f"p2p:{oid}") for oid, entry in raw_operations.items()
         if str(entry.get("collective_type", "")).upper() == "P2P"]
    for field in ("operation_links", "reduction_memory_activity"):
        normalized[field] = [dict(entry, operation_id=logical_id(entry.get("operation_id")))
                             for entry in stats.get(field, ()) or () if isinstance(entry, Mapping)]
    return normalized

def communication_energy_from_astrasim(
    link_stats: Mapping[str, Any], hw_config: Any, *, zero_moe_dimension_reconcile: bool = False,
    collective_launch_overhead: float = 0.0,
    launch_durations: Mapping[str, Mapping[int, float]] | None = None,
) -> CommunicationEnergyResult:
    if not isinstance(link_stats, Mapping):
        raise TypeError("link_stats must be a mapping")

    if zero_moe_dimension_reconcile:
        if launch_durations:
            normalized_launches: dict[str, dict[int, float]] = {}
            for entry in link_stats.get("zero_moe_operation_groups", ()) or ():
                oid = str(entry["operation_id"])
                logical = f"group:{entry['group_id']}:invocation:{entry['ordinal']}"
                normalized_launches.setdefault(logical, {}).update(launch_durations.get(oid, {}))
            launch_durations = normalized_launches
        link_stats = _zero_moe_normalize_traffic(link_stats)

    operations: dict[str, Mapping[str, Any]] = {}
    for entry in link_stats.get("operations", ()) or ():
        if isinstance(entry, Mapping):
            operations[str(entry.get("operation_id"))] = entry

    dimension_map: dict[int, int] = {}
    effective_latencies: dict[int, float] = {}
    gpu_counts: dict[int, int] = {}
    for entry in link_stats.get("dimensions", ()) or ():
        if not isinstance(entry, Mapping):
            continue
        local_index = int(entry.get("dimension_index", -1))
        hardware_index = int(entry.get("hardware_dimension_index", local_index))
        dimension_map[local_index] = hardware_index
        if "effective_latency_ns" in entry:
            effective_latencies[hardware_index] = _nonnegative(
                entry["effective_latency_ns"], "effective network latency")
        if "gpu_count" in entry:
            gpu_counts[hardware_index] = int(entry["gpu_count"])

    def _hardware_dimension_index(entry: Mapping[str, Any]) -> int:
        local_index = int(entry.get("dimension_index", -1))
        return dimension_map.get(local_index, local_index)

    usage: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in link_stats.get("operation_links", ()) or ():
        if not isinstance(entry, Mapping):
            continue
        transmitted = int(round(_nonnegative(entry.get("bytes", 0), "link bytes")))
        if transmitted == 0:
            continue
        operation_id = str(entry.get("operation_id"))
        dimension_index = _hardware_dimension_index(entry)
        item = usage.setdefault(
            (operation_id, dimension_index),
            {
                "transmitted_bytes": 0,
                "chunks": 0,
                "edges": set(),
                "serialization_time_ns": 0.0,
            },
        )
        item["transmitted_bytes"] += transmitted
        item["chunks"] += int(entry.get("chunks", 0) or 0)
        item["edges"].add(
            (int(entry.get("src_device_id", -1)), int(entry.get("dst_device_id", -1)))
        )
        item["serialization_time_ns"] = max(
            item["serialization_time_ns"],
            _nonnegative(entry.get("busy_time_ns", 0), "link busy time"),
        )

    reduction: dict[tuple[str, int], dict[str, int]] = {}
    for entry in link_stats.get("reduction_memory_activity", ()) or ():
        if not isinstance(entry, Mapping):
            continue
        key = (str(entry.get("operation_id")), _hardware_dimension_index(entry))
        item = reduction.setdefault(key, {"memory_access_bytes": 0})
        item["memory_access_bytes"] += int(
            round(_nonnegative(entry.get("memory_access_bytes", 0), "reduction HBM bytes"))
        )

    if zero_moe_dimension_reconcile:
        for key in list(reduction):
            if key in usage:
                continue
            candidates = [candidate for candidate in usage if candidate[0] == key[0]]
            devices = {int(e.get("device_id", -1)) for e in link_stats.get("reduction_memory_activity", ()) or () if isinstance(e, Mapping) and str(e.get("operation_id")) == key[0] and _hardware_dimension_index(e) == key[1]}
            if len(candidates) == 1:
                target = candidates[0]
                endpoints = {device for edge in usage[target]["edges"] for device in edge}
                if devices and devices <= endpoints:
                    value = reduction.pop(key)["memory_access_bytes"]
                    reduction.setdefault(target, {"memory_access_bytes": 0})["memory_access_bytes"] += value

    grouped_keys: dict[tuple[int, str, int], list[tuple[str, int]]] = {}
    for key in usage:
        operation = operations.get(key[0], {})
        signature = (
            key[1],
            str(operation.get("collective_type", "UNKNOWN")).upper(),
            int(operation.get("collective_size_bytes", 0) or 0),
        )
        grouped_keys.setdefault(signature, []).append(key)

    logical_usage: list[tuple[tuple[str, ...], int, str, int, dict[str, Any]]] = []
    for (dimension_index, collective_type, collective_size), keys in sorted(
        grouped_keys.items()
    ):
        dimension = _dimension(hw_config, dimension_index)
        participants = max(1, int(getattr(dimension, "size", 1) or 1))

        def _operation_sort_key(key: tuple[str, int]) -> tuple[int, str]:
            try:
                return (int(key[0]), key[0])
            except ValueError:
                return (2**63 - 1, key[0])

        if collective_type == "P2P":
            for key in sorted(keys, key=_operation_sort_key):
                logical_usage.append(
                    ((key[0],), dimension_index, collective_type, collective_size, dict(usage[key]))
                )
            continue

        if zero_moe_dimension_reconcile:
            for key in sorted(keys, key=_operation_sort_key):
                item = dict(usage[key])
                edges = item["edges"]
                nodes = {v for edge in edges for v in edge}
                if not nodes or min(nodes) < 0 or any(u == v for u, v in edges):
                    raise ValueError(f"invalid subgroup link endpoints for {key}")
                if {u for u, _ in edges} != nodes or {v for _, v in edges} != nodes:
                    raise ValueError(f"ambiguous rank-local subgroup operation {key}")
                remaining = set(nodes)
                components = []
                while remaining:
                    component = {remaining.pop()}
                    frontier = list(component)
                    while frontier:
                        u = frontier.pop()
                        adjacent = {v if u == a else a for a, v in edges if a == u or v == u}
                        new = adjacent & remaining
                        remaining.difference_update(new)
                        component.update(new)
                        frontier.extend(new)
                    components.append(component)
                sizes = {len(component) for component in components}
                if len(sizes) != 1 or min(sizes) < 2 or max(sizes) > participants:
                    raise ValueError(f"ambiguous subgroup sizes for {key}: {sorted(sizes)}")
                item["subgroup_participants"] = sizes.pop()
                logical_usage.append(((key[0],), dimension_index, collective_type, collective_size, item))
            continue

        rank_local: list[tuple[str, int]] = []
        for key in sorted(keys, key=_operation_sort_key):
            if len(usage[key]["edges"]) >= participants:
                chunks = ((key,),)
                for chunk in chunks:
                    item = dict(usage[key])
                    logical_usage.append(
                        ((key[0],), dimension_index, collective_type, collective_size, item)
                    )
            else:
                rank_local.append(key)
        if len(rank_local) % participants:
            raise ValueError(
                "rank-local AstraSim operation records do not form a complete "
                f"{participants}-rank collective for dimension {dimension_index}: "
                f"{[key[0] for key in rank_local]}"
            )
        for offset in range(0, len(rank_local), participants):
            rank_group = rank_local[offset : offset + participants]
            merged = {
                "transmitted_bytes": 0,
                "chunks": 0,
                "edges": set(),
                "serialization_time_ns": 0.0,
            }
            for key in rank_group:
                item = usage[key]
                merged["transmitted_bytes"] += item["transmitted_bytes"]
                merged["chunks"] += item["chunks"]
                merged["edges"].update(item["edges"])
                merged["serialization_time_ns"] = max(
                    merged["serialization_time_ns"], item["serialization_time_ns"]
                )
            logical_usage.append(
                (
                    tuple(key[0] for key in rank_group),
                    dimension_index,
                    collective_type,
                    collective_size,
                    merged,
                )
            )

    collectives: list[CollectiveCommunicationEnergy] = []
    matched_reductions: set[tuple[str, int]] = set()
    from sim_settings import nccl_latency_rule
    formula_latency = nccl_latency_rule(hw_config) in ("tuning_formula", "tuning_protocol")
    for operation_ids, dimension_index, collective_type, collective_size, item in logical_usage:
        dimension = _dimension(hw_config, dimension_index)
        algorithm = _collective_algorithm(hw_config, dimension, collective_type)
        network_transfers = _network_transfer_count(
            dimension, collective_type, algorithm, item.get("subgroup_participants")
        )
        serialization_time_ns = float(item["serialization_time_ns"])
        protocol = None
        if (nccl_latency_rule(hw_config) == "tuning_protocol"
                and collective_type in ("ALL_REDUCE", "ALL_GATHER", "REDUCE_SCATTER")):
            from program.nccl_latency import protocol_model
            from timing_model import CollectiveType
            n = item.get("subgroup_participants") or gpu_counts.get(dimension_index) or int(dimension.size)
            protocol = protocol_model(hw_config, dimension.parallelisms[0],
                                      CollectiveType(collective_type.lower()), n, collective_size)
            serialization_time_ns = protocol["transfer_seconds"] * 1e9
        latency_ns = _nonnegative(getattr(dimension, "latency", 0.0), "network latency") * 1e9
        if formula_latency:
            latency_ns = effective_latencies.get(dimension_index, latency_ns)

        completion_time_ns = serialization_time_ns + network_transfers * latency_ns
        system_power_w = _nonnegative(
            getattr(dimension, "system_power_w", 0.0), "system network power"
        )
        per_link_power_w = _nonnegative(
            getattr(dimension, "static_power_per_link_w", 0.0),
            "static power per link",
        )
        active_link_count = len(item["edges"])
        if system_power_w > 0.0:
            participants = {
                device for edge in item["edges"] for device in edge if device >= 0
            }
            dimension_size = max(1, int(getattr(dimension, "size", 1) or 1))
            share = (
                min(1.0, len(participants) / dimension_size) if participants else 1.0
            )
            active_network_j = system_power_w * share * completion_time_ns * 1e-9
        else:
            active_network_j = (
                per_link_power_w * active_link_count * completion_time_ns * 1e-9
            )
        if (
            algorithm == "direct"
            and str(collective_type).upper() == "ALL_REDUCE"
        ):
            active_network_j *= float(network_transfers)
        transmitted_bytes = int(item["transmitted_bytes"])
        participants_n = item.get("subgroup_participants") or len(
            {device for edge in item["edges"] for device in edge if device >= 0}
        ) or 2
        reduction_keys = tuple((operation_id, dimension_index) for operation_id in operation_ids)
        astra_reduction_memory_bytes = sum(
            reduction.get(key, {"memory_access_bytes": 0})["memory_access_bytes"]
            for key in reduction_keys
        )
        matched_reductions.update(key for key in reduction_keys if key in reduction)
        payload_bytes = transmitted_bytes
        if protocol is not None:
            transmitted_bytes *= protocol["wire_multiplier"]
        link_j = transmitted_bytes * 8.0 * _nonnegative(
            getattr(dimension, "energy_per_bit", 0.0), "link energy per bit"
        )
        active_time_ns = completion_time_ns
        if resolve_rule(hw_config, dimension) == "line_rate_while_active":
            ranks = {device for edge in item["edges"] for device in edge if device >= 0}

            gpu_count = gpu_counts.get(dimension_index)
            if gpu_count is None and _normalized_topology(dimension.topology_type) == "switch":
                gpu_count = int(dimension.size)
            if gpu_count is not None:
                ranks = {rank for rank in ranks if rank < gpu_count}
            if not ranks or any(device < 0 for edge in item["edges"] for device in edge):
                raise RuntimeError("line_rate_while_active requires participating GPU endpoints")
            launches = {}
            for oid in operation_ids:
                launches.update((launch_durations or {}).get(oid, {}))
            launch_sum = sum(_nonnegative(launches.get(rank, collective_launch_overhead),
                                         "collective launch duration") for rank in sorted(ranks)) if collective_type != "P2P" else 0.0

            active_time_ns += launch_sum / len(ranks) * 1e9
            link_j = (_nonnegative(getattr(dimension, "energy_per_bit", 0.0), "link energy per bit")
                      * line_rate_bps(dimension)
                      * (len(ranks) * completion_time_ns * 1e-9 + launch_sum))
        collectives.append(
            collective_energy(
                ProtocolCommunicationEnergy if protocol is not None else CollectiveCommunicationEnergy,
                operation_id=",".join(operation_ids),
                dimension_index=dimension_index,
                collective_type=collective_type,
                collective_size_bytes=collective_size,
                transmitted_bytes=transmitted_bytes,
                chunks=int(item["chunks"]),
                active_link_count=active_link_count,
                serialization_time_ns=serialization_time_ns,
                network_transfers=network_transfers,
                completion_time_ns=active_time_ns,
                link_dynamic_j=link_j,
                active_network_j=active_network_j,
                astra_reduction_memory_bytes=astra_reduction_memory_bytes,
                **nccl_memory_energy(hw_config, collective_type, participants_n, payload_bytes),
                **(dict(nccl_protocol=protocol["protocol"], nccl_protocol_source=protocol["source"],
                        wire_byte_multiplier=protocol["wire_multiplier"]) if protocol is not None else {}),
            )
        )

    orphan_reductions = set(reduction).difference(matched_reductions)
    if orphan_reductions:
        raise ValueError(
            "reduction-memory activity has no matching transmitted-link activity: "
            f"{sorted(orphan_reductions)}"
        )
    return CommunicationEnergyResult(tuple(collectives))

__all__ = [
    "nccl_memory_energy",
    "CollectiveCommunicationEnergy",
    "CommunicationEnergyResult",
    "communication_energy_from_astrasim",
]
