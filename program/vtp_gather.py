from dataclasses import dataclass

@dataclass(frozen=True)
class GatherPlan:
    root: int
    shard_bytes: int
    transfers: tuple[tuple[int, int, int], ...]
    local_copy_bytes: int

    @property
    def network_bytes(self):
        return sum(size for _, _, size in self.transfers)

def logits_gather_plan(participants: int, shard_bytes: int, root: int = 0) -> GatherPlan:
    n, size, root = int(participants), int(shard_bytes), int(root)
    if n < 1 or not 0 <= root < n or size < 0:
        raise ValueError('invalid gather participants, root or shard bytes')
    return GatherPlan(root, size, tuple((rank, root, size) for rank in range(n) if rank != root), size)

def gather_seconds(participants: int, shard_bytes: int, bandwidth: float, latency: float,
                   per_peer_links: bool) -> float:
    plan = logits_gather_plan(participants, shard_bytes)
    if not plan.transfers or not plan.shard_bytes:
        return 0.0
    if bandwidth <= 0 or latency < 0:
        raise ValueError("gather needs positive per-peer bandwidth and nonnegative latency")
    received = plan.shard_bytes if per_peer_links else plan.network_bytes
    return latency + received / bandwidth

def logical_gathers(program):
    from program.ir import ComputeOp
    from timing_model import CollectiveType
    if not program.meta.misc.get('vtp_per_rank'):
        return ()
    leader = min(program.compute_devices())
    return tuple(dict(label=call.name, participants=call.participants,
                      size_bytes=call.size_bytes, multiplicity=max(1, program.dp_count),
                      active_seconds=call.duration_seconds)
                 for op in program.ops if isinstance(op, ComputeOp) and op.device == leader
                 for call in op.embedded_collectives if call.coll is CollectiveType.GATHER)

def gather_energy(records, hw, dimension_index=0):
    from program.network_energy import (CollectiveCommunicationEnergy, CommunicationEnergyResult,
                                        _dimension, _nonnegative)
    from program.link_energy import resolve_rule, line_rate_bps
    dim = _dimension(hw, dimension_index)
    bit_j = _nonnegative(getattr(dim, 'energy_per_bit', 0), 'link energy per bit')
    tech = hw.tech_config
    hbm_j = _nonnegative(tech.DRAM.dynamic_energy_per_byte_value, 'DRAM energy per byte')
    l2_j = _nonnegative(tech.SRAML2.dynamic_energy_per_byte_value, 'L2 energy per byte')
    rows = []
    for record in records:
        plan = logits_gather_plan(record['participants'], record['size_bytes'])
        factor = _nonnegative(record.get('multiplicity', 1), 'gather multiplicity')
        rule = (resolve_rule(hw, dim) if plan.transfers and plan.shard_bytes and factor
                else 'per_payload_byte')
        tx = plan.network_bytes * factor
        user_bytes = 2 * (plan.network_bytes + plan.local_copy_bytes) * factor
        if rule == 'line_rate_while_active' and plan.transfers:
            seconds = _nonnegative(record['active_seconds'], 'gather active time')
            completion_ns = seconds * factor * 1e9
            link_j = bit_j * line_rate_bps(dim) * seconds * int(record['participants']) * factor
        else:
            completion_ns = 0
            link_j = tx * 8 * bit_j
        rows.append(CollectiveCommunicationEnergy(
            operation_id=record['label'], dimension_index=dimension_index, collective_type='GATHER',
            collective_size_bytes=plan.shard_bytes, transmitted_bytes=tx, chunks=0,
            active_link_count=len(plan.transfers), serialization_time_ns=0,
            network_transfers=len(plan.transfers) * factor, completion_time_ns=completion_ns,
            link_dynamic_j=link_j, active_network_j=0,
            nccl_hbm_bytes=user_bytes, nccl_hbm_j=user_bytes * hbm_j,
            nccl_l2_bytes=user_bytes, nccl_l2_j=user_bytes * l2_j,
        ))
    return CommunicationEnergyResult(tuple(rows))

def gather_coverage(coverage, records, dimension):
    from dataclasses import replace
    active = any(float(getattr(dimension, key, 0.) or 0.) > 0 for key in
                 ('system_power_w', 'static_power_per_link_w'))
    if not records or not active:
        return coverage
    missing = tuple(sorted(set(coverage.missing) | {'communication:gather_active_network_power'}))
    status = 'unavailable' if coverage.communication == 'unavailable' else 'partial'
    return replace(coverage, communication=status, missing=missing)
