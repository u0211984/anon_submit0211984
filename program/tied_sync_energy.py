from program.link_energy import line_rate_bps, resolve_rule
from program.network_energy import (
    CollectiveCommunicationEnergy, CommunicationEnergyResult, collective_energy, nccl_memory_energy,
)

def tied_sync_energy(workload, hw, *, tp, cp, dp, element_bytes=2):
    calls = [call for call in workload.embedded_collectives.get('optimizer_embedding', ())
             if call.name == 'embedding_group_allreduce' and call.size_bytes > 0]
    if not calls:
        return CommunicationEnergyResult(), None
    if len(calls) != 1 or calls[0].participants != 2:
        raise ValueError('expected one two-participant tied embedding sync')
    call = calls[0]
    matches = [(i, dim) for i, dim in enumerate(hw.network_layout.dimensions)
               if 'pp' in getattr(dim, 'parallelisms', ())]
    if len(matches) != 1:
        raise ValueError('tied embedding sync requires one pp network dimension')
    index, dim = matches[0]
    shards = max(1, int(tp)) * max(1, int(cp))
    replicas = max(1, int(dp))
    shard_bytes = float(call.size_bytes) / shards
    tx = 2 * float(call.size_bytes) * replicas
    bit_j = float(getattr(dim, 'energy_per_bit', 0.0) or 0.0)
    rule = resolve_rule(hw, dim)
    seconds = float(call.duration_seconds)
    if rule == 'line_rate_while_active':
        link_j = bit_j * line_rate_bps(dim) * seconds * 2 * shards * replicas
        completion_ns = seconds * shards * replicas * 1e9
    else:
        link_j = tx * 8 * bit_j
        completion_ns = 0.0
    memory = nccl_memory_energy(hw, 'ALL_REDUCE', 2, tx)

    if float(element_bytes) <= 0:
        raise ValueError('gradient communication precision must be positive')
    memory['reduction_flops'] = tx / (2 * float(element_bytes))
    memory['reduction_j'] = memory['reduction_flops'] * float(hw.tech_config.core.vector_energy)
    row = collective_energy(
        CollectiveCommunicationEnergy, operation_id='embedding_group_allreduce', dimension_index=index,
        collective_type='ALL_REDUCE', collective_size_bytes=int(shard_bytes),
        transmitted_bytes=tx, chunks=0, active_link_count=2,
        serialization_time_ns=0, network_transfers=4 * shards * replicas,
        completion_time_ns=completion_ns, link_dynamic_j=link_j,
        active_network_j=0, **memory,
    )
    return CommunicationEnergyResult((row,)), dim
