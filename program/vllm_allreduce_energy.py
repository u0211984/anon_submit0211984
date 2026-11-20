from program.network_energy import (
    CollectiveCommunicationEnergy, CommunicationEnergyResult, _dimension,
    _nonnegative, collective_energy,
)

def custom_allreduce_energy(records, hw, policy):
    from program.link_energy import resolve_rule, line_rate_bps
    from program.vllm_allreduce import AllReduceChoice, custom_seconds, custom_flag_traffic
    dimension_index = int(policy.get("dimension_index", 0))
    dimension = _dimension(hw, dimension_index)
    bit_j = _nonnegative(getattr(dimension, "energy_per_bit", 0), "link energy per bit")
    tech = hw.tech_config
    hbm_j = _nonnegative(tech.DRAM.dynamic_energy_per_byte_value, "DRAM energy per byte")
    l2_j = _nonnegative(tech.SRAML2.dynamic_energy_per_byte_value, "L2 energy per byte")
    flop_j = _nonnegative(tech.core.vector_energy, "vector energy per FLOP")
    element_bytes = float(policy["precision_bytes"])
    if element_bytes <= 0:
        raise ValueError("custom all-reduce precision_bytes must be positive")
    priced = []
    for record in records:
        algorithm = record["algorithm"]
        if algorithm not in ("custom_one_shot", "custom_two_shot"):
            continue
        n = int(record["participants"])
        size = float(record["size_bytes"])
        factor = _nonnegative(record["count"] * record.get("multiplicity", 1), "collective count")
        if n < 2 or size < 0:
            raise ValueError("custom all-reduce needs at least two ranks and nonnegative bytes")
        rule = resolve_rule(hw, dimension) if size and factor else 'per_payload_byte'
        if algorithm == "custom_one_shot":
            memory = (n + 1) * size
            remote = (n - 1) * size
            additions = (n - 1) * size / element_bytes
            hops = 1
        else:
            memory = (3 + 1 / n) * size
            remote = 2 * (n - 1) * size / n
            additions = (n - 1) * size / (n * element_bytes)
            hops = 2
        if not record["cuda_graph"]:
            memory += 2 * size
        if policy.get("sync_energy", False):
            flags = custom_flag_traffic(size, n)
            memory += flags["stores_bytes"] + flags["loads_bytes"]
            remote += flags["remote_store_bytes"]

        memory *= n * factor
        remote *= n * factor
        additions *= n * factor
        if rule == 'line_rate_while_active':
            choice = AllReduceChoice(algorithm, hops, not record['cuda_graph'])
            seconds = _nonnegative(custom_seconds(choice, size, n, policy), 'collective active time')
            completion_ns = seconds * factor * 1e9
            link_j = bit_j * line_rate_bps(dimension) * seconds * n * factor
        else:
            completion_ns = 0
            link_j = remote * 8 * bit_j
        priced.append(CollectiveCommunicationEnergy(
            operation_id=record["label"], dimension_index=dimension_index,
            collective_type=algorithm, collective_size_bytes=int(size),
            transmitted_bytes=remote, chunks=hops * factor,
            active_link_count=n * (n - 1), serialization_time_ns=0,
            network_transfers=hops * n * factor, completion_time_ns=completion_ns,
            link_dynamic_j=link_j, active_network_j=0,
            custom_hbm_bytes=memory, custom_hbm_j=memory * hbm_j,
            custom_l2_bytes=memory, custom_l2_j=memory * l2_j,
            custom_reduction_flops=additions, custom_reduction_j=additions * flop_j,
        ))
    return CommunicationEnergyResult(tuple(priced))

def analytical_nccl_allreduce_energy(records, hw, policy):
    from program.network_energy import nccl_memory_energy
    from program.link_energy import resolve_rule, line_rate_bps
    dimension_index = int(policy.get('dimension_index', 0))
    dimension = _dimension(hw, dimension_index)
    bit_j = _nonnegative(getattr(dimension, 'energy_per_bit', 0), 'link energy per bit')
    rows = []
    for record in records:
        if record['algorithm'] != 'nccl':
            continue
        n = int(record['participants'])
        size = float(record['size_bytes'])
        factor = _nonnegative(record['count'] * record.get('multiplicity', 1), 'collective count')
        if n < 2 or size < 0:
            raise ValueError('NCCL ring all-reduce needs at least two ranks and nonnegative bytes')
        rule = resolve_rule(hw, dimension) if size and factor else 'per_payload_byte'
        tx = 2 * (n - 1) * size * factor
        if rule == 'line_rate_while_active':
            seconds = _nonnegative(record['active_seconds'], 'collective active time')
            completion_ns = seconds * factor * 1e9
            link_j = bit_j * line_rate_bps(dimension) * seconds * n * factor
        else:
            completion_ns = 0
            link_j = tx * 8 * bit_j
        rows.append(collective_energy(
            CollectiveCommunicationEnergy, operation_id=record['label'], dimension_index=dimension_index,
            collective_type='ALL_REDUCE', collective_size_bytes=int(size),
            transmitted_bytes=tx, chunks=0, active_link_count=n,
            serialization_time_ns=0, network_transfers=2 * n * (n - 1) * factor,
            completion_time_ns=completion_ns, link_dynamic_j=link_j, active_network_j=0,
            **nccl_memory_energy(hw, 'ALL_REDUCE', n, tx),
        ))
    return CommunicationEnergyResult(tuple(rows))
