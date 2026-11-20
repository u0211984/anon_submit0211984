def simple_flash_input_broadcasts(*, batch, padded_batch, query_tokens,
                                  block_table_columns, selected_tokens):
    values = (batch, padded_batch, query_tokens, block_table_columns, selected_tokens)
    if any(not isinstance(x, int) or isinstance(x, bool) for x in values):
        raise ValueError("broadcast plan dimensions must be integers")
    if batch <= 0 or padded_batch < batch or query_tokens < padded_batch or block_table_columns < 0 or selected_tokens < 0:
        raise ValueError("invalid broadcast plan dimensions")
    sizes = {
        "input_tokens": 8 * query_tokens,
        "input_positions": 8 * query_tokens,
        "slot_mapping": 8 * query_tokens,
        "seq_lens_tensor": 4 * padded_batch,
        "query_start_loc": 4 * (batch + 1),
        "seq_start_loc": 4 * (padded_batch + 1),
        "context_lens_tensor": 4 * batch,
        "block_tables": 4 * padded_batch * block_table_columns,
        "selected_token_indices": 8 * selected_tokens,
    }
    return {name: size for name, size in sizes.items() if size}

def ring_broadcast_seconds(sizes, participants, *, base, latency, bandwidth):
    import math
    if participants < 2 or int(participants) != participants:
        raise ValueError("broadcast participants must be an integer >= 2")
    if any(not math.isfinite(x) or x < 0 for x in (base, latency, bandwidth)):
        raise ValueError("broadcast timing inputs must be finite and nonnegative")
    if not bandwidth:
        return float("inf")
    return sum(base + (participants - 1) * latency + size / bandwidth for size in sizes.values())

def input_broadcast_energy(record, hw, policy, multiplicity=1):
    from program.network_energy import (CollectiveCommunicationEnergy,
        CommunicationEnergyResult, _dimension, _nonnegative)
    from program.link_energy import resolve_rule, line_rate_bps
    if not record:
        return CommunicationEnergyResult()
    n = int(record["participants"])
    if n < 2:
        raise ValueError("input broadcasts require at least two ranks")
    factor = _nonnegative(multiplicity, "broadcast multiplicity")
    index = int(policy["dimension_index"])
    dim = _dimension(hw, index)
    bit_j = _nonnegative(getattr(dim, "energy_per_bit", 0), "link energy per bit")
    tech = hw.tech_config
    hbm_j = _nonnegative(tech.DRAM.dynamic_energy_per_byte_value, "HBM energy per byte")
    l2_j = _nonnegative(tech.SRAML2.dynamic_energy_per_byte_value, "L2 energy per byte")
    rule = resolve_rule(hw, dim)
    rows = []
    base = float(hw.sw_config.collective_launch_overhead)
    for name, size in record["sizes"].items():
        tx, memory = (n - 1) * size * factor, n * size * factor
        seconds = ring_broadcast_seconds({name: size}, n, base=base,
                    latency=policy["latency"], bandwidth=policy["bandwidth"])
        link_j = (tx * 8 * bit_j if rule == "per_payload_byte" else
                  bit_j * line_rate_bps(dim) * seconds * (n - 1) * factor)
        rows.append(CollectiveCommunicationEnergy(
            operation_id="input_broadcast." + name, dimension_index=index,
            collective_type="BROADCAST", collective_size_bytes=size,
            transmitted_bytes=tx, chunks=factor, active_link_count=n - 1,
            serialization_time_ns=0, network_transfers=(n - 1) * factor,
            completion_time_ns=seconds * factor * 1e9,
            link_dynamic_j=link_j, active_network_j=0,
            nccl_hbm_bytes=memory, nccl_hbm_j=memory * hbm_j,
            nccl_l2_bytes=memory, nccl_l2_j=memory * l2_j,
        ))
    return CommunicationEnergyResult(tuple(rows))
