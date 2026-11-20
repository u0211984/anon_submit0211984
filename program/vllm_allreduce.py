from dataclasses import dataclass
import math

from timing_model import CollectiveType

MAX_BYTES = 8192 * 1024

@dataclass(frozen=True)
class AllReduceChoice:
    algorithm: str
    hops: int = 0
    copy_input: bool = False

def choose_allreduce(tp_allreduce, world_size, fully_connected, tensor_bytes, cuda_graph):
    if tp_allreduce == "nccl":
        return AllReduceChoice("nccl")
    if tp_allreduce != "vllm_size_rule":
        raise ValueError(f"Unknown TP all-reduce option {tp_allreduce!r}")
    if world_size < 1 or int(world_size) != world_size:
        raise ValueError("vLLM all-reduce requires a positive integral world size")
    if not math.isfinite(tensor_bytes) or tensor_bytes < 0 or int(tensor_bytes) != tensor_bytes:
        raise ValueError("vLLM all-reduce requires integral nonnegative tensor bytes")
    if world_size == 1 or tensor_bytes == 0:
        return AllReduceChoice("none")
    if world_size not in (2, 4, 6, 8):
        return AllReduceChoice("nccl")
    if tensor_bytes % 16 or tensor_bytes >= MAX_BYTES or (world_size > 2 and not fully_connected):
        return AllReduceChoice("nccl")
    one_shot = (world_size == 2 or (world_size == 4 and tensor_bytes < 512 * 1024)
                or (world_size in (6, 8) and tensor_bytes < 256 * 1024))
    return AllReduceChoice("custom_one_shot" if one_shot else "custom_two_shot",
                           1 if one_shot else 2, not cuda_graph)

def choice_for(policy, kind, participants, size_bytes, axis):
    if not policy or kind is not CollectiveType.ALL_REDUCE or axis != "tp":
        return AllReduceChoice("nccl")
    return choose_allreduce(policy["tp_allreduce"], participants, policy["fully_connected"],
                            size_bytes, policy["cuda_graph"])

def custom_flag_traffic(size_bytes, participants):
    if (not math.isfinite(size_bytes) or size_bytes <= 0 or int(size_bytes) != size_bytes
            or size_bytes % 16 or participants not in (2, 4, 6, 8)):
        raise ValueError("custom flag traffic requires eligible size and rank count")
    blocks = min(36, (int(size_bytes) + 8191) // 8192)
    rounds = 2
    stores = rounds * blocks * participants * 4
    return dict(blocks=blocks, barrier_rounds=rounds, stores_bytes=stores,
                loads_bytes=stores, remote_store_bytes=rounds * blocks * (participants - 1) * 4)

def custom_seconds(choice, size_bytes, participants, policy):
    if not choice.algorithm.startswith("custom_"):
        raise ValueError("custom_seconds requires a custom all-reduce")
    traffic = (participants - 1) * size_bytes
    if choice.hops == 2:
        traffic = 2 * traffic / participants

    bandwidth = policy["bandwidth"] * (participants - 1 if policy["per_peer_links"] else 1)
    transfer = traffic / bandwidth if bandwidth else float("inf")
    copy = 2 * size_bytes / policy["hbm_bandwidth"] if choice.copy_input else 0.0
    timing = policy.get("barrier_timing", "legacy_data_hops")
    if timing == "legacy_data_hops":
        barrier = 0.0
    elif timing == "explicit_flag_barriers":
        flags = custom_flag_traffic(size_bytes, participants)

        flag_transfer = flags["remote_store_bytes"] / bandwidth if bandwidth else float("inf")
        barrier = flags["barrier_rounds"] * policy["latency"] + flag_transfer
    else:
        raise ValueError(f"Unknown vLLM custom barrier timing {timing!r}")
    return policy["kernel_overhead"] + choice.hops * policy["latency"] + transfer + copy + barrier

def op_choice(program, op):
    if not hasattr(op, "coll"):
        return AllReduceChoice("nccl")
    return choice_for(program.meta.misc.get("vllm_allreduce"), op.coll,
                      op.participants, op.size_bytes, op.interconnect)

def logical_records(program, active_durations=None):
    from program.ir import CollectiveOp, ComputeOp
    policy = program.meta.misc.get("vllm_allreduce")
    if not policy:
        return ()
    records = []
    for op in program.ops:
        calls = (op,) if isinstance(op, CollectiveOp) else (
            op.embedded_collectives if isinstance(op, ComputeOp) else ())
        for call in calls:
            if call.coll is not CollectiveType.ALL_REDUCE or call.interconnect != "tp" or call.participants <= 1:
                continue
            if getattr(call, "launch_continuation", False):
                raise ValueError("fragmented vLLM all-reduce is outside the blocking model")
            choice = op_choice(program, call)

            if isinstance(call, CollectiveOp):
                if call.group is None:
                    raise ValueError("TP all-reduce requires explicit communicator membership")
                if call.device != min(call.group.members):
                    continue
            elif program.meta.misc.get("vtp_per_rank"):
                if op.device != min(program.compute_devices()):
                    continue
            record = dict(algorithm=choice.algorithm, count=1,
                                multiplicity=max(1, program.dp_count),
                                participants=call.participants, size_bytes=call.size_bytes,
                                cuda_graph=policy["cuda_graph"], label=call.name)
            if active_durations is not None:
                record["active_seconds"] = (active_durations[call.uid] if isinstance(call, CollectiveOp)
                                             else call.duration_seconds)
            records.append(record)
    return tuple(records)
