from decimal import Decimal, ROUND_HALF_EVEN

from program.ir import CollectiveOp
from timing_model import CollectiveType

def is_nccl_call(kind, *, participants, size_bytes, continuation=False, algorithm="nccl"):
    return (kind in (CollectiveType.ALL_REDUCE, CollectiveType.ALL_GATHER, CollectiveType.GATHER,
                     CollectiveType.REDUCE_SCATTER, CollectiveType.ALL_TO_ALL)
            and algorithm == "nccl" and not continuation and size_bytes > 0 and participants > 1)

def is_nccl_collective(op, *, algorithm="nccl"):
    return (isinstance(op, CollectiveOp)
            and is_nccl_call(op.coll, participants=op.participants,
                             size_bytes=op.size_bytes,
                             continuation=op.launch_continuation, algorithm=algorithm))

class LaunchAccounting:
    def __init__(self, seconds):
        self.seconds = Decimal(str(seconds))
        self.counts = {}
        self.emitted_us = {}
        self.calls = {}
        self.totals = {}

    def add(self, rank, *, label, emit=False, seconds=None):
        seconds = self.seconds if seconds is None else Decimal(str(seconds))
        count = self.counts.get(rank, 0)
        self.counts[rank] = count + 1
        calls = self.calls.setdefault(rank, {})
        row = calls.setdefault(str(label), {'count': 0, 'emitted_us': 0, 'seconds': Decimal(0)})
        row['count'] += 1
        row['seconds'] += seconds
        self.totals[rank] = self.totals.get(rank, Decimal(0)) + seconds
        if not emit:
            return 0
        total = self.totals[rank] * Decimal(1000000)
        rounded = int(total.to_integral_value(rounding=ROUND_HALF_EVEN))
        delta = rounded - self.emitted_us.get(rank, 0)
        self.emitted_us[rank] = rounded
        row["emitted_us"] += delta
        return delta

    def records(self, *, emitted):
        return {str(rank): {'count': count,
                            'exact_seconds': float(self.totals[rank]),
                            'emitted_seconds': self.emitted_us.get(rank, 0) * 1e-6 if emitted else None,
                            'calls': {name: {'count': row['count'],
                                             'exact_seconds': float(row['seconds']),
                                             'emitted_seconds': row['emitted_us'] * 1e-6 if emitted else None}
                                      for name, row in self.calls[rank].items()}}
                for rank, count in self.counts.items()}

def launch_issue_consumers(program):
    from program.ir import ComputeOp, TransferOp

    ops = program.ops
    resolved = {}

    def effective(dep):
        dep = int(dep)
        if dep not in resolved:
            op = ops[dep]
            if isinstance(op, TransferOp) and op.src_device == op.dst_device:
                resolved[dep] = tuple(d for parent in op.deps for d in effective(parent))
            else:
                resolved[dep] = (dep,)
        return resolved[dep]

    users, inputs = {}, {}
    for op in ops:
        if isinstance(op, ComputeOp):
            inputs[op.uid] = set(d for raw in op.deps for d in effective(raw))
            for dep in inputs[op.uid]:
                users.setdefault(dep, []).append(op)
    out = {}
    for op in ops:
        if not isinstance(op, CollectiveOp):
            continue
        consumers = tuple(op.launch_overlap_consumers)
        if op.launch_overlap_consumer is not None:
            consumers += (op.launch_overlap_consumer,)
        follower = min((u.uid for raw in op.deps for dep in effective(raw) for u in users.get(dep, ())
                        if u.device == op.device and u.uid > op.uid and op.uid not in inputs[u.uid]), default=None)
        if follower is not None:
            consumers += (follower,)
        if consumers:
            out[op.uid] = tuple(dict.fromkeys(consumers))
    return out
