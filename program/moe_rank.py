from dataclasses import dataclass
from math import comb, isfinite

@dataclass(frozen=True)
class Collective:
    name: str
    kind: str
    participants: tuple
    payload_bytes: int
    root: int | None = None
    network_energy_owner: int = 0

def collectives(token_counts, hidden, experts, *, tp=1, tp_rank=0):
    counts = tuple(token_counts)
    if not counts or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in counts):
        raise ValueError("token_counts must contain positive integers")
    if not isinstance(hidden, int) or not isinstance(experts, int) or min(hidden, experts) <= 0:
        raise ValueError("hidden and experts must be positive integers")
    if len(counts) == 1:
        return ()
    if not isinstance(tp, int) or tp < 1 or not isinstance(tp_rank, int) or not 0 <= tp_rank < tp:
        raise ValueError("invalid attention TP rank")
    ranks = tuple(r*tp+tp_rank for r in range(len(counts)))
    return tuple(Collective(f"broadcast_{label}_{r}", "broadcast", ranks, 2*n*width, ranks[r], ranks[0])
                 for label, width in (("hidden", hidden), ("logits", experts))
                 for r, n in enumerate(counts)) + (
        Collective("dp_all_reduce", "all_reduce", ranks, 2*sum(counts)*hidden, network_energy_owner=ranks[0]),)

@dataclass(frozen=True)
class CollectivePrice:
    event: Collective
    seconds: float
    transmitted_bytes: float
    network_joules: float

def price_collectives(events, *, bandwidth_bytes_s, launch_s, joules_per_byte, latency_s=0):
    bw, launch, energy, latency = map(float, (bandwidth_bytes_s, launch_s, joules_per_byte, latency_s))
    if not all(isfinite(x) for x in (bw, launch, energy, latency)) or bw <= 0 or min(launch, energy, latency) < 0:
        raise ValueError("collective model requires positive bandwidth and nonnegative launch/energy")
    out = []
    for event in events:
        p = len(event.participants)
        if p < 2:
            raise ValueError("collective requires at least two ranks")
        if event.kind == "broadcast":
            wire, time_bytes, hops = (p-1)*event.payload_bytes, (p-1)*event.payload_bytes, p-1
        elif event.kind == "all_reduce":
            wire = 2*(p-1)*event.payload_bytes
            time_bytes, hops = wire/p, 2*(p-1)
        else:
            raise ValueError(f"unsupported collective {event.kind!r}")
        out.append(CollectivePrice(event, launch+hops*latency+time_bytes/bw, wire, wire*energy))
    return tuple(out)

DEEPEP_BACKENDS = ("deepep_low_latency", "deepep_high_throughput", "deepep_low_latency_received")

def local_expert_count(experts, ep, rank):
    q, rem = divmod(experts, ep)
    return q + (rank < rem)

def deepep_rank_bytes(backend, tokens, hidden, experts, top_k, ep, rank, *, act_bytes=2):
    for name, value in (("tokens", tokens), ("hidden", hidden), ("experts", experts), ("top_k", top_k), ("ep", ep)):
        if not isinstance(value, int) or isinstance(value, bool) or value < (0 if name == "tokens" else 1):
            raise ValueError(f"{name} must be a positive integer")
    if top_k > experts or ep > experts or not 0 <= rank < ep:
        raise ValueError("invalid DeepEP traffic dimensions")
    row = act_bytes*hidden
    if backend in ("deepep_low_latency", "deepep_low_latency_received"):
        pairs = tokens*top_k*(experts - local_expert_count(experts, ep, rank))/experts
        return pairs*(row + 8), pairs*row
    if backend == "deepep_high_throughput":
        total = comb(experts, top_k)
        hit = sum(1 - comb(experts - local_expert_count(experts, ep, r), top_k)/total for r in range(ep) if r != rank)
        rows = tokens*hit
        return rows*(row + top_k*(8 + 4)), rows*row
    raise ValueError(f"not a DeepEP backend: {backend!r}")

def deepep_ht_received_tokens(tokens, experts, top_k, ep, rank):
    hit = 1 - comb(experts - local_expert_count(experts, ep, rank), top_k)/comb(experts, top_k)
    return tokens * hit

def price_all_to_all(name, participants, rank_bytes, *, bandwidth_bytes_s, launch_s, joules_per_byte, latency_s=0):
    bw, launch, energy, latency = map(float, (bandwidth_bytes_s, launch_s, joules_per_byte, latency_s))
    if not all(isfinite(x) for x in (bw, launch, energy, latency)) or bw <= 0 or min(launch, energy, latency) < 0:
        raise ValueError("collective model requires positive bandwidth and nonnegative launch/energy")
    sizes = tuple(float(x) for x in rank_bytes)
    p = len(participants)
    if p < 2 or len(sizes) != p or any(not isfinite(x) or x < 0 for x in sizes):
        raise ValueError("all-to-all requires one nonnegative byte count per rank and at least two ranks")
    busiest = max(sizes)
    event = Collective(name, "all_to_all", tuple(participants), int(round(busiest)), None, min(participants))
    wire = sum(sizes)
    return CollectivePrice(event, launch + latency + busiest/bw, wire, wire*energy)

def stage_latency(rank_seconds, collective_prices=()):
    durations = tuple(float(x) for x in rank_seconds)
    if not durations or any(not isfinite(x) or x < 0 for x in durations):
        raise ValueError("rank durations must be finite and nonnegative")
    return max(durations) + sum(x.seconds for x in collective_prices)
