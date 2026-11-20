from math import lcm

from kernel_trace import Kernel
from kernel_trace_megatron_te import fused_adam, split_32bit
from l2_sim import RAccess as A

PARAM_ALIGN = 64
BUCKET_ALIGN = 128

MLA_LAYER_ORDER = ('norm1', 'output', 'mla_q_proj', 'mla_q_down', 'mla_q_norm', 'mla_q_up', 'mla_kv_down',
                   'mla_kv_norm', 'mla_kv_up', 'norm2', 'ffn1', 'ffn2')

MOE_EXPERTS_MODES = ('te_grouped', 'sequential')

def _mla_layer_rank(suffix, experts):
    if suffix in MLA_LAYER_ORDER:
        return (MLA_LAYER_ORDER.index(suffix),)
    after_norm2 = MLA_LAYER_ORDER.index('norm2')
    if suffix == 'router':
        return (after_norm2, 0)
    if suffix in ('shared.fc1', 'shared.fc2'):
        return (after_norm2, 2, suffix)
    head, _, which = suffix.partition('.')
    if head.startswith('expert') and head[6:].isdigit() and which in ('fc1', 'fc2'):
        if experts not in MOE_EXPERTS_MODES:
            raise ValueError(f'MLA + MoE registration order needs megatron_moe_experts in {MOE_EXPERTS_MODES}, '
                             f'got {experts!r}')
        g = int(head[6:])
        return (after_norm2, 1) + ((which, g) if experts == 'te_grouped' else (g, which))
    return None

def registration_order(weights, experts=None):
    groups, order = {}, []
    for name in weights:
        key = name.split('.', 2)[1] if name.startswith('param.L') and name.count('.') >= 2 else name
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(name)
    out = {}
    for key in order:
        names = groups[key]
        if any(n.rsplit('.', 1)[-1].startswith('mla_') for n in names):
            rank = {n: _mla_layer_rank(n.split('.', 2)[2], experts) for n in names}
            if None in rank.values():
                raise ValueError(f'MLA layer parameters outside the Megatron registration order: '
                                 f'{sorted(n for n, r in rank.items() if r is None)}')
            names = sorted(names, key=rank.__getitem__)
        for n in names:
            out[n] = weights[n]
    return out

def _pad(n, m):
    return ((int(n) + m - 1) // m) * m

def buffer_layout(weights, dp, distributed):
    position, layout = 0, []
    for name, n in reversed(list(weights)):
        if distributed:
            position = _pad(position, PARAM_ALIGN)
        layout.append((name, position, int(n)))
        position += int(n)
    total = _pad(position, lcm(int(dp), BUCKET_ALIGN)) if distributed else position
    return layout, total

def buffer_spans(weights, dp, distributed):
    layout, total = buffer_layout(weights, dp, distributed)
    ends = [start for _, start, _ in layout[1:]] + [total]
    return {name: end - start for (name, start, _), end in zip(layout, ends)}

def shard_pieces(weights, dp, rank=0):
    layout, total = buffer_layout(weights, dp, True)
    size = total // int(dp)
    lo, hi = rank * size, (rank + 1) * size
    out = []
    for name, start, n in layout:
        a, b = max(lo, start), min(hi, start + n)
        if b > a:
            out.append((name, a - start, b - a))
    return out

class DataParallel:
    def __init__(self, *, dp, distributed, end_of_backward, pre_scale, precision):
        self.dp, self.distributed = int(dp), bool(distributed)
        self.end_of_backward, self.pre_scale = bool(end_of_backward), bool(pre_scale)
        self.p = precision
        self.reduced = False
        self.gathered = False

    def _grad(self, name, n, off=0):
        return A(name + '.main_grad', float(self.p.gradients * n), False, float(self.p.gradients * off))

    def _param(self, name, n, off=0):
        return A(name, float(self.p.parameters * n), True, float(self.p.parameters * off))

    def _ranges(self, weights, names, size):
        remaining, out = float(size), []
        for name in names:
            n = min(weights[name], remaining / float(self.p.gradients))
            if n > 0:
                out.append((name, n))
                remaining -= n * float(self.p.gradients)
        if remaining > 1e-6 * max(1.0, float(size)):
            raise ValueError(f'Megatron data-parallel collective exceeds its owned main-grad buffers by {remaining} bytes')
        return out

    def grad_sync(self, weights, names, size, kind):
        if self.end_of_backward:
            if self.reduced:
                return [], (), ()
            self.reduced = True
            ranges = list(weights.items())
        else:
            ranges = self._ranges(weights, names, size)
        before = []
        if self.pre_scale:
            for launch in split_32bit(ranges, self.p.gradients):
                acc = tuple(self._grad(n, c, o) for n, o, c in launch)
                before.append(Kernel('main_grad', 'backward.grad_scale', 'pointwise', acc, acc,
                                     float(sum(c for _, _, c in launch))))
        reads = tuple(self._grad(n, c) for n, c in ranges)
        if kind == 'reduce_scatter':
            own = dict(weights) if self.end_of_backward else {n: c for n, c in ranges}
            writes = tuple(self._grad(n, c, o) for n, o, c in shard_pieces(list(weights.items()), self.dp)
                           if n in own)
        else:
            writes = reads
        return before, reads, writes

    def param_gather(self, weights, names, size):
        if self.end_of_backward:
            if self.gathered:
                return (), ()
            self.gathered = True
            own = dict(weights)
        else:
            own = {n: weights[n] for n in names}
        reads = tuple(self._param(n, c, o) for n, o, c in shard_pieces(list(weights.items()), self.dp) if n in own)
        writes = tuple(self._param(n, c) for n, c in own.items())
        return reads, writes

    def optimizer(self, weights):
        if not self.distributed:
            return None
        pieces = shard_pieces(list(weights.items()), self.dp)
        keys = [f'{name}@{off}' for name, off, _ in pieces]
        sized = dict(zip(keys, (n for _, _, n in pieces)))
        ks = []
        for k in fused_adam(sized, self.p):
            ks.append(_relocate(k, {key: (name, off) for key, (name, off, _) in zip(keys, pieces)}, self.p))
        for name, off, n in pieces:
            main = A(name + '.master', float(self.p.master_parameters * n), False, float(self.p.master_parameters * off))
            ks.append(Kernel('optimizer', 'optimizer.param_copy', 'pointwise', (main,), (self._param(name, n, off),)))
        return ks

def _relocate(kernel, where, p):
    def move(access):
        tensor = access.tensor
        for suffix, width in (('.main_grad', p.gradients), ('.master', p.master_parameters),
                              ('.m1', p.optimizer_states), ('.m2', p.optimizer_states)):
            if tensor.endswith(suffix):
                name, off = where[tensor[:-len(suffix)]]
                return A(name + suffix, access.nbytes, access.weight, float(access.offset) + float(width * off))
        raise ValueError(f'unexpected optimizer tensor {tensor}')
    return Kernel(kernel.op, kernel.name, kernel.kind, tuple(map(move, kernel.reads)), tuple(map(move, kernel.writes)),
                  kernel.flops, kernel.gemm, kernel.activity_name)
