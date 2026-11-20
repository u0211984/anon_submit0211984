from dataclasses import replace

from kernel_trace import Kernel, LayerDims
import kernel_trace_train as eager

def expand(op, direction, reads, writes, *, fused, sm_count, stats_width=4, prefix='',
           elements=None, rows=None, activation_width=2, heads=1, batch=1,
           model_type='llama', qk_norm=False, softmax_reads=3):
    if op not in ('norm', 'rope', 'activation', 'residual_add', 'loss'):
        raise ValueError(f'unknown pointwise operation {op!r}')
    if direction not in ('forward', 'backward'):
        raise ValueError(f'unknown pointwise direction {direction!r}')
    if elements is None or rows is None or int(rows) <= 0 or int(elements) <= 0:
        raise ValueError('pointwise expansion requires positive elements and rows')
    e, r, heads = int(elements), int(rows), int(heads)
    if e % r or heads < 1:
        raise ValueError('pointwise elements must divide evenly into rows; heads must be positive')
    reads, writes = tuple(reads), tuple(writes)
    counts = {'norm': ((2, 2), (5, 2)), 'rope': ((3, 1), (3, 1)),
              'activation': ((2, 1), (3, 2)), 'residual_add': ((2, 1), (2, 1)),
              'loss': ((2, 2), (4, 1))}
    nr, nw = counts[op][direction == 'backward']
    if len(reads) != nr or len(writes) != nw:
        raise ValueError(f'{op} {direction} expects {nr} reads and {nw} writes')
    d = LayerDims(tokens=r, batch=int(batch), hidden=e // r, heads=heads,
                  kv_heads=heads, head_dim=e // r, intermediate=e // r,
                  vocab=e // r, act_bytes=activation_width,
                  weight_bytes=activation_width, kv_bytes=activation_width,
                  model_type=model_type)
    opts = eager.TrainOpts(sm_count=sm_count, loss_bytes=stats_width, softmax_reads=softmax_reads)
    forward = direction == 'forward'
    mapping = {}
    if op == 'norm':
        if qk_norm:
            if r % heads:
                raise ValueError('Qwen3 norm rows must divide into heads')
            d = replace(d, tokens=r // heads)
            originals = (eager.qwen3_norm_fwd('X', 'Y', 'W', heads, d, direction)
                         if forward else
                         eager.qwen3_norm_bwd('GY', 'X', 'Y', 'W', 'GX', heads, d, opts))
        else:
            originals = (eager.rmsnorm_fwd(op, 'X', 'Y', d, 'W') if forward else
                         eager.rmsnorm_bwd(op, 'GY', 'X', 'Y', 'W', 'GX', d, False, opts))
        mapping = ({'X': reads[0], 'W': reads[1], 'Y': writes[0], 'Y.rs': writes[1]} if forward else
                   {'GY': reads[0], 'X': reads[1], 'Y': reads[2], 'W': reads[3],
                    'Y.rs': reads[4], 'GX': writes[0], 'W.grad': writes[1]})
    elif op == 'rope':
        if e % (r * heads):
            raise ValueError('rope elements must split into rows and heads')
        d = replace(d, head_dim=e // (r * heads))
        if forward:
            originals = eager.rope_fwd('X', 'Y', heads, d,
                                       input_bytes=eager.F32 if model_type == 'qwen3' else eager.BF16)
        else:
            originals = (eager.qwen3_rope_bwd('X', 'Y', heads, d) if model_type == 'qwen3' else
                         eager.rope_bwd('X', 'Y', heads, d))
        mapping = {'X': reads[0], 'cos': reads[1], 'sin': reads[2], 'Y': writes[0], 'Y.t': writes[0]}
    elif op == 'activation':
        originals = (eager.activation_fwd('G', 'U', 'S', 'Y', d) if forward else
                     eager.activation_bwd('GY', 'G', 'U', 'S', 'GS', 'GG', 'GU', d))
        mapping = ({'G': reads[0], 'U': reads[1], 'Y': writes[0]} if forward else
                   {'GY': reads[0], 'G': reads[1], 'U': reads[2], 'GG': writes[0], 'GU': writes[1]})
    elif op == 'residual_add':
        originals = [eager.residual_add(op, 'X', 'Y', 'Z', d, direction)]
        mapping = {'X': reads[0], 'Y': reads[1], 'Z': writes[0]}
    else:
        originals = eager.loss_fwd(d, opts) if forward else eager.loss_bwd(d, opts)
        mapping = ({'logits': reads[0], 'ids': reads[1], 'loss': writes[0],
                    'total_weight': writes[1]} if forward else
                   {'logits': reads[0], 'total_weight': reads[1], 'ids': reads[2],
                    'glogits': writes[0]})

    if fused:
        return [Kernel(op, direction + '.' + op + '.fused', 'pointwise', reads, writes,
                       sum(k.flops for k in originals), None, 'train-' + op)]

    def convert(access):
        if access.tensor in mapping:
            return replace(mapping[access.tensor], nbytes=access.nbytes, weight=access.weight)
        return replace(access, tensor=f'{prefix}.{op}.{access.tensor}')
    return [replace(k, op=op, reads=tuple(map(convert, k.reads)),
                    writes=tuple(map(convert, k.writes))) for k in originals]
