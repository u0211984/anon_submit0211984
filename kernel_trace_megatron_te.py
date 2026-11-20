from kernel_trace import Kernel
from kernel_trace_train import ADAMW_FLOPS, F32
from l2_sim import RAccess as A

INT32_MAX = 2 ** 31 - 1
MTA_CHUNK = 2048 * 32
MTA_MAX_TENSORS = (110, 64, 48, 36, 30, 24)
MTA_MAX_BLOCKS = 320

def _k(op, name, reads, writes, flops=0.0, kind='pointwise'):
    return Kernel(op, name, kind, tuple(reads), tuple(writes), float(flops), None, 'megatron-' + name)

def _part(access, fraction):
    return A(access.tensor, access.nbytes * fraction, access.weight, getattr(access, 'offset', 0.0))

def rope(direction, x, out, freqs, *, elements, width, prefix):
    e, table = float(elements), freqs.nbytes / F32
    t = lambda name, n: A(f'{prefix}.{name}', float(n), False, 0.0)       # noqa: E731
    k = lambda role, rd, wr, f=0.0: _k('rope', f'{direction}.rope.{role}', rd, wr, f)   # noqa: E731
    cos, sin = t('cos', width * table), t('sin', width * table)
    if direction == 'forward':
        ks = []
        for fn, cast in (('cos', cos), ('sin', sin)):
            ks += [k(fn, [freqs], [t(fn + '32', F32 * table)], table),
                   k(fn + '_scale', [t(fn + '32', F32 * table)], [t(fn + '32s', F32 * table)], table),
                   k(fn + '_cast', [t(fn + '32s', F32 * table)], [cast])]
        return ks + [
            k('mul_cos', [x, cos], [t('c', width * e)], e),
            k('neg', [_part(x, 0.5)], [t('neg', width * e / 2)], e / 2),
            k('cat_rot', [t('neg', width * e / 2), _part(x, 0.5)], [t('rot', width * e)]),
            k('mul_sin', [t('rot', width * e), sin], [t('s', width * e)], e),
            k('add', [t('c', width * e), t('s', width * e)], [t('sum', width * e)], e),
            k('cat_pass', [t('sum', width * e)], [out]),
        ]
    grot, gt = t('grot', width * e), t('gt', width * e)
    return [
        k('mul_sin_grad', [x, sin], [grot], e),
        k('neg_bwd', [_part(grot, 0.5)], [t('gx2', width * e / 2)], e / 2),
        k('chunk_cat', [t('gx2', width * e / 2), _part(grot, 0.5)], [t('gchunk', width * e)]),
        k('mul_cos_grad', [x, cos], [t('gc', width * e)], e),
        k('accum', [t('gchunk', width * e), t('gc', width * e)], [gt], e),
        k('slice_zeros', [], [out]),
        k('slice_add', [gt, out], [out], e),
    ]

def norm_backward(reads, writes, *, sm_count, width, flops, prefix):
    gy, x, _y, w, rs = reads
    gx, dw = writes
    part = A(f'{prefix}.norm.dgamma_part', F32 * sm_count * width, False, 0.0)
    return [_k('norm', 'backward.norm.bwd', [gy, x, w, rs], [gx, part], flops),
            _k('norm', 'backward.norm.finalize', [part], [dw], sm_count * width)]

def native_loss(direction, reads, writes, *, rows, vocab, prefix):
    r, e = float(rows), float(rows) * vocab
    t = lambda name, n: A(f'{prefix}.{name}', float(n), False, 0.0)       # noqa: E731
    k = lambda role, rd, wr, f=0.0: _k('loss', f'{direction}.loss.{role}', rd, wr, f)   # noqa: E731
    l32, mx, sm = t('logits32', F32 * e), t('max', F32 * r), t('sum', F32 * r)
    pred, softmax = t('predicted', F32 * r), t('softmax', F32 * e)
    if direction == 'forward':
        logits, labels = reads
        loss, pack = writes
        return [
            k('float_max', [logits], [l32, mx], e),
            k('target_gather', [labels, _part(l32, 1.0 / vocab), mx], [pred], r),
            k('pack', [pred], [pack]),
            k('exp_sub_sum', [l32, mx], [l32, sm], 3 * e),
            k('log_sub', [sm, pack], [loss], 2 * r),
            k('div', [l32, sm], [softmax], e),
        ]
    _logits, _pack, labels, gloss = reads
    glogits, = writes
    g32 = t('grad32', F32 * e)
    return [
        k('index_sub', [softmax, labels], [g32], r),
        k('index_sub_update', [labels], [t('update', F32 * r)], r),
        k('mul_cast', [g32, gloss], [glogits], e),
    ]

EMBEDDING_SORT_THRESHOLD = 3072

def embedding_backward_sorted(grad, tokens, rows, *, count, hidden, prefix):
    i64 = tokens.nbytes
    t = lambda name, n=i64: A(f'{prefix}.{name}', float(n), False, 0.0)       # noqa: E731
    k = lambda role, rd, wr, f=0.0: _k('embedding', f'backward.embedding.{role}', rd, wr, f)   # noqa: E731
    keys, values = t('sort_keys'), t('sort_values')
    before = [k('arange', [], [values]),
              k('sort_histogram', [tokens], [t('histogram', 8 * 256 * 8)]),
              k('sort_exclusive_sum', [t('histogram', 8 * 256 * 8)], [t('histogram', 8 * 256 * 8)])]
    source = tokens
    for sweep in range(8):
        before.append(k('sort_onesweep', [source, values], [keys, values]))
        source = keys
    after = [k('unique_init', [], [t('tile_state', 64)]),
             k('unique_sweep', [keys], [t('segment_offsets')]),
             k('partials_per_segment', [t('segment_offsets')], [t('partials_per_segment')]),
             k('scan_init', [], [t('tile_state', 64)]),
             k('scan', [t('partials_per_segment')], [t('partial_offsets')]),
             k('partial_segment_count', [t('partials_per_segment', 8), t('partial_offsets', 8)], [t('partial_count', 8)]),
             k('partial_segment_offset', [t('partial_offsets'), t('segment_offsets')], [t('partial_segments')]),
             k('compute_grad_weight', [grad, values, t('partial_segments')], [t('partials', grad.nbytes)], count * hidden),
             k('sum_and_scatter', [t('partials', grad.nbytes), keys], [rows], count * hidden)]
    return before, after

def split_32bit(ranges, itemsize):
    total = sum(int(n) for _, n in ranges)

    def pieces(start, count):
        if count <= INT32_MAX and 1 + (count - 1) * int(itemsize) <= INT32_MAX:
            return [(start, count)]
        half = count // 2
        return pieces(start, half) + pieces(start + half, count - half)
    out = []
    for start, count in pieces(0, total):
        launch, position = [], 0
        for name, n in ranges:
            n = int(n)
            lo, hi = max(start, position), min(start + count, position + n)
            if hi > lo:
                launch.append((name, lo - position, hi - lo))
            position += n
        out.append(launch)
    return out

def mta_launches(sizes, depth):
    max_tensors = MTA_MAX_TENSORS[depth - 1]
    launches, current, tensors, blocks = [], [], 0, 0
    last = len(sizes) - 1
    for index, size in enumerate(sizes):
        size = int(size)
        chunks = (size + MTA_CHUNK - 1) // MTA_CHUNK
        tensors += 1
        begin = 0
        for chunk in range(chunks):
            blocks += 1
            final = chunk == chunks - 1
            if (tensors == max_tensors and final) or blocks == MTA_MAX_BLOCKS or (index == last and final):
                end = min(size, (chunk + 1) * MTA_CHUNK)
                current.append((index, begin, end - begin))
                launches.append(current)
                current, blocks, begin = [], 0, end
                tensors = 0 if final else 1
        if begin < size:
            current.append((index, begin, size - begin))
    return launches

def fused_adam(weights, precision):
    names = list(weights)
    ks = []
    for launch in mta_launches([weights[n] for n in names], 4):
        reads, writes, elements = [], [], 0
        for index, offset, n in launch:
            name = names[index]
            grad = A(name + '.main_grad', precision.gradients * n, False, precision.gradients * offset)
            main = A(name + '.master', precision.master_parameters * n, False, precision.master_parameters * offset)
            m1, m2 = (A(name + s, precision.optimizer_states * n, False, precision.optimizer_states * offset)
                      for s in ('.m1', '.m2'))
            reads += [main, grad, m1, m2]
            writes += [main, m1, m2]
            elements += n
        ks.append(_k('optimizer', 'optimizer.te_fused_adam', reads, writes, ADAMW_FLOPS * elements))
    return ks
