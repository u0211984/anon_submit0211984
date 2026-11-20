import math

from kernel_trace import RESIDUAL_ADD_FLOPS, RMSNORM_SCALE_WEIGHT_FLOPS, RMSNORM_SQUARE_ACC_FLOPS

def estimate(k, *, sms, l2_latency, smem_latency, l2_bw, vector_rate,
             architecture=None, clock_hz=0):
    if (architecture != 'sm80' or clock_hz <= 0 or
            min(sms, l2_latency, l2_bw, vector_rate) <= 0 or smem_latency < 0):
        return None
    name = k.name
    if name in ('rms_norm', 'fused_add_rms_norm'):
        if len(k.reads) < 2 or not k.reads[-1].nbytes:
            return None
        row_bytes = k.reads[-1].nbytes
        rows_f = k.writes[-1].nbytes / row_bytes
        if not rows_f.is_integer() or row_bytes % 2:
            return None
        rows, width = int(rows_f), int(row_bytes / 2)
        if not rows or rows >= sms:
            return None
        fused = name == 'fused_add_rms_norm'

        if fused and width % 8:
            return None
        threads = min(width, 1024 if not fused or rows < 256 else 256)
        vector = 16 if fused else 2
        active = min(threads, math.ceil(row_bytes / vector))

        pass_bytes = [row_bytes * (2 if fused else 1), row_bytes * 2]
        streams = [2 if fused else 1, 2]

        per_pass_flops = [((RESIDUAL_ADD_FLOPS if fused else 0.0) + RMSNORM_SQUARE_ACC_FLOPS) * width,
                          RMSNORM_SCALE_WEIGHT_FLOPS * width]
        reductions = 1
    elif name in ('sampler.softmax', 'sampler.log_softmax', 'sampler.argmax'):
        if len(k.reads) != 1 or len(k.writes) != 1:
            return None
        if name == 'sampler.argmax':
            rows_f = k.writes[0].nbytes / 8
        else:
            rows_f = getattr(k, 'smallgrid_rows', 0)
        if name == 'sampler.argmax':
            rows = int(rows_f)
        else:
            rows = int(rows_f)
        if rows <= 0 or rows >= sms or k.reads[0].nbytes % (rows * 4):
            return None
        width = int(k.reads[0].nbytes / (rows * 4))

        if width < 49152 or width % 4:
            return None
        row_bytes = width * 4
        if name == 'sampler.argmax':
            threads, vector = 512, 16

            if math.ceil(width / threads) >= 256:
                return None
            pass_bytes, streams, per_pass_flops, reductions = [row_bytes], [1], [0], 1
        else:
            threads, vector = 1024, 16

            pass_bytes, streams = [row_bytes] * 3, [1] * 3
            per_pass_flops = [0, 12 * width, (12 if name == 'sampler.softmax' else 2) * width]
            reductions = 2
        active = min(threads, math.ceil(row_bytes / vector))
    else:
        return None
    if active <= 0:
        return None
    rate = vector_rate * rows / sms
    total = sum(max((n / (active * vector * s)) * l2_latency,
                    (n * rows) / l2_bw, f * rows / rate)
                for n, s, f in zip(pass_bytes, streams, per_pass_flops))
    if active > 32:
        if name in ('sampler.softmax', 'sampler.log_softmax') and architecture == 'sm80' and clock_hz > 0:
            tree_cycles = 3 * threads / 16 + 5 * (threads + 32) / 32
            total += reductions * (2 * smem_latency + tree_cycles / clock_hz)
        elif name in ('rms_norm', 'fused_add_rms_norm') and threads == 1024:
            tree_cycles = 2 * threads / 16 + 5 * threads / 32
            total += 2 * smem_latency + tree_cycles / clock_hz
        elif name == 'sampler.argmax':
            tree_cycles = 5 * threads / 16 + 5 * threads * 3 / 32
            total += 4 * smem_latency + tree_cycles / clock_hz
        else:
            total += reductions * smem_latency
    return total
