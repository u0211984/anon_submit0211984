import json

from count_definitions import read_hopper_csv, UNCONFIRMED, UNUSED_CANDIDATE_ALIASES, HOPPER_LAUNCH_METRICS

def overlay_raw(rows, path):
    measured = read_hopper_csv(path)
    ids = [int(row['id']) for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(measured):
        raise ValueError('Hopper raw/per-kernel CSV launch IDs disagree or repeat')
    for row in rows:
        row.update(measured[int(row['id'])])
    return measured

def write_summary(path, rows, measured):
    fields = {'hbm_bytes': 'dram_bytes', 'l2_bytes': 'l2_bytes',
              'smem_bytes': 'smem_model', 'tensor_flops': 'tensor_flops',
              'vector_fp16_weighted_ops': 'vector_fp16_flops',
              'vector_fp32_flops': 'vector_fp32_flops',
              'l2_lookup_bytes': 'l2_lookup_bytes'}
    totals = {name: sum(row[field] for row in measured.values()) for name, field in fields.items()}
    path.write_text(json.dumps(dict(architecture='sm90', totals=totals,
        metric_presence_confirmed=sorted(UNCONFIRMED),
        metric_semantics='UNCONFIRMED: presence does not establish units, coverage or disjointness',
        packed_half_normalization='UNCONFIRMED; half and single counters are kept separate',
        unused_unconfirmed_candidates=list(UNUSED_CANDIDATE_ALIASES),
        launch_metrics_missing={kernel: [name for name in HOPPER_LAUNCH_METRICS if name not in record]
                                for kernel, record in measured.items()},
        unassigned_kernels=[dict(id=int(row['id']), kernel=row['kernel'], label=row['label']) for row in rows],
        kernels=measured), indent=2) + '\n')

def annotate_markdown(parts):
    return [part.replace('ld+st+ldsm', 'ld+st+ldsm+gmma+stsm')
            .replace('SMEM measured reads (ld + ldsm)', 'SMEM legacy reads (ld + ldsm; GMMA excluded here)')
            .replace('writes (st)', 'legacy writes (st; STSM excluded here)')
            .replace('Non-GEMM kernels show about one HMMA-pipe instruction per warp (counter definition, bin d).',
                     'Hopper uses ops_path sums; full-launch totals are in the JSON sidecar.')
            for part in parts]
