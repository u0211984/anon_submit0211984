import csv
import math

HOPPER_SUM_METRICS = (
    'dram__sectors_read.sum', 'dram__sectors_write.sum',
    *(f'lts__t_sectors_srcunit_tex_op_{d}_lookup_{h}.sum' for d in ('read', 'write') for h in ('hit', 'miss')),
    *(f'sm__sass_data_bytes_mem_shared_op_{op}.sum' for op in ('ld', 'st')),
    'sm__sass_data_bytes_mem_global_op_ldgsts.sum', 'sm__sass_data_bytes_mem_shared_op_ldsm.sum',
    'sm__inst_executed_pipe_tensor_op_hmma.sum',
    *(f'sm__sass_thread_inst_executed_op_{op}_pred_on.sum' for op in ('hadd', 'hmul', 'hfma', 'fadd', 'fmul', 'ffma')),
    'gpu__time_duration.sum', 'l1tex__m_xbar2l1tex_read_bytes.sum', 'l1tex__m_l1tex2xbar_write_bytes.sum',
    'sm__inst_executed_pipe_tensor_op_gmma.sum', 'sm__sass_data_bytes_mem_shared_op_gmma.sum',
    'l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum',
    'l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld_dest_multicast.sum',
    'sm__sass_data_bytes_mem_shared_op_stsm.sum',
    'l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum', 'l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum',
    'sm__ops_path_tensor_src_fp16.sum', 'sm__ops_path_tensor_src_bf16_dst_fp32.sum',
    'sm__ops_path_tensor_src_tf32_dst_fp32.sum',
)

UNCONFIRMED = frozenset(HOPPER_SUM_METRICS)
PACKED_HALF_NORMALIZATION = 'UNCONFIRMED'

HOPPER_LAUNCH_METRICS = (
    'launch__registers_per_thread', 'launch__shared_mem_per_block_dynamic',
    'launch__shared_mem_per_block_static', 'launch__grid_size', 'launch__block_size',
    'launch__cluster_dim_x', 'launch__cluster_dim_y', 'launch__cluster_dim_z',
    'launch__cluster_max_active', 'launch__cluster_max_potential_size',
    'launch__cluster_scheduling_policy',
)

UNUSED_CANDIDATE_ALIASES = (
    'dram__bytes_read.sum', 'dram__bytes_write.sum',
    'l1tex__m_xbar2l1tex_read_sectors_mem_dshared.sum',
    'l1tex__m_l1tex2xbar_write_sectors_mem_dshared.sum',
    'l1tex__m_xbar2l1tex_read_sectors_mem_global_op_tma_ld.sum',
    'sm__sass_data_bytes_mem_shared.sum',
    'sm__ops_path_tensor_src_fp16_dst_fp16.sum',
    'sm__ops_path_tensor_src_fp16_dst_fp32.sum',
    *(f'sm__ops_path_tensor_src_{op}_sparsity_off.sum' for op in
      ('fp16_dst_fp16', 'fp16_dst_fp32', 'bf16_dst_fp32', 'tf32_dst_fp32')),
    *(f'smsp__sass_thread_inst_executed_op_{op}_pred_on.sum' for op in ('hadd', 'hmul', 'hfma', 'fadd', 'fmul', 'ffma')),
)

def _number(value, kernel, name):
    try:
        result = float(value.replace(',', ''))
    except (ValueError, AttributeError):
        raise ValueError(f'kernel {kernel}: invalid {name}: {value!r}') from None
    if not math.isfinite(result):
        raise ValueError(f'kernel {kernel}: nonfinite {name}: {value!r}')
    return result

def _derive(m, kernel):
    missing = set(HOPPER_SUM_METRICS) - m.keys()
    if missing:
        raise ValueError(f'kernel {kernel}: missing required metrics: {", ".join(sorted(missing))}')
    for name in HOPPER_SUM_METRICS:
        m[name] = _number(m[name], kernel, name)
    g = m.__getitem__
    r = g('l1tex__m_xbar2l1tex_read_bytes.sum') - g('l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum')
    w = g('l1tex__m_l1tex2xbar_write_bytes.sum') - g('l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum')
    for direction, value in (('read', r), ('write', w)):
        if value < 0:
            raise ValueError(f'kernel {kernel}: negative L2 {direction} residual: {value}')
    m['time_ns'] = g('gpu__time_duration.sum')
    for direction in ('read', 'write'):
        m[f'dram_{direction}_bytes'] = 32 * g(f'dram__sectors_{direction}.sum')
    m['dram_bytes'] = m['dram_read_bytes'] + m['dram_write_bytes']
    for direction in ('read', 'write'):
        m[f'l2_lookup_{direction}_bytes'] = 32 * sum(g(f'lts__t_sectors_srcunit_tex_op_{direction}_lookup_{h}.sum') for h in ('hit', 'miss'))
    m['l2_lookup_bytes'] = m['l2_lookup_read_bytes'] + m['l2_lookup_write_bytes']
    m['xbar2l1_read_bytes'], m['l12xbar_write_bytes'] = r, w
    m['l2_delivered_read_bytes'], m['l2_delivered_write_bytes'] = r, w
    m['l2_bytes'] = r + w
    for op in ('ld', 'st', 'ldsm', 'gmma', 'stsm'):
        m[f'smem_{op}'] = g(f'sm__sass_data_bytes_mem_shared_op_{op}.sum')
    m['smem_ldgsts'] = g('sm__sass_data_bytes_mem_global_op_ldgsts.sum')
    m['smem_model'] = sum(m[f'smem_{op}'] for op in ('ld', 'st', 'ldsm', 'gmma', 'stsm'))
    m['smem_old'] = m['smem_ld'] + m['smem_st'] + m['smem_ldgsts']
    m['smem_with_ldsm'] = m['smem_old'] + m['smem_ldsm']
    m['tensor_flops'] = sum(g(f'sm__ops_path_tensor_src_{op}.sum') for op in ('fp16', 'bf16_dst_fp32', 'tf32_dst_fp32'))
    m['tensor_flops_old'] = m['tensor_flops']
    for precision, ops in (('fp16', ('hadd', 'hmul', 'hfma')), ('fp32', ('fadd', 'fmul', 'ffma'))):
        m[f'vector_{precision}_flops'] = sum(weight * g(f'sm__sass_thread_inst_executed_op_{op}_pred_on.sum') for op, weight in zip(ops, (1, 1, 2)))
    return m

def read_hopper_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        lines = (line for line in f if line.startswith('"'))
        reader = csv.DictReader(lines)
        if not reader.fieldnames or not {'ID', 'Metric Name', 'Metric Value'} <= set(reader.fieldnames):
            raise ValueError(f'{path}: missing NCU CSV header')
        kernels = {}
        for row in reader:
            try:
                kernel = int(row['ID'])
            except (ValueError, TypeError):
                raise ValueError(f'{path}: invalid kernel ID {row.get("ID")!r}') from None
            name = row['Metric Name']
            m = kernels.setdefault(kernel, {})
            if name in m:
                raise ValueError(f'kernel {kernel}: duplicate metric {name}')
            m[name] = row['Metric Value']
    if not kernels:
        raise ValueError(f'{path}: no kernel metrics')
    return {kernel: _derive(m, kernel) for kernel, m in kernels.items()}

def read_counts_csv(path, architecture):
    if architecture == 'sm90':
        return read_hopper_csv(path)
    if architecture == 'sm80':
        raise ValueError('sm80 uses the existing legacy scorer; raw overlay is sm90 only')
    raise ValueError(f'unsupported architecture: {architecture}')
