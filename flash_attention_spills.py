from dataclasses import replace

def select_program(spec, kind, splits=1):
    from flash_attention import _compiled_resources
    if spec.source == 'dao-pip-unknown':
        raise NotImplementedError('S1b spills require the pinned vLLM or torch binary; the pip binary is unverified')
    build = spec.source
    m = (128 if kind == 'fwd' or spec.head_dim == 64 else 64)
    n = (128 if spec.head_dim == 64 else 64) if kind == 'fwd' else (256 if spec.head_dim == 64 else 128)
    if kind.startswith('bwd'):
        n = 128
    even = spec.q_len % m == 0 and spec.kv_len % n == 0
    if kind == 'fwd':
        even = even and build != 'vllm'
        flags = [False, spec.causal, False, False, even, True, False, False]
    elif kind == 'splitkv':
        m = 64
        even = spec.q_len % m == 0 and spec.kv_len % n == 0 and spec.kv_layout != 'paged'
        flags = [spec.causal, False, False, even, True, False, splits > 1, False]
    elif kind == 'bwd_dq_dk_dv':
        flags = [False, spec.causal, False, False, even, True, False]
    else:
        return {'zero_spills': True, 'regions': [], 'symbol': None}
    entry = _compiled_resources()['builds'][build][kind][str(spec.head_dim)]
    found = [v for v in entry['sass_spills']['variants'] if v['program'] and v['program']['flags'] == flags]
    if len(found) != 1:
        raise NotImplementedError(f'no unique SASS spill program for {build}/{kind}/d{spec.head_dim}, flags={flags}')
    program = dict(found[0]['program'], symbol=found[0]['symbol'])
    program['regions'] = {r['name']: r for r in program['regions']}
    return program

def regions(program, phase, index=0, total=0):
    if not program or program.get('zero_spills') or total == 0:
        return []
    rs = program['regions']
    if phase in ('prologue', 'epilogue'):
        return [rs[phase]]
    masked = program['masked_copies']
    if index < masked:
        result = [rs[f'masked_{index}']]
        if index == min(total, masked)-1 and 'masked_exit' in rs:
            result.append(rs['masked_exit'])
        return result
    names = ['loop']
    if index == masked and 'loop_preheader' in rs:
        names.insert(0, 'loop_preheader')
    if index < total-1 and 'loop_prefetch_helper' in rs:
        names.append('loop_prefetch_helper')
    return [rs[name] for name in names]

def byte_counts(rs, total):
    entries = [e for r in rs for e in r['entries']
               if e.get('min_tiles', 0) <= total <= e.get('max_tiles', total)]
    return (sum(e['bytes'] for e in entries if e['opcode'].startswith('LDL')),
            sum(e['bytes'] for e in entries if e['opcode'].startswith('STL')))

def totals(program, tiles, threads):
    loads = stores = 0
    for nt in tiles:
        rs = regions(program, 'prologue', total=nt) + regions(program, 'epilogue', total=nt)
        for index in range(nt):
            rs += regions(program, 'tile', index, nt)
        read, write = byte_counts(rs, nt)
        loads += read * threads
        stores += write * threads
    return loads, stores

def apply_counts(kernel, program, tiles, threads):
    loads, stores = totals(program, tiles, threads)
    diag = dict(kernel.diagnostics, spill_symbol=program.get('symbol'),
                l1_spill_read_bytes=loads, l1_spill_write_bytes=stores,
                l2_spill_write_bytes=stores,
                spill_load_scope='static path upper bound; all loads priced at L1, prefetch/lane predication not dynamically resolved',
                spill_store_policy='write_through; one L2 write per executed store byte',
                spill_resource='shared per-SM smem service; A100 whitepaper printed p.33',
                spill_l2_scope='delivered writes only; logical tensor IO and HBM simulation unchanged')
    return replace(kernel, l2_write_bytes=kernel.l2_write_bytes+stores, diagnostics=diag)

def service(program, phase, index, total, deps, node, smem_resource,
            smem_rate, l2_rate, warps, pending_writes, *,
            smem_latency_cycles, l2_latency_cycles):
    rs = regions(program, phase, index, total)
    load, store = byte_counts(rs, total)
    if not load and not store:
        return tuple(deps)

    done = [node(deps, smem_resource, (load+store)*32/smem_rate,
                 smem_latency_cycles if load else 0) for _ in range(warps)]
    if store:
        pending_writes.append(node(done, 'l2', store*32*warps/l2_rate, l2_latency_cycles))
    return tuple(done)
