import csv, json, re, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _environment import reject_model_environment
reject_model_environment()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--raw-dir', type=Path, required=True)
parser.add_argument('--config-dir', type=Path, required=True)
parser.add_argument('--out-dir', type=Path, required=True)
parser.add_argument('variant')
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)
V = args.variant
CAST_ON = V.startswith('v2') or V.startswith('m12')
RAW = args.raw_dir
COUNTERS = ('hbm_read', 'hbm_write', 'l2', 'smem', 'tensor_useful', 'tensor_padding', 'vector')
CLASSES = ('gemm', 'attention', 'elementwise', 'norm', 'embedding', 'optimizer', 'other')

MOE_ROLE_CLASSES = {'moe_router': 'gemm', 'moe_w13': 'gemm', 'moe_w2': 'gemm', 'moe_routing': 'elementwise',
                    'moe_align': 'elementwise', 'moe_swiglu': 'elementwise', 'moe_sum': 'elementwise'}

def counts(a):
    useful, issued = float(a['useful_tensor_flops']), float(a['issued_tensor_flops'])
    tr = lambda lv: float(a[lv]['read_bytes']) + float(a[lv]['write_bytes'])
    return {'hbm_read': float(a['hbm']['read_bytes']), 'hbm_write': float(a['hbm']['write_bytes']),
            'l2': tr('l2'), 'smem': tr('smem'), 'tensor_useful': useful,
            'tensor_padding': max(0.0, issued - useful), 'vector': float(a['vector_flops'])}

def add(dst, src, w=1.0):
    for k in COUNTERS:
        dst[k] += src[k] * w

def zero():
    return {k: 0.0 for k in COUNTERS}

def split(op_counts, parts):
    total = zero()
    for p in parts.values():
        add(total, p)
    out = {c: zero() for c in parts}
    dev = 0.0
    for k in COUNTERS:
        if total[k] > 0:
            dev = max(dev, abs(op_counts[k] / total[k] - 1))
            for c, p in parts.items():
                out[c][k] = op_counts[k] * p[k] / total[k]
        elif op_counts[k] > 0 and k == 'hbm_write' and total['hbm_read'] > 0:
            for c, p in parts.items():
                out[c][k] = op_counts[k] * p['hbm_read'] / total['hbm_read']
        elif op_counts[k] != 0:
            first = next(iter(parts))
            out[first][k] = op_counts[k]
    return out, dev

def workload(path):
    d = json.loads(path.read_text())
    phase = {'train': 'train', 'prefill': 'prefill', 'decode': 'decode'}[d['phase']]
    p = d['phases'][phase]
    want_cls = {'train': 'TimeCalculationLLM', 'prefill': 'TimeCalculationLLMInference',
                'decode': 'TimeCalculationLLMInference.decode'}[phase]
    tpls = [e for e in d['events'] if e['kind'] == 'template' and e['cls'] == want_cls]
    tpl = tpls[-1]['timings']

    def tpart(name, direction):
        a = (tpl.get(name) or {}).get(direction)
        return counts(a) if a else zero()

    import yaml as _yaml
    hwp = _yaml.safe_load(open(d['hardware_config']))['sw_param']['precision']
    nb = {'fp32': 4.0, 'float32': 4.0, 'bf16': 2.0, 'fp16': 2.0}
    P, C, G = nb[str(hwp['parameters'])], nb[str(hwp['tensor_format'])], nb[str(hwp['gradients'])]
    cfg = json.load(open(args.config_dir / f"{d['model']}.json"))
    hh, H, K, I, V = cfg['hidden_size'], cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg['intermediate_size'], cfg['vocab_size']
    dd = hh // H
    cast_params = {'qkv_proj': hh * (H * dd + 2 * K * dd), 'output_proj': H * dd * hh, 'MLP': 3 * hh * I, 'linear_softmax': V * hh}
    impl = str((_yaml.safe_load(open(d['model_config'])).get('model_param') or {}).get('implementation', 'fused'))
    eager_train = phase == 'train' and impl == 'eager'
    has_cast = phase == 'train' and P != C and CAST_ON and not eager_train

    def trace_part(base, direction):
        if not eager_train:
            return None
        tname = base[:-2] if base in ('embedding_b', 'linear_softmax_b') else base
        names = ('ffn1', 'gelu', 'ffn2') if tname == 'MLP' else (tname,)
        out, hit = zero(), False
        for n in names:
            a = (tpl.get(n) or {}).get(direction + '_eager_trace')
            if a:
                add(out, counts(a)); hit = True
        return out if hit else None

    def cast_counts(base, direction):
        n = cast_params.get('linear_softmax' if base.startswith('linear_softmax') else base, 0)
        r, w = (n * P, n * C) if direction == 'forward' else (n * C, n * G)
        out = zero(); out['hbm_read'], out['hbm_write'], out['l2'] = r, w, r + w
        return out

    def take_cast(oc, base, direction):
        if not has_cast or (base.split('_b')[0] if base.startswith('linear_softmax') else base) not in cast_params:
            return oc, None
        c = cast_counts(base, direction)
        rest = {k: oc[k] - c[k] for k in COUNTERS}
        assert all(rest[k] >= -1e-6 * max(1.0, oc[k]) for k in COUNTERS), (base, direction, oc, c)
        return rest, c

    ev = {}
    for e in d['events']:
        if e['kind'] in ('gemm', 'roof'):
            if phase == 'prefill' and e['name'] in ev:
                continue
            ev[e['name']] = counts(e['activity'])
    res = {c: zero() for c in CLASSES}
    roles = {}
    def addr(cls, role, v):
        add(res[cls], v)
        add(roles.setdefault((cls, role), zero()), v)
    devs = []
    for name, rec in p['by_operation'].items():
        oc = counts(rec['activity'])
        m = re.match(r'^(?P<op>[A-Za-z0-9_]+?)_(?P<dir>forward|backward)_layer', name)
        base = m.group('op') if m else re.sub(r'_mb\d+.*$', '', name)
        direction = (m.group('dir') if m else ('backward' if base.endswith('_b') else 'forward'))
        oc, cast = take_cast(oc, base, direction)
        if cast is not None:
            addr('elementwise', 'cast', cast)
        tp = trace_part(base, direction)
        if tp is not None:
            tname = base[:-2] if base in ('embedding_b', 'linear_softmax_b') else base
            if tname in ('layernorm1', 'layernorm2', 'embedding') or (tname == 'MLP' and False):
                pass
            else:
                oc = {k: oc[k] - tp[k] for k in COUNTERS}
                addr('elementwise', f'trace_{tname}', tp)
        if base == 'MLP':
            def tcast(name, n):
                t = tpart(name, direction)
                tr = (tpl.get(name) or {}).get(direction + '_eager_trace') if eager_train else None
                if tr:
                    t = dict(t); trc = counts(tr)
                    for kk in COUNTERS:
                        t[kk] -= trc[kk]
                if not has_cast:
                    return t
                r, w = (n * P, n * C) if direction == 'forward' else (n * C, n * G)
                t = dict(t); t['hbm_read'] -= r; t['hbm_write'] -= w; t['l2'] -= r + w
                return t
            parts, dev = split(oc, {'mlp_gate_up': tcast('ffn1', 2 * hh * I), 'mlp_down': tcast('ffn2', hh * I),
                                    'swiglu': tcast('gelu', 0)})
            devs.append(dev)
            addr('gemm', 'mlp_gate_up', parts['mlp_gate_up']); addr('gemm', 'mlp_down', parts['mlp_down'])
            addr('elementwise', 'swiglu', parts['swiglu'])
        elif base in ('linear_softmax', 'linear_softmax_b'):
            s = 'b' if base.endswith('_b') else 'f'
            parts, dev = split(oc, {'gemm': ev.get(f'linear_softmax_{s}', zero()),
                                    'elementwise': ev.get(f'pointwise-linear-softmax-{s}', zero()),
                                    'norm': ev.get(f'pointwise-final-norm-{s}', zero())})
            devs.append(dev)
            addr('gemm', 'lm_head', parts['gemm']); addr('elementwise', 'head_softmax', parts['elementwise'])
            if any(parts['norm'].values()):
                addr('norm', 'final_norm', parts['norm'])
        else:
            cls = {'qkv_proj': 'gemm', 'output_proj': 'gemm', 'attention': 'attention', 'rope': 'elementwise',
                   'layernorm1': 'norm', 'layernorm2': 'norm', 'embedding': 'embedding', 'embedding_b': 'embedding',
                   **MOE_ROLE_CLASSES}.get(base)
            if cls is None and base.startswith('optimizer'):
                cls = 'optimizer'
            role = {'qkv_proj': 'qkv', 'output_proj': 'o'}.get(base, base)
            addr(cls or 'other', role, oc)
            if cls is None and not base.startswith('optimizer'):
                print(f'  other: {path.stem} {name}', file=sys.stderr)
    total = zero()
    for v in res.values():
        add(total, v)
    check = counts(p['activity'])
    for k in COUNTERS:
        if check[k] and abs(total[k] / check[k] - 1) > 1e-9:
            raise ValueError(f'{path.stem}: class sum {k} {total[k]} != phase activity {check[k]}')
    return d, res, total, max(devs) if devs else 0.0, roles

rows, md = [], []
md.append(f'# Predicted whole-step counts, variant {V}\n\n')
md.append(f'Tree: {(RAW / "TREE_COMMIT.txt").read_text().strip()}\n\n')
md.append('| workload | HBM read (GB) | HBM write (GB) | L2 (GB) | SMEM (GB) | tensor useful (GFLOP) | tensor padding (GFLOP) | vector (GFLOP) | composite split deviation |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n')
for f in sorted(RAW.glob('*.json')):
    d, res, total, dev, roles = workload(f)
    wl = f'{d["model"]}_{d["phase"]}_b{d["batch"]}_s{d["seq_len"]}' + ('_nockpt' if d.get('nockpt') else '')
    mods = d['modules']; tree = d['tree']
    assert all(v.startswith(tree + '/') for v in mods.values()), (wl, mods)
    for c in CLASSES + ('total',):
        v = total if c == 'total' else res[c]
        for k in COUNTERS:
            rows.append({'variant': V, 'workload': wl, 'model': d['model'], 'phase': d['phase'], 'op_class': c,
                         'role': '', 'counter': k, 'value': v[k]})
    for (c, role), v in sorted(roles.items()):
        for k in COUNTERS:
            rows.append({'variant': V, 'workload': wl, 'model': d['model'], 'phase': d['phase'], 'op_class': c,
                         'role': role, 'counter': k, 'value': v[k]})
    md.append(f'| {wl} | {total["hbm_read"]/1e9:.3f} | {total["hbm_write"]/1e9:.3f} | {total["l2"]/1e9:.3f} | {total["smem"]/1e9:.3f} | '
              f'{total["tensor_useful"]/1e9:.1f} | {total["tensor_padding"]/1e9:.1f} | {total["vector"]/1e9:.2f} | {dev:.1e} |\n')
with (args.out_dir / f'pred_{V}.csv').open('w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
(args.out_dir / f'pred_{V}_totals.md').write_text(''.join(md))
print(''.join(md))
