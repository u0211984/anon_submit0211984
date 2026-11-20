import csv, json, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _environment import reject_model_environment
reject_model_environment()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--capture-root', type=Path, required=True)
parser.add_argument('--vllm-capture-root', type=Path, required=True)
parser.add_argument('--capture-v2-root', type=Path, required=True)
parser.add_argument('--vllm-capture-v2-root', type=Path, required=True)
parser.add_argument('--config-dir', type=Path, required=True)
parser.add_argument('--prediction-dir', type=Path, required=True)
parser.add_argument('--out-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--architecture', choices=('sm80', 'sm90'), default='sm80')
args = parser.parse_args()
HERE = Path(__file__).resolve().parent
args.out_dir.mkdir(parents=True, exist_ok=True)
def join_args(root, capture_set):
    return ['--capture-dir', str(root / 'cap'), '--config-dir', str(args.config_dir),
            '--prediction-dir', str(args.prediction_dir), '--out-dir', str(args.out_dir),
            '--capture-set', capture_set, '--architecture', args.architecture]

V = ['m12_fused', 'm12_eager_l2sim', 'm12_eager_l2cls', 'm12_eager_l2cls_epi']
FINAL = 'm12_eager_l2cls_epi'

def cells(md, key):
    if args.architecture == 'sm90':
        key = key.replace('ld+st+ldsm', 'ld+st+ldsm+gmma+stsm')
    return [x.strip() for x in re.search(r'^\| ' + re.escape(key) + r' \|(.*)$', md, re.M).group(1).split('|') if x.strip()]

def l2_cells(md):
    m = re.search(r'^\| (L2 GB \(delivered\)|L2 lookups GB \(reference only[^|]*) \|(.*)$', md, re.M)
    c = [x.strip() for x in m.group(2).split('|') if x.strip()]
    if m.group(1) == 'L2 GB (delivered)':
        ref = float(re.search(r'beside: L2 lookups \(reference only, not L2 traffic\) ([\d.]+) GB', md).group(1)) * 1e9
        return c, float(c[0]) * 1e9, ref
    return c, '', float(c[0]) * 1e9

def l2_extra(meas, ref):
    return {'l2_definition': 'delivered' if meas != '' else 'not_captured', 'measured_l2_lookup_bytes_ref': ref}

def params(cfg):
    h, H, K, I, V_, L = (cfg['hidden_size'], cfg['num_attention_heads'], cfg['num_key_value_heads'],
                         cfg['intermediate_size'], cfg['vocab_size'], cfg['num_hidden_layers'])
    d = cfg.get('head_dim') or h // H
    per = h * (H * d + 2 * K * d) + H * d * h + 3 * h * I + 2 * h
    if cfg.get('model_type') == 'qwen3_moe':
        E, Im = cfg['num_experts'], cfg['moe_intermediate_size']
        per = h * (H * d + 2 * K * d) + H * d * h + E * 3 * h * Im + h * E + 2 * h + 2 * d
    return V_ * h * (1 if cfg.get('tie_word_embeddings') else 2) + L * per + h

man = list(csv.DictReader(open(args.capture_root / 'MANIFEST.csv')))
rows_all = man
notes_p = args.capture_root / 'MANIFEST_NOTES.md'
notes = notes_p.read_text() if notes_p.exists() else ''
tags = []
for r in man:
    if r['status'].startswith('scored') and r['tag'] not in tags:
        tags.append(r['tag'])
out = []
for t in tags:
    last = [r for r in rows_all if r['tag'] == t][-1]
    if t.startswith('SmolLM2-360M_decode') and 're-capture' not in last['status']:
        continue
    if not (args.capture_root / 'cap' / f'{t}_kernels.csv').exists():
        print(f'skip {t}: no per-kernel file (being re-captured?)', file=sys.stderr)
        continue
    model, phase, _, suffix = re.match(r'^(.*)_(prefill|decode|train)_(\d+x\d+)(_nockpt)?$', t).groups()
    subprocess.run([sys.executable, str(HERE / 'join.py')] + join_args(args.capture_root, 'v1') + [t] + V, stdout=subprocess.DEVNULL, check=True)
    md = (args.out_dir / f'{t}.md').read_text()
    cfg = json.load(open(args.config_dir / f'{model}.json'))
    base = {'capture_set': 'v1', 'measured_source': 'pytorch', 'model': model, 'phase': phase, 'batch': 1, 'seq_len': 1024,
            'params': params(cfg), 'in_counts_set': model != 'SmolLM2-135M',
            'grad_ckpt': phase == 'train' and model == 'Llama-3.2-3B' and not suffix}
    if args.architecture == 'sm90':
        base['_hopper_summary'] = str(args.out_dir / f'{t}_hopper.json')
    for key, name, unit in (('HBM GB', 'hbm_bytes', 1e9), ('L2', 'l2_bytes', 1e9),
                            ('SMEM GB (ld+st+ldsm)', 'smem_bytes', 1e9), ('vector GFLOP', 'vector_flops', 1e9)):
        if args.architecture == 'sm90' and name == 'vector_flops':
            continue
        if name == 'l2_bytes':
            c, meas, ref = l2_cells(md)
            extra = l2_extra(meas, ref)
        else:
            c = cells(md, key); meas, extra = float(c[0]) * unit, {}
        row = {**base, **extra, 'count': name, 'measured': meas, 'pred_fused': float(c[1]) * unit,
               'pred_eager_l2sim': float(c[3]) * unit, 'pred_eager_l2cls': float(c[5]) * unit,
               'pred_eager_l2cls_epi': float(c[7]) * unit}
        row['pred_final'] = row['pred_' + FINAL.replace('m12_', '')]
        out.append(row)
    m = re.search(r'tensor on GEMM \+ attention kernels only: measured ([\d.]+) GFLOP, (.*?)\. (?:Non|Hopper)', md)
    p = dict(re.findall(r'(m12_\w+) ([\d.]+) \(ratio', m.group(2)))
    row = {**base, 'count': 'tensor_flops', 'measured': float(m.group(1)) * 1e9,
           'pred_fused': float(p['m12_fused']) * 1e9, 'pred_eager_l2sim': float(p['m12_eager_l2sim']) * 1e9,
           'pred_eager_l2cls': float(p['m12_eager_l2cls']) * 1e9, 'pred_eager_l2cls_epi': float(p['m12_eager_l2cls_epi']) * 1e9}
    row['pred_final'] = row['pred_' + FINAL.replace('m12_', '')]
    out.append(row)

VLLM_V = ['m12_fused_vctx', 'm12_fused_l2cls_epi']
vman = args.vllm_capture_root / 'MANIFEST.csv'
vtags = []
if vman.exists():
    for r in csv.DictReader(open(vman)):
        t = r['tag']
        if r['status'].startswith('scored') and not t.endswith('_eager') and t not in vtags:
            vtags.append(t)
for t in vtags:
    if not (args.vllm_capture_root / 'cap' / f'{t}_kernels.csv').exists():
        continue
    model, phase, _ = re.match(r'^(.*)_(prefill|decode)_(\d+x\d+)$', t).groups()
    subprocess.run([sys.executable, str(HERE / 'join_vllm.py')] + join_args(args.vllm_capture_root, 'v1') + [t] + VLLM_V, stdout=subprocess.DEVNULL, check=True)
    md = (args.out_dir / f'vllm_{t}.md').read_text()
    cfg = json.load(open(args.config_dir / f'{model}.json'))
    base = {'capture_set': 'v1', 'measured_source': 'vllm', 'model': model, 'phase': phase, 'batch': 1, 'seq_len': 1024,
            'params': params(cfg), 'in_counts_set': model != 'SmolLM2-135M', 'grad_ckpt': False}
    if args.architecture == 'sm90':
        base['_hopper_summary'] = str(args.out_dir / f'vllm_{t}_hopper.json')
    empty = {'pred_eager_l2sim': '', 'pred_eager_l2cls': '', 'pred_eager_l2cls_epi': ''}
    for key, name, unit in (('HBM GB', 'hbm_bytes', 1e9), ('L2', 'l2_bytes', 1e9),
                            ('SMEM GB (ld+st+ldsm)', 'smem_bytes', 1e9), ('vector GFLOP', 'vector_flops', 1e9)):
        if args.architecture == 'sm90' and name == 'vector_flops':
            continue
        if name == 'l2_bytes':
            c, meas, ref = l2_cells(md)
            extra = l2_extra(meas, ref)
        else:
            c = cells(md, key); meas, extra = float(c[0]) * unit, {}
        out.append({**base, **extra, 'count': name, 'measured': meas, 'pred_fused': float(c[1]) * unit, **empty,
                    'pred_final': float(c[3]) * unit})
    m = re.search(r'tensor on GEMM \+ attention kernels only: measured ([\d.]+) GFLOP, (.*?)\. (?:Non|Hopper)', md)
    pv = dict(re.findall(r'(m12_\w+) ([\d.]+) \(ratio', m.group(2)))
    out.append({**base, 'count': 'tensor_flops', 'measured': float(m.group(1)) * 1e9,
                'pred_fused': float(pv['m12_fused_vctx']) * 1e9, **empty, 'pred_final': float(pv['m12_fused_l2cls_epi']) * 1e9})

def _v2_rows(capdir, joiner, variants, source, pred_cols):
    man = capdir / 'MANIFEST.csv'
    if not man.exists():
        return
    seen = []
    for r in csv.DictReader(open(man)):
        t = r['tag']
        if r['status'].startswith('scored') and not t.endswith('_eager') and t not in seen:
            seen.append(t)
    for t in seen:
        if not (capdir / 'cap' / f'{t}_kernels.csv').exists():
            continue
        model, phase, _, suffix = re.match(r'^(.*)_(prefill|decode|train)_(\d+x\d+)(_nockpt)?$', t).groups()
        subprocess.run([sys.executable, str(HERE / joiner)] + join_args(capdir, 'v2') + [t] + variants, stdout=subprocess.DEVNULL, check=True)
        md = (args.out_dir / (('vllm_' if source == 'vllm' else '') + f'{t}_v2.md')).read_text()
        cfg = json.load(open(args.config_dir / f'{model}.json'))
        base = {'capture_set': 'v2', 'measured_source': source, 'model': model,
                'phase': phase, 'batch': 1, 'seq_len': 1024, 'params': params(cfg),
                'in_counts_set': model != 'SmolLM2-135M', 'grad_ckpt': False}
        if args.architecture == 'sm90':
            prefix = 'vllm_' if source == 'vllm' else ''
            base['_hopper_summary'] = str(args.out_dir / f'{prefix}{t}_v2_hopper.json')
        for key, name, unit in (('HBM GB', 'hbm_bytes', 1e9), ('L2', 'l2_bytes', 1e9),
                                ('SMEM GB (ld+st+ldsm)', 'smem_bytes', 1e9), ('vector GFLOP', 'vector_flops', 1e9)):
            if args.architecture == 'sm90' and name == 'vector_flops':
                continue
            if name == 'l2_bytes':
                c, meas, ref = l2_cells(md)
                extra = l2_extra(meas, ref)
            else:
                c = cells(md, key); meas, extra = float(c[0]) * unit, {}
            row = {**base, **extra, 'count': name, 'measured': meas}
            for i, col in enumerate(pred_cols):
                row[col] = float(c[1 + 2 * i]) * unit if col else ''
            out.append(row)
        m = re.search(r'tensor on GEMM \+ attention kernels only: measured ([\d.]+) GFLOP, (.*?)\. (?:Non|Hopper)', md)
        pv = dict(re.findall(r'(m12_\w+) ([\d.]+) \(ratio', m.group(2)))
        row = {**base, 'count': 'tensor_flops', 'measured': float(m.group(1)) * 1e9}
        for v, col in zip(variants, pred_cols):
            row[col] = float(pv[v]) * 1e9
        out.append(row)

_v2_rows(args.vllm_capture_v2_root, 'join_vllm.py', ['m12_fused_vctx', 'm12_fused_l2cls_epi'], 'vllm',
         ['pred_fused', 'pred_final'])
_v2_rows(args.capture_v2_root, 'join.py', ['m12_fused', 'm12_eager_l2sim', 'm12_eager_l2cls', 'm12_eager_l2cls_epi'],
         'pytorch', ['pred_fused', 'pred_eager_l2sim', 'pred_eager_l2cls', 'pred_eager_l2cls_epi'])
for r in out:
    if r.get('capture_set') == 'v2' and r['measured_source'] == 'pytorch':
        r['pred_final'] = r['pred_eager_l2cls_epi']
if args.architecture == 'sm90':
    for row in out:
        summary = json.loads(Path(row.pop('_hopper_summary')).read_text())
        totals = summary['totals']
        row.update(architecture='sm90', metric_semantics=summary['metric_semantics'],
                   vector_fp16_weighted_ops=totals['vector_fp16_weighted_ops'],
                   vector_fp32_flops=totals['vector_fp32_flops'],
                   packed_half_normalization='UNCONFIRMED',
                   unassigned_kernel_count=len(summary['unassigned_kernels']))
        if row['count'] == 'vector_flops':
            row['measured'] = ''
        else:
            row['measured'] = totals[row['count']]
        if row['count'] == 'l2_bytes':
            row.update(l2_definition='delivered', measured_l2_lookup_bytes_ref=totals['l2_lookup_bytes'])
cols = ['capture_set', 'l2_definition']
for r in out:
    for k in r:
        if k not in cols:
            cols.append(k)
for r in out:
    for k in cols:
        r.setdefault(k, '')
with open(args.output, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(out)
print(len(out), 'rows ->', args.output)
