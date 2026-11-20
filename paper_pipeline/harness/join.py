import csv, json, re, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _environment import reject_model_environment
reject_model_environment()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--capture-dir', type=Path, required=True)
parser.add_argument('--config-dir', type=Path, required=True)
parser.add_argument('--prediction-dir', type=Path, required=True)
parser.add_argument('--out-dir', type=Path, required=True)
parser.add_argument('--capture-set', choices=('v1', 'v2'), default='v1')
parser.add_argument('--architecture', choices=('sm80', 'sm90'), default='sm80')
parser.add_argument('tag')
parser.add_argument('variants', nargs='*')
args = parser.parse_args()

CAPSET, CAP = args.capture_set, args.capture_dir
SUF = '' if CAPSET == 'v1' else '_v2'
TAG, VARS = args.tag, args.variants or ['v2_default', 'v2_hbm4']
model, phase, bs, suffix = re.match(r'^(.*)_(prefill|decode|train)_(\d+x\d+)(_nockpt)?$', TAG).groups()
b, s = bs.split('x')

WL = f'{model}_{phase}_b{b}_s{s}' + (suffix if suffix and model == 'Llama-3.2-3B' else '')
cfg = json.load(open(args.config_dir / f'{model}.json'))
L = cfg['num_hidden_layers']

def _l2_columns(r):
    if 'l2_lookup_bytes' in r and r['l2_lookup_bytes'] not in ('', None):
        r['l2_lookup_bytes'] = float(r['l2_lookup_bytes']); r['l2_bytes'] = float(r['l2_bytes'])
        return
    r['l2_lookup_bytes'] = float(r.pop('l2_lts_bytes'))
    if 'xbar2l1_read_bytes' in r:
        r['l2_bytes'] = float(r['xbar2l1_read_bytes']) + float(r['l12xbar_write_bytes'])
    else:
        r['l2_bytes'] = float(r.get('l2_xbar_bytes') or 0.0)

MEAS = ('dram_bytes', 'l2_lookup_bytes', 'l2_bytes', 'smem_model', 'smem_old', 'smem_with_ldsm', 'smem_ld', 'smem_st', 'smem_ldsm', 'smem_ldgsts', 'tensor_flops_old', 'vector_fp16_flops')
if args.architecture == 'sm90':
    MEAS += ('vector_fp32_flops',)
krows = list(csv.DictReader(open(CAP / f'{TAG}_kernels.csv')))
if args.architecture == 'sm90':
    from hopper_join import overlay_raw, write_summary, annotate_markdown
    hopper_raw = overlay_raw(krows, CAP / f'{TAG}.csv')
for r in krows:
    for k in ('dram_bytes', 'smem_ld', 'smem_st', 'smem_ldgsts', 'smem_ldsm', 'tensor_flops_old', 'vector_fp16_flops'):
        r[k] = float(r[k])
    _l2_columns(r)
    r['smem_old'] = r['smem_ld'] + r['smem_st'] + r['smem_ldgsts']
HAS_DELIV = sum(r['l2_bytes'] for r in krows) > 0
if args.architecture == 'sm90':
    HAS_DELIV = True
L2LAB = 'L2 GB (delivered)' if HAS_DELIV else 'L2 lookups GB (reference only, not L2 traffic; no delivered counter in this capture)'

_raw = [l for l in open(CAP / f'{TAG}.csv') if l.startswith('"')]
_fp32 = defaultdict(float)
_has_fp32 = False
if _raw:
    _rows = list(csv.reader(_raw)); _h = _rows[0]
    _iN, _iV, _iI = _h.index('Metric Name'), _h.index('Metric Value'), _h.index('ID')
    _w = {'sm__sass_thread_inst_executed_op_fadd_pred_on.sum': 1, 'sm__sass_thread_inst_executed_op_fmul_pred_on.sum': 1,
          'sm__sass_thread_inst_executed_op_ffma_pred_on.sum': 2}
    for _r in _rows[1:]:
        if _r[_iN] in _w:
            _has_fp32 = True
            _fp32[int(_r[_iI])] += _w[_r[_iN]] * float(_r[_iV].replace(',', ''))
for r in krows:
    if args.architecture == 'sm80':
        r['vector_fp16_flops'] += _fp32.get(int(r['id']), 0.0)
    r['smem_with_ldsm'] = r['smem_old'] + r['smem_ldsm']

    r['smem_model'] = r['smem_ld'] + r['smem_st'] + r['smem_ldsm']
    if args.architecture == 'sm90':
        r['smem_model'] += r['smem_gmma'] + r['smem_stsm']

def assign_roles(rows):
    first_attn = next((i for i, r in enumerate(rows) if r['label'] == 'attention'), len(rows))
    rotary = {i for i, r in enumerate(rows[:first_attn]) if r['label'] == 'gemm'
              and ('sgemm' in r['kernel'] or ('gemmk1_kernel' in r['kernel'] and ', float,' in r['kernel']))}
    gemm_ids = {i for i, r in enumerate(rows) if r['label'] == 'gemm' and i not in rotary
                and 'splitKreduce' not in r['kernel']}
    is_gemm = lambda r: id(r) in gemm_obj  # noqa: E731
    gemm_obj = {id(rows[i]) for i in gemm_ids}
    is_reduce = lambda r: r['label'] == 'gemm' and 'splitKreduce' in r['kernel']  # noqa: E731
    gemm_idx = [i for i, r in enumerate(rows) if is_gemm(r)]
    if len(gemm_idx) != 7 * L + 1:
        raise SystemExit(f'{TAG}: {len(gemm_idx)} GEMM kernels, expected 7 x {L} + 1; role split needs a new rule')
    gnames = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']
    out = [None] * len(rows)
    seg = 'layernorm1'
    g = 0
    for i, r in enumerate(rows):
        if is_gemm(r):
            if g == 7 * L:
                out[i] = ('gemm', 'lm_head'); seg = 'head_softmax'
            else:
                n = gnames[g % 7]
                out[i] = ('gemm', {'q': 'qkv', 'k': 'qkv', 'v': 'qkv', 'o': 'o', 'gate': 'mlp_gate_up', 'up': 'mlp_gate_up',
                                   'down': 'mlp_down'}[n])
                seg = {'q': 'qk', 'k': 'qk', 'v': 'rope', 'o': 'layernorm2', 'gate': 'gu', 'up': 'swiglu',
                       'down': 'layernorm1' if g < 7 * L - 1 else 'final_norm'}[n]
            g += 1
            continue
        if is_reduce(r):
            out[i] = out[i - 1] if i and out[i - 1] and out[i - 1][0] == 'gemm' else ('gemm', 'splitk_reduce'); continue
        if r['label'] == 'attention':
            out[i] = ('attention', 'attention'); seg = 'post_attn'; continue
        if r['label'] == 'embedding':
            out[i] = ('embedding', 'embedding'); continue
        if i in rotary:
            out[i] = ('elementwise', 'rope_table'); continue
        if seg == 'gu':
            seg = 'swiglu_gu'
        cls = {'layernorm1': 'norm', 'layernorm2': 'norm', 'final_norm': 'norm', 'rope': 'elementwise',
               'swiglu': 'elementwise', 'swiglu_gu': 'elementwise', 'head_softmax': 'elementwise'}.get(seg, 'other')
        seg_role = 'swiglu' if seg == 'swiglu_gu' else seg
        out[i] = (cls, seg_role if cls != 'other' else f'other_{seg}')
        if seg == 'swiglu_gu':
            seg = 'gu'
        continue

    for i, r in enumerate(rows):
        if r['label'] == 'attention' and (i == 0 or rows[i - 1]['label'] != 'attention'):
            j = i - 1
            while j >= 0 and 'CatArrayBatchedCopy' in rows[j]['kernel'] and out[j] == ('elementwise', 'rope'):
                out[j] = ('attention', 'kv_write'); j -= 1
    return out

meas_role = defaultdict(lambda: defaultdict(float))
meas_cls = defaultdict(lambda: defaultdict(float))
roles = ([(r['label'], 'unassigned') for r in krows] if args.architecture == 'sm90'
         else assign_roles(krows) if phase != 'train' else [(r['label'], '') for r in krows])
for r, (c, ro) in zip(krows, roles):
    for k in MEAS:
        meas_role[(c, ro)][k] += r[k]; meas_cls[c][k] += r[k]; meas_cls['total'][k] += r[k]
    meas_role[(c, ro)]['n'] += 1

pred = {}
for V in VARS:
    p = defaultdict(lambda: defaultdict(float))
    for row in csv.DictReader(open(args.prediction_dir / f'pred_{V}.csv')):
        if row['workload'] == WL:
            p[(row['op_class'], row['role'])][row['counter']] += float(row['value'])
    pred[V] = p

def pv(p, key):
    d = p.get(key, {})
    g = lambda k: d.get(k, 0.0)  # noqa: E731
    return {'hbm': g('hbm_read') + g('hbm_write'), 'l2': g('l2'), 'smem': g('smem'),
            'tensor': g('tensor_useful') + g('tensor_padding'), 'tensor_useful': g('tensor_useful'), 'vector': g('vector')}

def mv(d):
    l2m, l2o = ('l2_bytes' if HAS_DELIV else 'l2_lookup_bytes'), 'l2_lookup_bytes'
    return {'hbm': d['dram_bytes'], 'l2': d[l2m], 'l2x': d[l2o], 'smem': d['smem_model'],
            'smem2': d['smem_old'], 'tensor': d['tensor_flops_old'], 'vector': d['vector_fp16_flops']}

def ratio(a, b):
    return f'{a / b:.2f}' if b > 0 else ('—' if a == 0 else 'inf')

out = [f'# {TAG}: predicted vs measured counts\n\nvector counter: half + single precision ({"FP32 included" if _has_fp32 else "half pipe only: this capture has no FP32 counters"})\n\nmeasured: {CAP / (TAG + "_kernels.csv")} ({len(krows)} kernels)\n'
       f'predicted: {", ".join(f"pred_{V}.csv" for V in VARS)} (workload {WL})\n\n']
V0 = VARS[0]
out.append('## Totals\n\n| counter | measured | ' + ' | '.join(f'{V} | ratio' for V in VARS) + ' |\n|---|---:|' + '---:|---:|' * len(VARS) + '\n')
m = mv(meas_cls['total'])
unit = {'hbm': 1e9, 'l2': 1e9, 'smem': 1e9, 'tensor': 1e9, 'vector': 1e9}
for k, lab in (('hbm', 'HBM GB'), ('l2', L2LAB), ('smem', 'SMEM GB (ld+st+ldsm)'), ('tensor', 'tensor GFLOP (issued)'),
               ('vector', 'vector GFLOP')):
    if args.architecture == 'sm90' and k == 'vector':
        continue
    cells = []
    for V in VARS:
        p = pv(pred[V], ('total', ''))
        cells.append(f'{p[k] / unit[k]:.3f} | {ratio(p[k], m[k])}')
    out.append(f'| {lab} | {m[k] / unit[k]:.3f} | ' + ' | '.join(cells) + ' |\n')
mga = sum(mv(meas_cls[c])['tensor'] for c in ('gemm', 'attention') if c in meas_cls)
out.append(f'\ntensor on GEMM + attention kernels only: measured {mga / 1e9:.3f} GFLOP, ' + ', '.join(
    f'{V} {sum(pv(pred[V], (c, ""))["tensor"] for c in ("gemm", "attention")) / 1e9:.3f} (ratio '
    f'{ratio(sum(pv(pred[V], (c, ""))["tensor"] for c in ("gemm", "attention")), mga)})' for V in VARS)
    + '. Non-GEMM kernels show about one HMMA-pipe instruction per warp (counter definition, bin d).\n')
_t = meas_cls['total']
out.append(f'\nSMEM measured reads (ld + ldsm) {(_t["smem_ld"] + _t["smem_ldsm"]) / 1e9:.3f} GB, writes (st) {_t["smem_st"] / 1e9:.3f} GB, '
           f'staging (ldgsts, not in the default model) {_t["smem_ldgsts"] / 1e9:.3f} GB.\n')
out.append(f'\nbeside: L2 lookups (reference only, not L2 traffic) {m["l2x"] / 1e9:.3f} GB; SMEM old definition ld+st+ldgsts {m["smem2"] / 1e9:.3f} GB; '
           f'tensor useful ({V0}) {pv(pred[V0], ("total", ""))["tensor_useful"] / 1e9:.3f} GFLOP\n\n')

for k, lab in (('hbm', 'HBM (GB)'), ('l2', 'L2 delivered (GB)' if HAS_DELIV else 'L2 lookups (GB; reference only, not L2 traffic)'), ('smem', 'SMEM ld+st+ldsm (GB)'), ('tensor', 'tensor issued (GFLOP)'),
               ('vector', 'vector (GFLOP)')):
    if args.architecture == 'sm90' and k == 'vector':
        continue
    out.append(f'## {lab} by class / role\n\n| class | role | kernels | measured | ' + ' | '.join(f'{V} | ratio' for V in VARS)
               + ' |\n|---|---|---:|---:|' + '---:|---:|' * len(VARS) + '\n')
    keys = sorted(set(meas_role) | {kk for V in VARS for kk in pred[V] if kk[1]}, key=lambda x: (x[0], x[1]))
    for c in ('gemm', 'attention', 'elementwise', 'norm', 'embedding', 'optimizer', 'other'):
        mc = mv(meas_cls[c]) if c in meas_cls else {kk: 0.0 for kk in ('hbm', 'l2', 'l2x', 'smem', 'smem2', 'tensor', 'vector')}
        cells = [f'{pv(pred[V], (c, ""))[k] / 1e9:.3f} | {ratio(pv(pred[V], (c, ""))[k], mc[k])}' for V in VARS]
        out.append(f'| **{c}** | | | **{mc[k] / 1e9:.3f}** | ' + ' | '.join(cells) + ' |\n')
        for kk in keys:
            if kk[0] != c or not kk[1]:
                continue
            md = meas_role.get(kk)
            mm = mv(md)[k] if md else 0.0
            cells = [f'{pv(pred[V], kk)[k] / 1e9:.3f} | {ratio(pv(pred[V], kk)[k], mm)}' for V in VARS]
            out.append(f'| | {kk[1]} | {int(md["n"]) if md else 0} | {mm / 1e9:.3f} | ' + ' | '.join(cells) + ' |\n')
    out.append('\n')

args.out_dir.mkdir(parents=True, exist_ok=True)
if args.architecture == 'sm90':
    write_summary(args.out_dir / f'{TAG}{SUF}_hopper.json', krows, hopper_raw)
    out = annotate_markdown(out)
    out[0] = out[0].replace('vector counter: half + single precision (FP32 included)',
        'scalar half weighted operations and FP32 FLOPs are separate in the Hopper JSON sidecar')
    out.append('\nHopper definitions: XBAR minus DSMEM; tensor ops_path sums. Metric semantics and packed-half normalization UNCONFIRMED. All kernel roles unassigned; exact totals and raw diagnostics in the Hopper JSON sidecar.\n')
(args.out_dir / f'{TAG}{SUF}.md').write_text(''.join(out))
with (args.out_dir / f'{TAG}{SUF}_roles.csv').open('w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['tag', 'class', 'role', 'kernels'] + [
        'vector_fp16_weighted_ops' if args.architecture == 'sm90' and k == 'vector_fp16_flops' else k for k in MEAS])
    for (c, ro), d in sorted(meas_role.items()):
        w.writerow([TAG, c, ro, int(d['n'])] + [d[k] for k in MEAS])
print(''.join(out))
