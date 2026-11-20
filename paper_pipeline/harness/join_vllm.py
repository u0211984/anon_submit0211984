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
model, phase, bs = re.match(r'^(.*)_(prefill|decode)_(\d+x\d+)(?:_eager)?$', TAG).groups()
b, s = bs.split('x')
WL = f'{model}_{phase}_b{b}_s{s}'
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
        r[k] = float(r.get(k) or 0.0)
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
    is_reduce = lambda r: r['label'] == 'gemm' and 'splitKreduce' in r['kernel']  # noqa: E731
    gemm_idx = [i for i, r in enumerate(rows) if r['label'] == 'gemm' and not is_reduce(r)]
    if len(gemm_idx) != 4 * L + 1:
        raise SystemExit(f'{TAG}: {len(gemm_idx)} GEMM kernels, expected 4 x {L} + 1; role split needs a new rule')
    gnames = ['qkv', 'o', 'mlp_gate_up', 'mlp_down']
    out = [None] * len(rows)
    seg = 'pre'
    g = 0
    for i, r in enumerate(rows):
        k = r['kernel']
        if r['label'] == 'gemm' and not is_reduce(r):
            if g == 4 * L:
                out[i] = ('gemm', 'lm_head'); seg = 'sampler'
            else:
                n = gnames[g % 4]
                out[i] = ('gemm', n)
                seg = {'qkv': 'attn', 'o': 'layernorm2', 'mlp_gate_up': 'swiglu',
                       'mlp_down': 'layernorm1' if g < 4 * L - 1 else 'final_norm'}[n]
            g += 1
        elif is_reduce(r):
            out[i] = out[i - 1] if i and out[i - 1] and out[i - 1][0] == 'gemm' else ('gemm', 'splitk_reduce')
        elif 'rms_norm' in k:
            out[i] = ('norm', 'layernorm1' if seg == 'pre' else seg)
            if seg == 'pre':
                seg = 'layernorm1'
        elif 'rotary_embedding' in k:
            out[i] = ('elementwise', 'rope')
        elif 'act_and_mul' in k:
            out[i] = ('elementwise', 'swiglu')
        elif r['label'] in ('attention', 'kv_cache') or 'FillFunctor' in k:
            out[i] = ('attention', 'attention')
        elif seg == 'pre':
            out[i] = ('embedding', 'embedding')
        elif seg == 'final_norm' and 'indexSelect' in k:
            out[i] = ('gemm', 'lm_head')
        elif seg == 'sampler':
            out[i] = ('elementwise', 'head_softmax')
        else:
            out[i] = ('other', f'other_{seg}')
    return out

def assign_roles_moe(rows):
    is_reduce = lambda r: r['label'] == 'gemm' and 'splitKreduce' in r['kernel']  # noqa: E731
    gemm_idx = [i for i, r in enumerate(rows) if r['label'] == 'gemm' and not is_reduce(r)]
    moe_idx = [i for i, r in enumerate(rows) if r['label'] == 'moe_gemm']
    if len(gemm_idx) != 3 * L + 1 or len(moe_idx) != 2 * L:
        raise SystemExit(f'{TAG}: {len(gemm_idx)} GEMM and {len(moe_idx)} fused_moe kernels, expected 3 x {L} + 1 and '
                         f'2 x {L}; role split needs a new rule')
    gnames = ['qkv', 'o', 'moe_router']
    out = [None] * len(rows)
    seg, g, mg = 'pre', 0, 0
    for i, r in enumerate(rows):
        k = r['kernel']
        if r['label'] == 'gemm' and not is_reduce(r):
            if g == 3 * L:
                out[i] = ('gemm', 'lm_head'); seg = 'sampler'
            else:
                n = gnames[g % 3]
                out[i] = ('gemm', n)
                seg = {'qkv': 'attn', 'o': 'layernorm2', 'moe_router': 'moe_routing'}[n]
            g += 1
        elif is_reduce(r):
            out[i] = out[i - 1] if i and out[i - 1] and out[i - 1][0] == 'gemm' else ('gemm', 'splitk_reduce')
        elif r['label'] == 'moe_gemm':
            out[i] = ('gemm', 'moe_w13' if mg % 2 == 0 else 'moe_w2')
            seg = 'moe_swiglu' if mg % 2 == 0 else 'moe_sum'
            mg += 1
        elif 'rms_norm' in k:
            if seg == 'moe_sum':
                seg = 'layernorm1' if mg < 2 * L else 'final_norm'
            out[i] = ('norm', 'layernorm1' if seg == 'pre' else seg)
            if seg == 'pre':
                seg = 'layernorm1'
        elif 'rotary_embedding' in k:
            out[i] = ('elementwise', 'rope')
        elif r['label'] in ('attention', 'kv_cache') or 'FillFunctor' in k:
            out[i] = ('attention', 'attention')
        elif seg == 'pre':
            out[i] = ('embedding', 'embedding')
        elif seg == 'final_norm' and 'indexSelect' in k:
            out[i] = ('gemm', 'lm_head')
        elif seg == 'sampler':
            out[i] = ('elementwise', 'head_softmax')
        elif seg in ('moe_routing', 'moe_swiglu', 'moe_sum'):
            align = r['label'] == 'moe_routing' and re.search(r'moe_align|count_and_sort', k)
            out[i] = ('elementwise', 'moe_align' if align else seg)
        else:
            out[i] = ('other', f'other_{seg}')
    return out

MOE = cfg.get('model_type') == 'qwen3_moe'
meas_role = defaultdict(lambda: defaultdict(float))
meas_cls = defaultdict(lambda: defaultdict(float))
roles = ([(r['label'], 'unassigned') for r in krows] if args.architecture == 'sm90'
         else (assign_roles_moe(krows) if MOE else assign_roles(krows)) if phase != 'train'
         else [(r['label'], '') for r in krows])
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
    write_summary(args.out_dir / f'vllm_{TAG}{SUF}_hopper.json', krows, hopper_raw)
    out = annotate_markdown(out)
    out[0] = out[0].replace('vector counter: half + single precision (FP32 included)',
        'scalar half weighted operations and FP32 FLOPs are separate in the Hopper JSON sidecar')
    out.append('\nHopper definitions: XBAR minus DSMEM; tensor ops_path sums. Metric semantics and packed-half normalization UNCONFIRMED. All kernel roles unassigned; exact totals and raw diagnostics in the Hopper JSON sidecar.\n')
(args.out_dir / f'vllm_{TAG}{SUF}.md').write_text(''.join(out))
with (args.out_dir / f'vllm_{TAG}{SUF}_roles.csv').open('w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['tag', 'class', 'role', 'kernels'] + [
        'vector_fp16_weighted_ops' if args.architecture == 'sm90' and k == 'vector_fp16_flops' else k for k in MEAS])
    for (c, ro), d in sorted(meas_role.items()):
        w.writerow([TAG, c, ro, int(d['n'])] + [d[k] for k in MEAS])
print(''.join(out))
