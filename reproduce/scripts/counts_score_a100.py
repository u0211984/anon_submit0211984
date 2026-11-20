import copy, csv, json, os, shutil, sys
from pathlib import Path
TREE = Path(__file__).resolve().parents[2]
R = Path(sys.argv[1]).resolve()
EG = TREE / 'data'
D = EG / 'workload_set/counts_heldout'
sys.path.insert(0, str(TREE))
from paper_pipeline import driver as DR
from paper_pipeline import counts_stage as C
C.CAPTURE_FORMATS.update(vllm_heldout='vllm_v2', pytorch_heldout='pytorch_v2', vllm_long='vllm_v2')
head = DR.git('rev-parse', 'HEAD')
NV = R / 'new_view'; NV.mkdir(parents=True, exist_ok=True)
if not (NV / 'cap').exists(): (NV / 'cap').symlink_to(D / 'cap')
done = {}
for l in open(D / 'done.txt'):
    t, rc = l.split()[0], l.split()[1]
    done[t] = rc
man = [dict(tag=t, status='scored') for t, rc in done.items() if rc == 'rc=0' and (D / 'cap' / f'{t}_kernels.csv').is_file()]
with open(NV / 'MANIFEST.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['tag', 'status']); w.writeheader(); w.writerows(man)
LONG = EG / 'workload_set/counts_fullstep_vllm_long'
V = R / 'old_long_view'; (V / 'cap').mkdir(parents=True, exist_ok=True)
ren = {'Llama-3.1-8B_prefill_1x4096': 'Llama-3.1-8B_prefill_1x4096', 'Llama-3.1-8B_prefill_1x8192': 'Llama-3.1-8B_prefill_1x8192',
       'Llama-3.1-8B_decode_kv4096': 'Llama-3.1-8B_decode_1x4095', 'Llama-3.1-8B_decode_kv8192': 'Llama-3.1-8B_decode_1x8191'}
for a, b in ren.items():
    for suf in ('.csv', '_kernels.csv'):
        p = V / 'cap' / f'{b}{suf}'
        if not p.exists():
            p.symlink_to(LONG / 'cap' / f'{a}{suf}')
with open(V / 'MANIFEST.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['tag', 'status']); w.writeheader(); w.writerows([dict(tag=b, status='scored') for b in ren.values()])
e, reg, settings, rep, inputs, res, variants = DR._entry_runs('counts_fullstep')
ee = copy.deepcopy(e)
ee['measured'] += [dict(set='vllm_heldout', manifest=str(NV / 'MANIFEST.csv')), dict(set='pytorch_heldout', manifest=str(NV / 'MANIFEST.csv')),
                   dict(set='vllm_long', manifest=str(V / 'MANIFEST.csv'))]
OUT = R / 'score_out'; OUT.mkdir(parents=True, exist_ok=True)
preds, runs, meta = [], [], {}
for g in sorted(list(R.glob('pred/pred_new/kv*')) + list(R.glob('pred/pred_supp1/kv*')) + list(R.glob('pred/pred_old/kv*'))):
    preds += list(csv.DictReader(open(g / 'pred_default.csv')))
    for r in json.load(open(g / 'RUNS.json')):
        r = dict(r); cell = C._cell(r)
        if cell in meta:
            continue
        meta[cell] = dict(new_or_old=r.pop('new_or_old'), kv=r.pop('kv'), order=r.pop('order', ''))
        runs.append(r)
with open(OUT / 'pred_default.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(preds[0])); w.writeheader(); w.writerows(preds)
C.score(ee, runs, OUT, ['default'], head)
rows = []
cap_new = {m['tag'] for m in man}
for r in csv.DictReader(open(OUT / 'scores/cells.csv')):
    cell = r['cell']; m = meta[cell]; run = next(x for x in runs if C._cell(x) == cell)
    if r['status'] != 'ok':
        continue
    rows.append(dict(cell=cell, model=run['model'], phase=run['phase'], seq_or_cached=(m['kv'] if run['phase'] == 'decode' else run['seq_len']),
                     batch=run['batch'], quantity=r['quantity'], predicted=r['predicted'], measured=r['measured'], ratio=r['ratio'],
                     new_or_old=m['new_or_old'], capture_set=run['set'], commit=head, plan_order=m['order']))
MOE = R / 'pred/pred_moe'
if (MOE / 'pred_default.csv').exists():
    QV = R / 'qwen_view'; (QV / 'cap').mkdir(parents=True, exist_ok=True)
    QOLD = EG / 'workload_set/counts_fullstep_vllm_qwen'
    qrows = [r for r in csv.DictReader(open(QOLD / 'MANIFEST.csv')) if r['status'].startswith('scored')]
    for r in qrows:
        for suf in ('.csv', '_kernels.csv'):
            q = QV / 'cap' / f"{r['tag']}{suf}"
            if not q.exists(): q.symlink_to(QOLD / 'cap' / f"{r['tag']}{suf}")
    mtags = [m['tag'] for m in man if m['tag'].startswith('Qwen3-30B-A3B_')]
    v4ok = {l.split()[0] for l in open(D / 'v4/done.txt') if l.split()[1] == 'rc=0'} if (D / 'v4/done.txt').exists() else set()
    msrc = {}
    for t in mtags:
        src = D / 'v4/cap' if t in v4ok and (D / 'v4/cap' / f'{t}_kernels.csv').is_file() else D / 'cap'
        msrc[t] = 'v4_distinct' if src.parent.name == 'v4' else 'v3_identical'
        for suf in ('.csv', '_kernels.csv'):
            q = QV / 'cap' / f'{t}{suf}'
            if q.is_symlink() or q.exists(): q.unlink()
            q.symlink_to(src / f'{t}{suf}')
    with open(QV / 'MANIFEST.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['tag', 'status']); w.writeheader()
        w.writerows([dict(tag=r['tag'], status='scored') for r in qrows] + [dict(tag=t, status='scored') for t in mtags])
    em = copy.deepcopy(e)
    for m in em['measured']:
        if m['set'] == 'vllm_qwen_v2':
            m['manifest'] = str(QV / 'MANIFEST.csv'); m.pop('scored_statuses', None)
    mruns = [dict(r) for r in json.load(open(MOE / 'RUNS.json'))]
    mmeta = {}
    for r in mruns:
        mmeta[C._cell(r)] = dict(kv=r.pop('kv'), new_or_old=r.pop('new_or_old'))
        if C._tag(r) in [x for x in mtags] or r.get('capture_tag') in mtags:
            r.pop('missing', None)
    MOUT = R / 'score_out_moe'; MOUT.mkdir(exist_ok=True)
    shutil.copyfile(MOE / 'pred_default.csv', MOUT / 'pred_default.csv')
    C.score(em, mruns, MOUT, ['default'], head)
    for r in csv.DictReader(open(MOUT / 'scores/cells.csv')):
        if r['status'] != 'ok': continue
        run = next(x for x in mruns if C._cell(x) == r['cell'])
        rows.append(dict(cell=r['cell'], model=run['model'], phase=run['phase'],
                         seq_or_cached=(1025 if run['phase'] == 'decode' else run['seq_len']), batch=run['batch'],
                         quantity=r['quantity'], predicted=r['predicted'], measured=r['measured'], ratio=r['ratio'],
                         new_or_old='new', capture_set='vllm_heldout_moe (' + (run.get('moe_tile_rule') or 'rapid_grouped') + '; ' + msrc.get(run.get('capture_tag') or C._tag(run), '?') + ')',
                         commit=head, plan_order=''))
key = lambda x: (x['model'], x['phase'], x['seq_or_cached'], x['batch'], x['quantity'])
v2 = {key(x) for x in rows if x['new_or_old'] == 'old' and not x['capture_set'].endswith('v1')}
for x in rows:
    x['superseded_by_v2'] = x['new_or_old'] == 'old' and x['capture_set'].endswith('v1') and key(x) in v2
with open(R / 'COUNTS_A100.csv.tmp', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
os.replace(R / 'COUNTS_A100.csv.tmp', R / 'COUNTS_A100.csv')
print(len(rows), 'rows;', len({x['cell'] for x in rows if x['new_or_old'] == 'new'}), 'new cells,',
      len({x['cell'] for x in rows if x['new_or_old'] == 'old' and not x['superseded_by_v2']}), 'old cells (after v2 rule)')
