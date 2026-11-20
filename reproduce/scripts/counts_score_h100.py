import copy, csv, json, os, sys
from pathlib import Path
TREE = Path(__file__).resolve().parents[2]
R = Path(sys.argv[1]).resolve()
W = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 18
H = TREE / 'data/workload_set/counts_h100_heldout'
sys.path.insert(0, str(TREE))
from paper_pipeline import driver as D
from paper_pipeline import counts_stage as C
C.CAPTURE_FORMATS.update(vllm_heldout='vllm_v2')
head = D.git('rev-parse', 'HEAD')
e, reg, settings, rep, inputs, res, variants = D._entry_runs('counts_h100_fullstep')
VN = [v for v, _ in variants]
S = R / 'score_h100'; S.mkdir(parents=True, exist_ok=True)
OLD = R / 'pred/pred_h100_old'
old_runs = [r for r in res['run_list'] if not r.get('excluded') and not r.get('unsupported')]
if not (OLD / 'DONE').exists():
    for vname, overlay in variants:
        C.run(e, old_runs, settings, vname, overlay, OLD, head, W)
    (OLD / 'DONE').write_text(head + '\n')
runs = list(old_runs); meta = {C._cell(r): dict(new_or_old='old', kv=1025 if r['phase'] == 'decode' else None, order='') for r in old_runs}
for g in sorted((R / 'pred/pred_h100').glob('kv*')) + sorted((R / 'pred/pred_h100_add').glob('kv*')):
    for r in json.load(open(g / 'RUNS.json')):
        r = dict(r); kv = r.pop('kv'); runs.append(r); meta[C._cell(r)] = dict(new_or_old='new', kv=kv)
plan = {p['tag']: p for f in ('PLAN_H100.csv', 'PLAN_H100_ADD.csv') for p in csv.DictReader(open(H / f))}
for v in VN:
    rows = list(csv.DictReader(open(OLD / f'pred_{v}.csv')))
    for g in sorted((R / 'pred/pred_h100').glob('kv*')) + sorted((R / 'pred/pred_h100_add').glob('kv*')):
        rows += list(csv.DictReader(open(g / f'pred_{v}.csv')))
    with open(S / f'pred_{v}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
es = copy.deepcopy(e)
man = H / 'h100_vllm_heldout/MANIFEST.csv'
captured = {r['tag'] for r in csv.DictReader(open(man))} if man.exists() else set()
es['measured'].append(dict(set='vllm_heldout', manifest=str(man)))
for r in runs:
    if r['set'] == 'vllm_heldout' and C._tag(r) not in captured:
        r['missing'] = 'not captured'
C.score(es, runs, S, VN, head)
out = []
for r in csv.DictReader(open(S / 'scores/cells.csv')):
    if r['status'] not in ('ok', 'ok_unverified_counters'):
        continue
    run = next(x for x in runs if C._cell(x) == r['cell']); m = meta[r['cell']]
    out.append(dict(cell=r['cell'], model=run['model'], phase=run['phase'], seq_or_cached=m['kv'] if run['phase'] == 'decode' else run['seq_len'],
                    batch=run['batch'], quantity=r['quantity'], predicted=r['predicted'], measured=r['measured'], ratio=r['ratio'],
                    new_or_old=m['new_or_old'], capture_set=run['set'], variant=r['variant'], status=r['status'], commit=head,
                    plan_order=plan.get(C._tag(run), {}).get('order', '')))
if out:
    with open(R / 'COUNTS_H100.csv.tmp', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    os.replace(R / 'COUNTS_H100.csv.tmp', R / 'COUNTS_H100.csv')
print(len(out), 'rows;', len({x['cell'] for x in out if x['new_or_old'] == 'new'}), 'new cells,', len({x['cell'] for x in out if x['new_or_old'] == 'old'}), 'old cells')
