import copy, csv, json, sys
from pathlib import Path
TREE = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]).resolve()
PLAN = TREE / sys.argv[sys.argv.index('--plan') + 1]
W = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 24
sys.path.insert(0, str(TREE))
from paper_pipeline import driver as D
from paper_pipeline import counts_stage as C
C.CAPTURE_FORMATS.update(vllm_heldout='vllm_v2')
head = D.git('rev-parse', 'HEAD')
assert not [l for l in D.git('status', '--porcelain').splitlines() if l], 'tree not clean'
e, reg, settings, rep, inputs, res, variants = D._entry_runs('counts_h100_fullstep')
plan = list(csv.DictReader(open(PLAN)))
groups = {}
for p in plan:
    r = dict(set='vllm_heldout', model=p['model'], phase=p['phase'], batch=int(p['batch']), seq_len=int(p['seq']), variant='', family=p['family'])
    groups.setdefault(int(p['seq']) + 1 if p['phase'] == 'decode' else None, []).append(r)
rc = 0
for vname, overlay in variants:
    for kv, rs in sorted(groups.items(), key=lambda x: str(x[0])):
        ee = copy.deepcopy(e)
        if kv is not None:
            ee['workload']['rules']['decode_kv_tokens_on_entry'] = {'vllm': kv, 'pytorch': kv}
        sub = OUT / f'kv{kv}'
        rc |= C.run(ee, rs, settings, vname, overlay, sub, head, W)
        (sub / 'RUNS.json').write_text(json.dumps([dict(r, kv=kv) for r in rs], indent=1) + '\n')
print('done rc', rc, head, [v for v, _ in variants])
sys.exit(rc)
