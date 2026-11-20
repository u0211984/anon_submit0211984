import copy, csv, json, sys
from pathlib import Path
TREE = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]).resolve()
W = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 16
ONLY = sys.argv[sys.argv.index('--only') + 1] if '--only' in sys.argv else None
sys.path.insert(0, str(TREE))
from paper_pipeline import driver as D
from paper_pipeline import counts_stage as C
C.CAPTURE_FORMATS.update(vllm_heldout='vllm_v2', pytorch_heldout='pytorch_v2', vllm_long='vllm_v2')
head = D.git('rev-parse', 'HEAD')
assert not [l for l in D.git('status', '--porcelain').splitlines() if l], 'tree not clean'
e, reg, settings, rep, inputs, res, variants = D._entry_runs('counts_fullstep')
assert [v for v, _ in variants] == ['default'], variants
vname, overlay = variants[0]
runs = []
if ONLY in (None, 'old'):
    for r in res['run_list']:
        if r.get('excluded') or r['model'] in ('SmolLM2-135M', 'Qwen3-30B-A3B'):
            continue
        kv = 1025 if r['phase'] == 'decode' and r['set'].startswith('vllm') else (1024 if r['phase'] == 'decode' else None)
        runs.append(dict(r, kv=kv, new_or_old='old'))
    for m, ph, s, kv in (('Llama-3.1-8B', 'prefill', 4096, None), ('Llama-3.1-8B', 'prefill', 8192, None),
                         ('Llama-3.1-8B', 'decode', 4095, 4096), ('Llama-3.1-8B', 'decode', 8191, 8192)):
        runs.append(dict(set='vllm_long', model=m, phase=ph, batch=1, seq_len=s, variant='', family='llama', kv=kv,
                         new_or_old='old'))
if ONLY in (None, 'new'):
    PLAN = TREE / sys.argv[sys.argv.index('--plan') + 1]
    for p in csv.DictReader(open(PLAN)):
        tr = p['phase'] == 'train'
        runs.append(dict(set='pytorch_heldout' if tr else 'vllm_heldout', model=p['model'], phase=p['phase'],
                         batch=int(p['batch']), seq_len=int(p['seq']), variant='nockpt' if tr else '',
                         family=p['family'], kv=int(p['seq']) + 1 if p['phase'] == 'decode' else None,
                         new_or_old='new', order=int(p['order'])))
groups = {}
for r in runs:
    groups.setdefault(r['kv'], []).append(r)
rc = 0
for kv, rs in sorted(groups.items(), key=lambda x: str(x[0])):
    ee = copy.deepcopy(e)
    if kv is not None:
        ee['workload']['rules']['decode_kv_tokens_on_entry'] = {'vllm': kv, 'pytorch': kv}
    sub = OUT / f'kv{kv}'
    clean = [{k: v for k, v in r.items() if k not in ('kv', 'new_or_old', 'order')} for r in rs]
    print(f'group kv={kv}: {len(rs)} runs -> {sub}', flush=True)
    rc |= C.run(ee, clean, settings, vname, overlay, sub, head, W)
    (sub / 'RUNS.json').write_text(json.dumps(rs, indent=1) + '\n')
(OUT / 'PREDICT_INFO.json').write_text(json.dumps(dict(commit=head, entry='counts_fullstep', variant=vname,
                                                       runs=len(runs), rc=rc), indent=1) + '\n')
print('done rc', rc, 'commit', head)
sys.exit(rc)
