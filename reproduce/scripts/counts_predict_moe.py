import json, sys
from pathlib import Path
TREE = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]).resolve()
W = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 10
sys.path.insert(0, str(TREE))
from paper_pipeline import driver as D
from paper_pipeline import counts_stage as C
head = D.git('rev-parse', 'HEAD')
assert not [l for l in D.git('status', '--porcelain').splitlines() if l], 'tree not clean'
e, reg, settings, rep, inputs, res, variants = D._entry_runs('counts_fullstep')
assert [v for v, _ in variants] == ['default']
runs = [r for r in res['run_list'] if r['model'] == 'Qwen3-30B-A3B']
assert len(runs) == 10, len(runs)
rc = C.run(e, runs, settings, variants[0][0], variants[0][1], OUT, head, W)
(OUT / 'RUNS.json').write_text(json.dumps([dict(r, kv=1025 if r['phase'] == 'decode' else None, new_or_old='new') for r in runs], indent=1) + '\n')
print('done rc', rc, head)
sys.exit(rc)
