import csv, subprocess, sys
from pathlib import Path
TREE = Path(__file__).resolve().parents[2]
LABEL = 'vs RAW measured energy (no GPU-work basis available)'
V = 'collective_buffers=occupy_l2'


def convert(src):
    rows = list(csv.DictReader(src.open(newline='')))
    for r in rows:
        if r['predicted'] and r['measured_reference_secondary'] and r['status'] == 'no_gpu_work_basis':
            m, p = float(r['measured_reference_secondary']), float(r['predicted'])
            ratio = p / m
            r.update(measured=r['measured_reference_secondary'], ratio=repr(ratio), signed_err_pct=repr(100 * (ratio - 1)),
                     abs_err_pct=repr(abs(100 * (ratio - 1))), status='ok', label=LABEL)
    return rows


def occ(rows): return [r for r in rows if r['variant'] == V and r['quantity'] == 'energy_per_request']


head = subprocess.run(['git', '-C', str(TREE), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
src_csv = TREE / 'paper_pipeline/out' / head / 'energylens/scores/cells.csv'
dst = Path(sys.argv[1])
rows = occ(convert(src_csv))
src = {(r['cell'], r['variant'], r['quantity']): r for r in csv.DictReader(open(src_csv))}
for r in rows:
    s = src[(r['cell'], r['variant'], r['quantity'])]
    if 'rescaled' in s['label'] and 'rescaled' not in r['label']:
        r['label'] += '; ' + s['label'].split('; ', 1)[1]
with dst.open('w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f'{dst}: {len(rows)} rows, {sum(r["status"] == "ok" for r in rows)} ok')
