import csv, json, sys
from pathlib import Path
import numpy as np
TREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TREE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_pipeline.plots.style import plt, apply_style, COUNTS, COUNT_PHASE_COLORS, PHASE_MARKERS
from check import check
OUT = Path(sys.argv[1]).resolve()
PH = [('prefill', 'prefill'), ('decode', 'decode'), ('train', 'training')]
SHORT = ['HBM', 'L2', 'SMEM', 'FLOPs']
A = []
for r in csv.DictReader(open(OUT / 'COUNTS_A100.csv')):
    if r['superseded_by_v2'] == 'True' or not r['ratio'] or r['quantity'] not in COUNTS: continue
    moe = '30B-A3B' in r['model']
    if moe and (('v3_identical' in r['capture_set'] and int(r['batch']) > 1) or not ('vllm_default' in r['capture_set'] or 'vllm_default' in r['cell'])): continue
    A.append(dict(cell=r['cell'], model=r['model'], panel=r['phase'], quantity=r['quantity'], ratio=float(r['ratio']), moe=moe))
H = [dict(cell=r['cell'], panel=r['phase'], quantity=r['quantity'], ratio=float(r['ratio']), moe=False)
     for r in csv.DictReader(open(OUT / 'COUNTS_H100.csv'))
     if r['ratio'] and r['quantity'] in COUNTS and r['variant'] == 'h100_templates=S3']
for nm, D in (('A100', A), ('H100', H)):
    cells = {d['cell'] for d in D}; assert all(sum(1 for d in D if d['cell'] == c) == 4 for c in cells), nm
mape = lambda L: 100 * float(np.mean([abs(x - 1) for x in L]))
nums = {}
for nm, D in (('A100', A), ('H100', H)):
    o = dict(cells=len({d['cell'] for d in D}), per_phase={p: len({d['cell'] for d in D if d['panel'] == p}) for p, _ in PH},
             moe_cells=len({d['cell'] for d in D if d['moe']}), overall_mape=round(mape([d['ratio'] for d in D]), 2))
    for q, s in zip(COUNTS, SHORT):
        L = [d['ratio'] for d in D if d['quantity'] == q]; o[s] = dict(mape=round(mape(L), 2), min=round(min(L), 3), max=round(max(L), 3),
                                                                       by_phase={p: round(mape([d['ratio'] for d in D if d['quantity'] == q and d['panel'] == p]), 2) for p, _ in PH})
    nums[nm] = o
nums['A100']['models'] = sorted({d['model'] for d in A})
print(json.dumps(nums, indent=1))
apply_style('counts')
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(3.4, 2.0), dpi=300)
fig.subplots_adjust(left=.14, right=.985, bottom=.27, top=.80, wspace=.1)
hs = [plt.Line2D([], [], ls='', marker=PHASE_MARKERS[p], color=COUNT_PHASE_COLORS[p], markeredgewidth=0, markersize=3.1, label=l) for p, l in PH] + \
     [plt.Line2D([], [], ls='', marker='*', color='.3', markeredgewidth=0, markersize=4.6, label='MoE')]
fig.legend(handles=hs, loc='upper center', bbox_to_anchor=(.52, .99), ncol=4, fontsize=6.5, frameon=False, handlelength=.8, handletextpad=.35, columnspacing=1.4, borderaxespad=0)
YL = (.55, 1.25)
for gi, (nm, D, ax) in enumerate((('A100', A, axes[0]), ('H100', H, axes[1]))):
    ax.set(ylim=YL, yticks=[.6, .8, 1., 1.2]); ax.tick_params(labelsize=6.5, length=2, pad=2); ax.grid(axis='y', color='.9', lw=.4); ax.set_axisbelow(True)
    ax.axhline(1, color='.25', ls='--', lw=.6, zorder=2)
    ax.set(xlim=(-.5, 3.5), xticks=range(4), xticklabels=SHORT)
    ax.set_title(f'({chr(97 + gi)}) {nm}', loc='left', fontsize=6.5, pad=3)
    rng = np.random.default_rng(0)
    for pi, (p, _) in enumerate(PH):
        for i, q in enumerate(COUNTS):
            S = [d for d in D if d['panel'] == p and d['quantity'] == q]
            if not S: continue
            c = i + (pi - 1) * .25; xs = c + rng.uniform(-.055, .055, len(S))
            for x, d in zip(xs, S):
                if not (YL[0] < d['ratio'] < YL[1]):
                    yy = YL[1] - 0.02 if d['ratio'] > 1 else YL[0] + 0.02
                    ax.scatter(x, yy, marker='^' if d['ratio'] > 1 else 'v', s=12, color=COUNT_PHASE_COLORS[p], linewidths=0, zorder=6)
                    ax.text(x + 0.06, yy, f"{d['ratio']:.2f}", fontsize=5.2, va='center', ha='left', color=COUNT_PHASE_COLORS[p])
                    continue
                ax.scatter(x, d['ratio'], marker='*' if d['moe'] else PHASE_MARKERS[p], s=16 if d['moe'] else 7, color=COUNT_PHASE_COLORS[p], linewidths=0, alpha=.65 if not d['moe'] else .9, zorder=4 if d['moe'] else 3)
            ax.hlines(np.median([d['ratio'] for d in S]), c - .09, c + .09, color=COUNT_PHASE_COLORS[p], lw=.8, zorder=5)
axes[0].set_ylabel('Predicted / measured', fontsize=6.5, labelpad=3)
CROP_IN = 0.32
from matplotlib.transforms import Bbox
for ext in ('png', 'pdf'):
    w_, h_ = fig.get_size_inches(); fig.savefig(OUT / f'counts_A100_H100.{ext}', dpi=300, bbox_inches=Bbox([[0, CROP_IN], [w_, h_]]))
(OUT / 'counts_numbers.json').write_text(json.dumps(nums, indent=1))
ok = check('counts', {nm: (nums[nm]['cells'], 100 * float(np.mean([abs(d['ratio'] - 1) for d in D]))) for nm, D in (('A100', A), ('H100', H))})
sys.exit(0 if ok else 1)
