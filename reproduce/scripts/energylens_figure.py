import csv, re, subprocess, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check import check

TREE = Path(__file__).resolve().parents[2]
HEAD = subprocess.run(['git', '-C', str(TREE), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
out = Path(sys.argv[1])
ELENS = out / 'energylens_vsraw_cells.csv'
BLUE, ORANGE = '#2878B5', '#E07A2D'
INK, BAND = '#2b2e33', '#e8eef4'
LIM = (0.5, 1.4)
FIGSIZE = (3.4, 2.45)


def mae(vals): return 100 * sum(abs(v - 1) for v in vals) / len(vals)
def tj(r): return float(r['ratio'])
def elv(r): return float(r['reference_model_value']) / float(r['measured'])


el_all = [r for r in csv.DictReader(open(ELENS)) if r['status'] == 'ok']
assert len(el_all) == 91 and all('occupy_l2' in r['variant'] and r['reference_model_name'] == 'EnergyLens' and r['commit'] == HEAD for r in el_all)
GROUPS = [('Llama3-8B decode', 'Llama-3-8B', 'decode'), ('Llama3-70B prefill', 'Llama-3-70B', 'prefill'),
          ('Qwen3-30B-A3B prefill', 'Qwen3-30B-A3B', 'prefill')]
el = [r for r in el_all if r['group'] in {g for g, _, _ in GROUPS}]
dropped = [r for r in el_all if r not in el]
assert len(el) == 81, len(el)
for r in el:
    assert 'rescaled' not in r['label'] and not re.search(r'x\d', r['label']), r['cell']
    assert 'rescal' not in (r['measured_reference_secondary'] or '').lower(), r['cell']
for r in dropped:
    assert 'rescaled' in r['label'], r['cell']
JITTER = 0.12
RNG = np.random.default_rng(0)


def swarm(vals):
    return RNG.uniform(-JITTER, JITTER, len(vals))


def style():
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 6.5,
                         'ytick.labelsize': 7, 'axes.linewidth': .6, 'axes.edgecolor': '#666a70',
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def dot(c): return Line2D([], [], ls='none', marker='o', ms=4, mfc=c, mec='white', mew=.35)
def sq(c): return Line2D([], [], ls='none', marker='s', ms=3.8, mfc='none', mec=c, mew=.8)


def frame(cols, sections, xlabels, rotate):
    style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    n = len(cols)
    ax.set(xlim=(-.5, n - .5), ylim=LIM, yticks=np.round(np.arange(0.5, 1.401, 0.1), 1))
    ax.axhspan(0.95, 1.05, color=BAND, lw=0, zorder=0)
    ax.axhline(1.0, color=INK, lw=.6, zorder=1)
    for a, b, title in sections:
        if a > 0: ax.axvline(a - .5, color='#9aa0a6', lw=.7, ls=(0, (1, 1.6)), zorder=1)
        ax.text((a + b) / 2, 1.385, title, ha='center', va='top', fontsize=6.8, color=INK, fontweight='bold')
    for i, slots in enumerate(cols):
        for vals, c, marker, dx in slots:
            if not vals: continue
            xs = i + dx + swarm(vals)
            if marker == 'o':
                ax.scatter(xs, vals, s=13, color=c, alpha=.88, edgecolors='white', linewidths=.35, zorder=3)
            else:
                ax.scatter(xs, vals, s=11, marker='s', facecolors='none', edgecolors=c, linewidths=.75, alpha=.9, zorder=3)
    ax.set_ylabel('Predicted / measured energy', color=INK)
    ax.tick_params(axis='x', length=0, pad=2)
    if rotate:
        ax.set_xticks(range(n), xlabels, rotation=40, ha='right', rotation_mode='anchor', fontsize=6.2)
    else:
        ax.set_xticks(range(n), xlabels, fontsize=6.2)
    return fig, ax


def legend(fig, H, L):
    fig.legend(handles=H, labels=L, loc='lower center', bbox_to_anchor=(.56, .955), ncol=len(H), frameon=False,
               fontsize=6.3, handletextpad=.15, columnspacing=.8, handlelength=.8)


def save(fig, stem):
    for ext in ('.png', '.pdf'):
        fig.savefig(out / (stem + ext), format=ext[1:], dpi=300, bbox_inches='tight')
    plt.close(fig)


cols, labels = [], []
for g, name, ph in GROUPS:
    sub = [r for r in el if r['group'] == g]; c = BLUE if ph == 'prefill' else ORANGE
    cols.append([([tj(r) for r in sub], c, 'o', -.2), ([elv(r) for r in sub], c, 's', .2)])
    labels.append(f'{name}\n{ph}')
for v in [f(r) for r in el for f in (tj, elv)]:
    assert LIM[0] < v < LIM[1], v
fig, ax = frame(cols, [(0, 1, 'Dense'), (2, 2, 'MoE')], labels, True)
legend(fig, [dot('#555a60'), sq('#555a60'), Line2D([], [], ls='none'), dot(BLUE), dot(ORANGE)],
       ['TokenJoule', 'EnergyLens', '|', 'prefill', 'decode'])
save(fig, 'energylens')
dense = [r for r in el if not r['group'].startswith('Qwen3')]
num = ['## energylens (unrescaled EnergyLens references only)', '',
       'TokenJoule vs RAW measured energy, collective_buffers occupy_l2, host overheads 0.',
       'MAE = mean |predicted/measured - 1| (= MAPE). Not shown: ' + ', '.join(sorted({r['group'] for r in dropped})) + f' ({len(dropped)} rows, rescaled references).', '',
       '| group | n | TokenJoule MAE % | EnergyLens MAE % |', '|---|---:|---:|---:|']
num += [f'| {name} {ph} | {sum(r["group"] == g for r in el)} | {mae([tj(r) for r in el if r["group"] == g]):.2f} | {mae([elv(r) for r in el if r["group"] == g]):.2f} |' for g, name, ph in GROUPS]
num += [f'| dense | {len(dense)} | {mae([tj(r) for r in dense]):.2f} | {mae([elv(r) for r in dense]):.2f} |',
        f'| all | {len(el)} | {mae([tj(r) for r in el]):.2f} | {mae([elv(r) for r in el]):.2f} |', '']
(out / 'numbers_energylens.md').write_text('\n'.join(num))
print('\n'.join(num))
ok = check('energylens', {'tokenjoule': (len(el), mae([tj(r) for r in el])), 'energylens': (len(el), mae([elv(r) for r in el]))})
sys.exit(0 if ok else 1)
