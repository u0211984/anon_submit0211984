import csv, subprocess, sys
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
NEW = TREE / 'paper_pipeline/out' / HEAD
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
STEMS = ['single_gpu_energy', 'multi_gpu_energy']

BLUE, ORANGE, PURPLE, GREEN = '#2878B5', '#E07A2D', '#8e44ad', '#3a8f5c'
INK, BAND = '#2b2e33', '#e8eef4'
LIM = (0.5, 1.4)
FIGSIZE = (3.4, 2.45)
DROP = {'c01', 'c05', 'c09', 'c13', 'c21', 'c25'}
PARAMS = {'Llama-3.2-1B': 1.24, 'Qwen2.5-1.5B': 1.54, 'Qwen3-1.7B': 1.72, 'Llama-3.2-3B': 3.21, 'Qwen2.5-7B': 7.62,
          'Llama-3.1-8B': 8.03, 'Llama-2-13B': 13.0, 'Llama-3.1-70B': 70.6}


def rows(path, q='energy_per_step'):
    return [r for r in csv.DictReader(open(path, encoding='utf-8-sig')) if r['quantity'] == q]


def paper(r): return 'collective_buffers=occupy_l2' in r['variant'] and 'per_payload_byte' in r['variant']
def ratio(r): return float(r['ratio'])
def mae(vals): return 100 * sum(abs(v - 1) for v in vals) / len(vals)


def load(q):
    bare = rows(NEW / 'bare_step96/scores/cells.csv', q)
    assert len(bare) == 107 and all(r['status'] == 'ok' and r['commit'] == HEAD for r in bare)
    assert q != 'energy_per_step' or all(r['gpu_work_basis'] == 'GPU-work basis' for r in bare)
    meg1 = rows(NEW / 'megatron_single_a100/scores/cells.csv', q)
    assert len(meg1) == 20 and all(r['status'] == 'ok' and r['commit'] == HEAD for r in meg1)
    mg = [r for r in rows(NEW / 'multi_gpu32_locked1065/scores/cells.csv', q) if paper(r) and r['ratio']]
    assert len(mg) == 32 and all(r['status'] == 'ok' and r['commit'] == HEAD for r in mg)
    mm = [r for r in rows(NEW / 'megatron_multi_a100/scores/cells.csv', q) if r['status'] == 'ok' and r['ratio'] and paper(r) and r['cell'] not in DROP]
    assert len(mm) == 27 and all(r['commit'] == HEAD for r in mm)
    for r in mm: r['group'] = r['description'].split(' train ')[0]
    return bare, meg1, mg, mm


bare, meg1, mg, mm = load('energy_per_step')
for v in [ratio(r) for r in bare + meg1 + mg + mm]:
    assert LIM[0] < v < LIM[1], v

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
            ax.scatter(xs, vals, s=13, color=c, alpha=.88, edgecolors='white', linewidths=.35, zorder=3)
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


num = ['# Validation figure numbers', '', 'MAE = mean |predicted/measured - 1| over the cells shown.', '']
inf = [r for r in bare if r['panel'] in ('prefill', 'decode')]
pre = [r for r in inf if r['panel'] == 'prefill']; dec = [r for r in inf if r['panel'] == 'decode']
hf = [r for r in bare if r['panel'] == 'train']
m_inf = sorted({r['group'] for r in inf}, key=PARAMS.get); m_tr = sorted({r['group'] for r in hf + meg1}, key=PARAMS.get)
assert len(m_inf) == 7 and len(m_tr) == 4
cols = [[([ratio(r) for r in pre if r['group'] == m], BLUE, 'o', -.2), ([ratio(r) for r in dec if r['group'] == m], ORANGE, 'o', .2)] for m in m_inf]
cols += [[([ratio(r) for r in hf if r['group'] == m], PURPLE, 'o', -.2), ([ratio(r) for r in meg1 if r['group'] == m], GREEN, 'o', .2)] for m in m_tr]
fig, ax = frame(cols, [(0, len(m_inf) - 1, 'vLLM inference'), (len(m_inf), len(cols) - 1, 'Training')], m_inf + m_tr, True)
legend(fig, [dot(BLUE), dot(ORANGE), Line2D([], [], ls='none'), dot(PURPLE), dot(GREEN)], ['prefill', 'decode', '|', 'PyTorch (HF)', 'Megatron'])
save(fig, STEMS[0])
num += ['## single_gpu_energy', '', '| section | series | n | MAE % |', '|---|---|---:|---:|',
        f'| vLLM inference | all | {len(inf)} | {mae([ratio(r) for r in inf]):.2f} |',
        f'| vLLM inference | prefill | {len(pre)} | {mae([ratio(r) for r in pre]):.2f} |',
        f'| vLLM inference | decode | {len(dec)} | {mae([ratio(r) for r in dec]):.2f} |',
        f'| Training | all | {len(hf + meg1)} | {mae([ratio(r) for r in hf + meg1]):.2f} |',
        f'| Training | PyTorch (HF) | {len(hf)} | {mae([ratio(r) for r in hf]):.2f} |',
        f'| Training | Megatron | {len(meg1)} | {mae([ratio(r) for r in meg1]):.2f} |', '']
mpre = [r for r in mg if r['panel'] == 'prefill']; mdec = [r for r in mg if r['panel'] == 'decode']
mi = sorted({r['group'] for r in mg}, key=PARAMS.get); mt = sorted({r['group'] for r in mm}, key=PARAMS.get)
assert len(mi) == 3 and len(mt) == 2
cols = [[([ratio(r) for r in mpre if r['group'] == m], BLUE, 'o', -.2), ([ratio(r) for r in mdec if r['group'] == m], ORANGE, 'o', .2)] for m in mi]
cols += [[([ratio(r) for r in mm if r['group'] == m], GREEN, 'o', 0.)] for m in mt]
fig, ax = frame(cols, [(0, len(mi) - 1, 'vLLM inference'), (len(mi), len(cols) - 1, 'Training')], mi + mt, True)
legend(fig, [dot(BLUE), dot(ORANGE), Line2D([], [], ls='none'), dot(GREEN)], ['prefill', 'decode', '|', 'Megatron'])
save(fig, STEMS[1])
num += ['## multi_gpu_energy', '', '| section | series | n | MAE % |', '|---|---|---:|---:|',
        f'| vLLM inference (TP2/4/8) | all | {len(mg)} | {mae([ratio(r) for r in mg]):.2f} |',
        f'| vLLM inference | prefill | {len(mpre)} | {mae([ratio(r) for r in mpre]):.2f} |',
        f'| vLLM inference | decode | {len(mdec)} | {mae([ratio(r) for r in mdec]):.2f} |',
        f'| Training (Megatron TP/PP/DP) | all | {len(mm)} | {mae([ratio(r) for r in mm]):.2f} |']
num += [f'| Training | {m} | {sum(r["group"] == m for r in mm)} | {mae([ratio(r) for r in mm if r["group"] == m]):.2f} |' for m in mt] + ['']


def groups(bare, meg1, mg, mm):
    inf = [r for r in bare if r['panel'] in ('prefill', 'decode')]
    tr = [r for r in bare if r['panel'] == 'train'] + meg1
    return {'single_vllm': inf, 'single_training': tr, 'multi_vllm': mg, 'multi_megatron': mm}


energy = {k: (len(v), mae([ratio(r) for r in v])) for k, v in groups(bare, meg1, mg, mm).items()}
step = {k: (len(v), mae([ratio(r) for r in v])) for k, v in groups(*load('time_per_step')).items()}
num += ['## step time (time_per_step, same cells)', '', '| group | n | MAE % |', '|---|---:|---:|']
num += [f'| {k} | {n} | {m:.2f} |' for k, (n, m) in step.items()] + ['']
(out / 'numbers.md').write_text('\n'.join(num))
print('\n'.join(num))
ok = check('energy', energy)
ok = check('step_time', step) and ok
sys.exit(0 if ok else 1)
