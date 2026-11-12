"""Sensitivity of the L1 test loss to the number of material points m.

One figure; colour distinguishes the PRNN type, marker the training-set size.
Within a family every setting except mat_points is identical (verified against config.txt).
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

try:
    plt.style.use(['science', 'bright'])
    rc('text', usetex=True)
except:
    print("not using custom colors")

# type = 'mu'
# type = 'vfrac_ratio'
type = 'vary_all_v2'
folder = f'../trained_models/train_{type}/'
save_folder = f'{folder}matpoints/'

training_samples = [2, 4, 8, 12, 16, 24, 32, 64, 96, 128, 192, 256, 512]
samples_to_plot = [32, 512]  # one line per entry, marker per entry
sample_markers = ['o', 's', '^', 'D']
ylim = (0.01, 1.0)  # None  # set to e.g. (0, 0.1) to fix y-axis across plots, None for automatic
families = {  # pattern: dirname regex with one group capturing m
    'nonlin': {'label': 'PRNN (nonlinear)', 'color': 'C0', 'pattern': r'prnn_nonlin_3L8_(\d+)m'},
    'energy': {'label': 'PRNN (energy)', 'color': 'C1', 'pattern': r'prnn_energy_1L8_(\d+)m'},
}



def discover_configs(pattern):
    """Find {m: dirname} for dirs fully matching pattern."""
    configs = {}
    for d in sorted(os.listdir(folder)):
        match = re.fullmatch(pattern, d)
        if match and os.path.isdir(f"{folder}{d}"):
            configs[int(match.group(1))] = d
    if not configs:
        print(f"WARNING: no configs matching '{pattern}' in {folder}")
    else:
        print(f"'{pattern}' -> m = {sorted(configs)}")
    return configs


def read_config(name):
    """Parse config.txt into a dict."""
    cfg = {}
    with open(f"{folder}{name}/config.txt") as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                cfg[k.strip()] = v.strip()
    return cfg


def check_only_m_differs(configs):
    """Warn if configs in a family differ in anything other than mat_points."""
    if not configs:
        return
    ignore = {'name', 'mat_points'}
    ref_m, ref_name = sorted(configs)[0], configs[sorted(configs)[0]]
    ref = read_config(ref_name)
    for m, name in sorted(configs.items()):
        cfg = read_config(name)
        diff = {k for k in set(ref) | set(cfg) if k not in ignore and ref.get(k) != cfg.get(k)}
        if diff:
            print(f"  WARNING: {name} differs from {ref_name} in {sorted(diff)}")
        if cfg.get('mat_points') != str(m):
            print(f"  WARNING: {name} has mat_points={cfg.get('mat_points')}, expected {m}")


# load data
for fam in families.values():
    fam['configs'] = discover_configs(fam['pattern'])
    check_only_m_differs(fam['configs'])
    fam['losses'] = {}  # m -> [n_sample_counts, n_runs]
    for m, name in sorted(fam['configs'].items()):
        path = f"{folder}{name}/test_losses.txt"
        try:
            losses = np.loadtxt(path, delimiter=',')
            fam['losses'][m] = losses.reshape(-1, 1) if losses.ndim == 1 else losses
            print(f"Loaded {name}: {fam['losses'][m].shape}")
        except FileNotFoundError:
            print(f"WARNING: not found: {path}")

# plot
os.makedirs(save_folder, exist_ok=True)

for log_scale, log_x in [(True, False), (False, False), (True, True)]:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    plot_func = {(True, True): ax.loglog, (True, False): ax.semilogy,
                 (False, True): ax.semilogx, (False, False): ax.plot}[(log_scale, log_x)]

    all_ms = set()
    for fam in families.values():
        if not fam['losses']:
            continue
        ms = sorted(fam['losses'])
        all_ms.update(ms)
        for marker, n_train in zip(sample_markers, samples_to_plot):
            i = training_samples.index(n_train)
            vals = [np.mean(fam['losses'][m][i]) for m in ms]
            plot_func(ms, vals, linestyle='-', linewidth=1, marker=marker, markersize=4, color=fam['color'])

    # one legend, two blocks: colour = PRNN type, marker = training-set size
    color_handles = [plt.Line2D([], [], color=f['color'], linewidth=1, label=f['label']) for f in families.values() if
                     f['losses']]
    marker_handles = [plt.Line2D([], [], color='k', linestyle='none', marker=mk, markersize=4, label=f'{n} samples') for
                      mk, n in zip(sample_markers, samples_to_plot)]
    ax.legend(handles=color_handles + marker_handles, loc='upper right', fontsize=8, ncol=2)
    # ax.legend(handles=color_handles + marker_handles, loc=(1.05, 0.1), fontsize=7, ncol=2)

    ax.set_xlabel('Number of material points $m$')
    if log_scale:
        ax.set_ylabel('L$_1$ Test Loss', y=0.6, labelpad=0)
    else:
        ax.set_ylabel('L$_1$ Test Loss')
    ax.set_xticks(sorted(all_ms))
    ax.set_xticks([], minor=True)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    if not log_x:
        ax.set_xlim(0, max(all_ms) * 1.05)
    if ylim is not None:
        ax.set_ylim(ylim)
    elif not log_scale:
        ax.set_ylim(ymin=0)
    ax.grid(True, alpha=0.3)

    log_suffix = {(True, True): '_loglog', (True, False): '_log', (False, False): ''}[(log_scale, log_x)]
    out = f"{save_folder}matpoints_combined{log_suffix}.pdf"
    plt.savefig(out, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"Saved {out}")
