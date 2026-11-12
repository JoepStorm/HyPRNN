"""Plot learning curves from saved test_losses.txt files across categories."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

try:
    plt.style.use(['science', 'bright'])
    rc('text', usetex=True)
except:
    print("not using custom colors")

type = 'vary_all_v2'  # 'vfrac_ratio' 'vary_all_v2' 'mu'
highlight_keyword = None  # e.g. '_6m3' or ['_6m3', '_12m6']: configs containing any of these strings are highlighted (None to disable)
highlight_alpha = 1.0
non_highlight_linestyle = '-'

output_name = f"{type}_comparison"

save_folder = f'../trained_models/train_{type}/'
folder = f'../trained_models/train_{type}/'
training_samples = [2, 4, 8, 12, 16, 24, 32, 64, 96, 128, 192, 256, 512]
xticks = [2, 32, 64, 128, 192, 256, 512]
xlim_xticks = [2, 8, 16, 24, 32, 64, 96, 128]
xlog_xticks = [2, 4, 8, 16, 32, 64, 128, 256, 512]
ylim = (1/48, 1.4)  # None  # set to e.g. (0, 0.1) to fix y-axis across plots, None for automatic
# ylim = (0.01, 1.4)  # None  # set to e.g. (0, 0.1) to fix y-axis across plots, None for automatic
# ylim = (0.04, 250)


energy_types = [
    'prnn_energy_1L8_12m',
    'prnn_energy_2L8_12m',
    'prnn_energy_1L4_12m',
    'prnn_energy_1L12_12m',
    'prnn_energy_1L24_12m',
    'prnn_energy_1L8_12m_relu',
    'prnn_energy_1L8_12m_tanh',
    'prnn_energy_1L8_6m',
    'prnn_energy_1L8_12m6',
    'prnn_energy_1L8_18m',
    'prnn_energy_1L8_24m',
    'prnn_energy_1L8_12m_normdec',
    'prnn_energy_1L8_24m_normdec',
]

# In the mu runs the hypernetwork has no inputs (constant embedding), so these give results identical to 1L8_12m
mu_duplicates = [
    'prnn_energy_2L8_12m',
    'prnn_energy_1L8_12m_relu',
    'prnn_energy_1L8_12m_tanh',
]
energy_types_mu = [n for n in energy_types if n not in mu_duplicates]


model_types_mu = {
    'nn': {
        'configs': [
            'nn_8_2',
            'nn_16_2',
            'nn_16_3',
            'nn_32_3',
            'nn_64_2',
            'nn_64_3',
            'nn_64_4',
            'nn_128_3',
            'nn_relu_64_3',
            'nn_tanh_64_3',
        ],
        'color': 'C2',
        'linestyle': '--',
        'label': 'NN',
    },
    'prnn_nonlin': {
        'configs': [
            'prnn_nonlin_1L8_3m_sigmoid',
            'prnn_nonlin_1L8_6m_sigmoid',
            # 'prnn_nonlin_1L8_6m_sigmoid_scaleMU',
            'prnn_nonlin_1L8_12m_sigmoid',
            'prnn_nonlin_2m',
            'prnn_nonlin_3L16_6m_sigmoid',
            'prnn_nonlin_3m',
            'prnn_nonlin_3m_sigmoid',
            'prnn_nonlin_6m',
            'prnn_nonlin_6m3',
            'prnn_nonlin_6m_relu',
            'prnn_nonlin_6m_sigmoid',
            'prnn_nonlin_12m',
            'prnn_nonlin_12m_sigmoid',
            'prnn_nonlin_24m',
        ],
        'color': 'C0',
        'linestyle': ':',
        'label': 'PRNN (nonlinear)',
    },
    'prnn_energy': {
        'configs': energy_types_mu,
        'color': 'C1',
        'linestyle': '-.',
        'label': 'PRNN (energy)',
    },
}

model_types_vfrac_ratio = {
    'nn': {
        'configs': [
            'nn_8_2',
            'nn_16_2',
            'nn_32_3',
            'nn_64_2',
            'nn_64_3',
            'nn_64_4',
            'nn_128_3',
            'nn_relu_64_3',
            'nn_tanh_64_3',
        ],
        'color': 'C2',
        'linestyle': '--',
        'label': 'NN',
    },
    'prnn_nonlin': {
        'configs': [
            'prnn_nonlin_1L8_3m_sigmoid',
            'prnn_nonlin_1L8_6m',
            'prnn_nonlin_1L8_6m_sigmoid',
            'prnn_nonlin_1L8_12m_sigmoid',
            'prnn_nonlin_3L8_2m',
            'prnn_nonlin_3L8_3m',
            'prnn_nonlin_3L8_3m_sigmoid',
            'prnn_nonlin_3L8_6m',
            'prnn_nonlin_3L8_6m3',
            'prnn_nonlin_3L8_6m_relu',
            'prnn_nonlin_3L8_6m_sigmoid',
            'prnn_nonlin_3L8_12m',
            'prnn_nonlin_3L8_12m_sigmoid',
            'prnn_nonlin_3L8_24m',
            'prnn_nonlin_3L8relu_6m',
            'prnn_nonlin_3L16_6m_sigmoid',
        ],
        'color': 'C0',
        'linestyle': ':',
        'label': 'PRNN (nonlinear)',
    },
    'prnn_energy': {
        'configs': energy_types,
        'color': 'C1',
        'linestyle': '-.',
        'label': 'PRNN (energy)',
    },
}

model_types_all = {
    'nn': {
        'configs': [
            'nn_8_2',
            'nn_16_2',
            'nn_32_3',
            'nn_64_2',
            'nn_64_3',
            'nn_64_4',
            'nn_128_3',
            'nn_relu_64_3',
            'nn_tanh_64_3',
        ],
        'color': 'C2',
        'linestyle': '--',
        'label': 'NN',
    },
    'prnn_nonlin': {
        'configs': [
            'prnn_nonlin_1L8_3m_sigmoid',
            'prnn_nonlin_1HL8_1L_6m',
            'prnn_nonlin_1HL8_1L_6m_sigmoid',
            'prnn_nonlin_1HL8_3L_6m',
            'prnn_nonlin_1L8_6m_sigmoid',
            # 'prnn_nonlin_1L8_6m_sigmoid_scaleMu',
            'prnn_nonlin_1L8_12m_sigmoid',
            'prnn_nonlin_3L8_2m',
            'prnn_nonlin_3L8_3m',
            'prnn_nonlin_3L8_3m_sigmoid',
            'prnn_nonlin_3L8_6m',
            'prnn_nonlin_3L8_6m3',
            'prnn_nonlin_3L8_6m_relu',
            'prnn_nonlin_3L8_6m_sigmoid',
            'prnn_nonlin_3L8_12m',
            'prnn_nonlin_3L8_12m_sigmoid',
            'prnn_nonlin_3L8_24m',
            'prnn_nonlin_3L8relu_6m',
        ],
        'color': 'C0',
        'linestyle': ':',
        'label': 'PRNN (nonlinear)',
    },
    'prnn_energy': {
        'configs': energy_types,
        'color': 'C1',
        'linestyle': '-.',
        'label': 'PRNN (energy)',
    },
}

if type == 'mu':
    model_types = model_types_mu
elif type == 'vfrac_ratio':
    model_types = model_types_vfrac_ratio
elif type == 'vary_all_v2':
    model_types = model_types_all

# load data

for mtype in model_types.values():
    mtype['curves'] = []  # list of (name, samples, losses)
    mtype['curves_rel'] = []  # list of (name, samples, losses) for relative losses
    # mtype['curves_l2'] = []  # list of (name, samples, losses) for L2 losses
    for name in mtype['configs']:
        path = f"{folder}{name}/test_losses.txt"
        path_rel = f"{folder}{name}/test_losses_rel.txt"
        # path_l2 = f"{folder}{name}/test_losses_L2.txt"
        try:
            losses = np.loadtxt(path, delimiter=',')
            if losses.ndim == 1:
                losses = losses.reshape(-1, 1)
            mtype['curves'].append((name, training_samples, losses))
            print(f"Loaded {name}: {losses.shape}")
        except FileNotFoundError:
            print(f"WARNING: not found: {path}")
        try:
            losses_rel = np.loadtxt(path_rel, delimiter=',')
            if losses_rel.ndim == 1:
                losses_rel = losses_rel.reshape(-1, 1)
            mtype['curves_rel'].append((name, training_samples, losses_rel))
        except FileNotFoundError:
            pass  # relative losses are optional


# plot

if not any(mtype['curves'] for mtype in model_types.values()):
    print("No data loaded, nothing to plot.")
    exit()

has_rel = any(mtype['curves_rel'] for mtype in model_types.values())
# has_l2 = any(mtype['curves_l2'] for mtype in model_types.values())

datasets_to_plot = [
    ('curves', 'L$_1$ Test Loss', '', ylim),
]
if has_rel:
    datasets_to_plot.append(('curves_rel', 'Relative Test Loss', '_rel', None))
# if has_l2:
#     datasets_to_plot.append(('curves_l2', 'L2 Loss', '_l2', None))

for curves_key, ylabel_base, name_suffix, cur_ylim in datasets_to_plot:
  for agg_func, ylabel, suffix in [
      (lambda x: np.mean(x, axis=1), ylabel_base, name_suffix),
  ]:
    for log_scale, log_x in [(True, False), (False, False), (False, True), (True, True)]:
        fig, ax = plt.subplots(figsize=(4, 2.5))

        all_min_ys = []
        for mtype in model_types.values():
            color = mtype['color']
            linestyle = mtype['linestyle']

            # Collect all values per sample count for the envelope/min scatter
            sample_vals = {}
            for name, samples, losses in mtype[curves_key]:
                vals = agg_func(losses)
                keywords = [highlight_keyword] if isinstance(highlight_keyword, str) else (highlight_keyword or [])
                highlighted = any(k in name for k in keywords)
                alpha = highlight_alpha if highlighted else 0.2
                # Without a highlight keyword every curve keeps its model type's linestyle
                linestyle = mtype['linestyle'] if (highlighted or highlight_keyword is None) else non_highlight_linestyle
                if log_scale and log_x:
                    plot_func = ax.loglog
                elif log_scale:
                    plot_func = ax.semilogy
                elif log_x:
                    plot_func = ax.semilogx
                else:
                    plot_func = ax.plot
                if highlighted:
                    label_name = {
                        'prnn_nonlin_3L8_6m3': 'PRNN (nonlinear) [6, 3]',
                        'prnn_energy_1L8_12m6': 'PRNN (energy) [12, 6]',
                    }.get(name, mtype['label'])
                    plot_func(samples, vals, linestyle=linestyle, linewidth=1, color=color, alpha=alpha, zorder=2, label=label_name)
                else:
                    plot_func(samples, vals, linestyle=linestyle, linewidth=1, color=color, alpha=alpha, zorder=1)
                for s, v in zip(samples, vals):
                    sample_vals.setdefault(s, []).append(v)

            # Scatter the minimum across configs for each sample count
            if sample_vals:
                xs = sorted(sample_vals.keys())
                ys = [min(sample_vals[s]) for s in xs]
                all_min_ys.extend(ys)
                if log_scale and log_x:
                    plot_func = ax.loglog
                elif log_scale:
                    plot_func = ax.semilogy
                elif log_x:
                    plot_func = ax.semilogx
                else:
                    plot_func = ax.plot
                # Without highlighting, the min scatter carries the legend entry for each model type
                scatter_label = mtype['label'] if highlight_keyword is None else None
                plot_func(xs, ys, linestyle='none', marker='o', markersize=5, color=color, label=scatter_label, zorder=2)

        ax.set_xlabel('Number of Training Samples')
        if log_scale and log_x:
            ax.set_ylabel(ylabel, y=0.6, labelpad=-5)
        else:
            ax.set_ylabel(ylabel)
        if log_x:
            ax.set_xticks(xlog_xticks)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.set_xticks([], minor=True)
        else:
            ax.set_xticks(xticks)
            ax.set_xticks([], minor=True)
            ax.set_xlim(xmin=0)
        if cur_ylim is not None:
            ax.set_ylim(cur_ylim)
        elif all_min_ys:
            ymax = max(all_min_ys) * 1.2
            if log_scale:
                ax.set_ylim(top=ymax)
            else:
                ax.set_ylim(0, ymax)
        elif not log_scale:
            ax.set_ylim(ymin=0)
        legend_loc = 'lower left' if (log_scale and log_x and type == 'vary_all_v2') else 'upper right'
        # legend_loc = 'upper right'
        ax.legend(loc=legend_loc)
        ax.grid(True, alpha=0.3)
        if log_scale and log_x:
            log_suffix = '_loglog'
        elif log_scale:
            log_suffix = '_log'
        elif log_x:
            log_suffix = '_logx'
        else:
            log_suffix = ''
        plt.savefig(f"{save_folder}{output_name}{suffix}{log_suffix}.pdf", bbox_inches='tight', format='pdf')
        # plt.savefig(f"{save_folder}{output_name}{suffix}{log_suffix}.png", dpi=300, bbox_inches='tight', format='png')
        if not log_x:
            ax.set_xlim(0, 128)
            ax.set_xticks(xlim_xticks)
            plt.savefig(f"{save_folder}{output_name}{suffix}{log_suffix}_xlim.pdf", bbox_inches='tight', format='pdf')
            # plt.savefig(f"{save_folder}{output_name}{suffix}{log_suffix}_xlim.png", dpi=300, bbox_inches='tight', format='png')
        plt.close()

print(f"Saved {output_name}.{{pdf,png}}")
