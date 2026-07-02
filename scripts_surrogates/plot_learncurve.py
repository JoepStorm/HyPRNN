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
highlight_keyword = 'm3'  #'scaleMU'  #'scaleMu'  #'_relu'  #'_sigmoid'  # configs matching this get higher opacity (set to None to disable)
highlight_alpha = 1.0
non_highlight_linestyle = '-'

output_name = f"6m3/{type}_comparison"

save_folder = f'../trained_models/train_{type}/'
folder = f'../trained_models/train_{type}/'
training_samples = [2, 4, 8, 12, 16, 24, 32, 64, 96, 128, 192, 256, 512]
xticks = [2, 32, 64, 128, 192, 256, 512]
xlim_xticks = [2, 8, 16, 24, 32, 64, 96, 128]
xlog_xticks = [2, 4, 8, 16, 32, 64, 128, 256, 512]
# ylim = (1/48, 1.4)  # None  # set to e.g. (0, 0.1) to fix y-axis across plots, None for automatic
# ylim = (0.01, 1.4)  # None  # set to e.g. (0, 0.1) to fix y-axis across plots, None for automatic
ylim = (0.04, 250)

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
'prnn_lin': {
        'configs': [
            'prnn_lin_2m',
            'prnn_lin_3m',
            'prnn_lin_6m',
            # 'prnn_lin_6m3',
            'prnn_lin_12m',
        ],
        'color': 'C1',
        'linestyle': '-.',
        'label': 'PRNN (linear)',
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
    'prnn_lin': {
        'configs': [
            'prnn_lin_1L_6m',
            'prnn_lin_3L_2m',
            'prnn_lin_3L_3m',
            'prnn_lin_3L_6m',
            'prnn_lin_3L_6m3',
            'prnn_lin_3L_12m',
        ],
        'color': 'C1',
        'linestyle': '-.',
        'label': 'PRNN (linear)',
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
    'prnn_lin': {
        'configs': [
            'prnn_lin_1L_6m',
            'prnn_lin_3L_2m',
            'prnn_lin_3L_3m',
            'prnn_lin_3L_6m',
            'prnn_lin_3L_6m3',
            'prnn_lin_3L_12m',
        ],
        'color': 'C1',
        'linestyle': '-.',
        'label': 'PRNN (linear)',
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
        # try:
        #     losses_l2 = np.loadtxt(path_l2, delimiter=',')
        #     if losses_l2.ndim == 1:
        #         losses_l2 = losses_l2.reshape(-1, 1)
        #     mtype['curves_l2'].append((name, training_samples, losses_l2))
        # except FileNotFoundError:
        #     pass  # L2 losses are optional

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
                alpha = highlight_alpha if (highlight_keyword and highlight_keyword in name) else 0.1
                linestyle = non_highlight_linestyle if not highlight_keyword in name else mtype['linestyle']
                if log_scale and log_x:
                    plot_func = ax.loglog
                elif log_scale:
                    plot_func = ax.semilogy
                elif log_x:
                    plot_func = ax.semilogx
                else:
                    plot_func = ax.plot
                if highlight_keyword in name:
                    if name == 'prnn_nonlin_3L8_6m3':
                        label_name = 'PRNN (nonlinear) [6, 3]'
                    if name == 'prnn_lin_3L_6m3':
                        label_name = 'PRNN (linear) [6, 3]'
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
                # plot_func(xs, ys, linestyle='none', marker='o', markersize=5, color=color, label=mtype['label'], zorder=2)
                plot_func(xs, ys, linestyle='none', marker='o', markersize=5, color=color, zorder=2)

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
        # legend_loc = 'lower left' if (log_scale and log_x and type == 'vary_all_v2') else 'upper right'
        legend_loc = 'upper right'
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
