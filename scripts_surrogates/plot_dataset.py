import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')  # force jax to use cpu; which showed to be faster

import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use(['science', 'bright'])
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
rc('text', usetex=True)
components = ['$_{xx}$', '$_{yy}$', '$_{xy}$']


def plot_PK2_curves(E_true, PK2_true, PK2_pred=None, savename='tmp', png=False, figsize=(8, 2.5),
                    color_variable=None, color_label='', cmap='viridis', cbar_rect=None):

    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # if only a single sample is provided, add dimension for batch size
    if E_true.ndim == 3:
        E_true = E_true[np.newaxis, ...]
    if PK2_true.ndim == 3:
        PK2_true = PK2_true[np.newaxis, ...]
    if PK2_pred is not None:
        if PK2_pred.ndim == 3:
            PK2_pred = PK2_pred[np.newaxis, ...]

    batch_size = E_true.shape[0]

    if color_variable is not None:
        color_variable = np.asarray(color_variable)
        cmap_obj = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=color_variable.min(), vmax=color_variable.max())

    # Loop over batch dimension
    for b in range(batch_size):
        if color_variable is not None:
            color = cmap_obj(norm(color_variable[b]))
        else:
            color = colours[b % len(colours)]

        # Plot F vs sigma_F (top row)
        ax[0].plot(E_true[b, :, 0, 0], PK2_true[b, :, 0, 0], '-', color=color, label='True' if b == 0 else '')
        ax[1].plot(E_true[b, :, 1, 1], PK2_true[b, :, 1, 1], '-', color=color, label='True' if b == 0 else '')
        ax[2].plot(E_true[b, :, 0, 1], PK2_true[b, :, 0, 1], '-', color=color, label='True' if b == 0 else '')

        # Plot predicted values if provided
        if PK2_pred is not None:
            ax[0].plot(E_true[b, :, 0, 0], PK2_pred[b, :, 0, 0], '--', color=color, label='Predicted' if b == 0 else '')
            ax[1].plot(E_true[b, :, 1, 1], PK2_pred[b, :, 1, 1], '--', color=color, label='Predicted' if b == 0 else '')
            ax[2].plot(E_true[b, :, 0, 1], PK2_pred[b, :, 0, 1], '--', color=color, label='Predicted' if b == 0 else '')

    # Set all the axes labels. First row is F, sigma_F, second row is U, sigma_U
    ax[0].set_xlabel(r'$E_{xx}$', labelpad=0.05)
    # ax[0].set_ylabel(r'$S^{eq}_{xx}$', labelpad=0.02)
    ax[0].set_ylabel(r'$S_{xx}$', labelpad=0.02)
    ax[1].set_xlabel(r'$E_{yy}$', labelpad=0.05)
    # ax[1].set_ylabel(r'$S^{eq}_{yy}$', labelpad=0.02)
    ax[1].set_ylabel(r'$S_{yy}$', labelpad=0.02)
    ax[2].set_xlabel(r'$E_{xy}$', labelpad=0.05)
    # ax[2].set_ylabel(r'$S^{eq}_{xy}$', labelpad=0.02)
    ax[2].set_ylabel(r'$S_{xy}$', labelpad=0.02)

    if color_variable is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        rect = cbar_rect if cbar_rect is not None else [0.25, 0.23, 0.01, 0.4]
        cax = fig.add_axes(rect)
        cb = fig.colorbar(sm, cax=cax, label=color_label)
        # cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, .9])
    elif PK2_pred is not None:
        ax[0].legend()

    plt.tight_layout()
    plt.savefig(f"{savename}{color_label}.pdf", bbox_inches='tight', format='pdf')
    if png:
        plt.savefig(f"{savename}{color_label}.png", bbox_inches='tight', format='png', dpi=300)

    plt.close()

    return fig, ax


if __name__ == '__main__':
    # Load a dataset and plot its PK2 stress-strain curves.
    from data_utils import LDDataset

    settings = {
        'data_path': f'../data/vary_all/mixed_t50_mergedv2',
        # 'data_path': f'../data/deposition/filler/dataset_combi_v4v6/mixed_t50_merged',
        'seq_length': 50,
    }
    settings['matdata_path'] = settings['data_path'] + '_matparam.data'

    # Vary all
    dataset = LDDataset(settings['data_path'], settings['seq_length'], mat_file=settings['matdata_path'])
    # # Deposition
    # dataset = LDDataset(settings['data_path'], settings['seq_length'], mat_file=settings['matdata_path'], mat_features=['fil_frac'])

    # dataset = LDDataset(settings['data_path'], settings['seq_length'])
    test_indices = np.arange(dataset.F.shape[0])
    testset = dataset.get_subset(test_indices)

    # Standard
    plot_PK2_curves(
        E_true=testset['x'],
        PK2_true=testset['PK2_eq_unnorm'],
        savename=settings['data_path'] + '_PK2_curves',
        png=True
    )

    # # Custom color
    # plot_PK2_curves(
    #     E_true=testset['x'],
    #     PK2_true=testset['PK2_eq_unnorm'],
    #     savename=settings['data_path'] + '_PK2_curves',
    #     color_variable=testset['m'][:, 0],  # color by 'mix' material parameter
    #     # color_label='mix',
    #     color_label='Pellet fraction',
    #     png=True
    # )