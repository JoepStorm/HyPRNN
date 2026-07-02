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


def plot_stress_strain_curves(F_true, sigma_F_true, U_eq_true, sigma_U_eq_true,
                              sigma_F_pred=None, sigma_U_pred=None, savename='tmp', png=False, color_converged=False, plot_U_U=False,
                              F_P=True, figsize=(10, 4)):
    """
    Plot stress-strain curves for deformation gradient F and strain U.

    Parameters:
    -----------
    F_true : array-like, shape [batch_size, n_steps, 2, 2]
        True deformation gradient values
    sigma_F_true : array-like, shape [batch_size, n_steps, 2, 2]
        True stress values corresponding to F
    U_eq_true : array-like, shape [batch_size, n_steps, 2, 2]
        True strain values
    sigma_U_eq_true : array-like, shape [batch_size, n_steps, 2, 2]
        True stress values corresponding to U
    F_pred : array-like, shape [batch_size, n_steps, 2, 2], optional
        Predicted deformation gradient values
    sigma_F_pred : array-like, shape [batch_size, n_steps, 2, 2], optional
        Predicted stress values corresponding to F
    U_pred : array-like, shape [batch_size, n_steps, 2, 2], optional
        Predicted strain values
    sigma_U_pred : array-like, shape [batch_size, n_steps, 2, 2], optional
        Predicted stress values corresponding to U
    F_P : bool, optional
        If True, sigma_U is treated as non-symmetric (4 components instead of 3),
        and sigma_F predictions show individual components instead of symmetric ones
    figsize : tuple, optional
        Figure size (width, height)

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(2, 4, figsize=figsize)

    # if only a single sample is provided, add dimension for batch size
    if F_true.ndim == 3:
        F_true = F_true[np.newaxis, ...]
    if sigma_F_true.ndim == 2:
        sigma_F_true = sigma_F_true[np.newaxis, ...]
    if U_eq_true.ndim == 3:
        U_eq_true = U_eq_true[np.newaxis, ...]
    if sigma_U_eq_true.ndim == 3:
        sigma_U_eq_true = sigma_U_eq_true[np.newaxis, ...]
    if sigma_F_pred is not None and sigma_F_pred.ndim == 3:
        sigma_F_pred = sigma_F_pred[np.newaxis, ...]
    if sigma_U_pred is not None and sigma_U_pred.ndim == 3:
        sigma_U_pred = sigma_U_pred[np.newaxis, ...]

    batch_size = F_true.shape[0]
    num_converged = 0

    # Loop over batch dimension
    for b in range(batch_size):

        if color_converged:
            # Color basd on whether the sample has converged, measured by whether the stresses start with 0, 0. (paths are shifted with constant length)
            if np.allclose(sigma_F_true[b, 0], 0, atol=1e-3) and np.allclose(sigma_U_eq_true[b, 0], 0, atol=1e-3):
                color = 'tab:red'   # not converged
            else:
                color = 'tab:green' # converged
                num_converged += 1
        else:
            # Get color for this batch sample
            color = colours[b % len(colours)]

        # Plot F vs sigma_F (top row)
        ax[0, 0].plot(F_true[b, :, 0, 0], sigma_F_true[b, :, 0], '-', color=color, label='True' if b == 0 else '')
        ax[0, 3].plot(F_true[b, :, 1, 1], sigma_F_true[b, :, 1], '-', color=color, label='True' if b == 0 else '')
        ax[0, 1].plot(F_true[b, :, 1, 0], sigma_F_true[b, :, 2], '-', color=color, label='True' if b == 0 else '')
        # ax[0, 2].plot(F_true[b, :, 0, 1], sigma_F_true[b, :, 2], '-', color=color, label='True' if b == 0 else '')
        ax[0, 2].plot(F_true[b, :, 0, 1], sigma_F_true[b, :, 3], '-', color=color, label='True' if b == 0 else '')

        # Plot U vs sigma_U (bottom row)
        ax[1, 0].plot(U_eq_true[b, :, 0, 0], sigma_U_eq_true[b, :, 0, 0], '-', color=color, label='True' if b == 0 else '')
        ax[1, 1].plot(U_eq_true[b, :, 1, 0], sigma_U_eq_true[b, :, 1, 0], '-', color=color, label='True' if b == 0 else '')
        if F_P:
            ax[1, 2].plot(U_eq_true[b, :, 0, 1], sigma_U_eq_true[b, :, 0, 1], '-', color=color,label='True' if b == 0 else '')
            ax[1, 3].plot(U_eq_true[b, :, 1, 1], sigma_U_eq_true[b, :, 1, 1], '-', color=color, label='True' if b == 0 else '')
        else:
            ax[1, 2].plot(U_eq_true[b, :, 1, 1], sigma_U_eq_true[b, :, 1, 1], '-', color=color, label='True' if b == 0 else '')

        # Plot predicted values if provided
        if  sigma_F_pred is not None:
            ax[0, 0].plot(F_true[b, :, 0, 0], sigma_F_pred[b, :, 0, 0], '--', color=color, label='Predicted' if b == 0 else '')
            ax[0, 1].plot(F_true[b, :, 1, 0], sigma_F_pred[b, :, 1, 0], '--', color=color, label='Predicted' if b == 0 else '')
            ax[0, 2].plot(F_true[b, :, 0, 1], sigma_F_pred[b, :, 0, 1], '--', color=color, label='Predicted' if b == 0 else '')
            ax[0, 3].plot(F_true[b, :, 1, 1], sigma_F_pred[b, :, 1, 1], '--', color=color, label='Predicted' if b == 0 else '')

        if sigma_U_pred is not None:
            ax[1, 0].plot(U_eq_true[b, :, 0, 0], sigma_U_pred[b, :, 0, 0], '--', color=color, label='Predicted' if b == 0 else '')
            ax[1, 1].plot(U_eq_true[b, :, 1, 0], sigma_U_pred[b, :, 1, 0], '--', color=color, label='Predicted' if b == 0 else '')
            if F_P:
                ax[1, 2].plot(U_eq_true[b, :, 0, 1], sigma_U_pred[b, :, 0, 1], '--', color=color,label='Predicted' if b == 0 else '')
                ax[1, 3].plot(U_eq_true[b, :, 1, 1], sigma_U_pred[b, :, 1, 1], '--', color=color, label='Predicted' if b == 0 else '')
            else:
                ax[1, 2].plot(U_eq_true[b, :, 1, 1], sigma_U_pred[b, :, 1, 1], '--', color=color, label='Predicted' if b == 0 else '')


    if color_converged:
        print(f"Number of converged paths: {num_converged}")
    # Set all the axes labels. First row is F, sigma_F, second row is U, sigma_U
    ax[0, 0].set_xlabel(r'$F_{xx}$', labelpad=0.05)
    ax[0, 0].set_ylabel(r'$\sigma^F_{xx}$', labelpad=0.02)
    ax[0, 1].set_xlabel(r'$F_{xy}$', labelpad=0.05)
    ax[0, 1].set_ylabel(r'$\sigma^F_{xy}$', labelpad=0.02)
    ax[0, 2].set_xlabel(r'$F_{yx}$', labelpad=0.05)
    ax[0, 2].set_ylabel(r'$\sigma^F_{yx}$', labelpad=0.02)
    ax[0, 3].set_xlabel(r'$F_{yy}$', labelpad=0.05)
    ax[0, 3].set_ylabel(r'$\sigma^F_{yy}$', labelpad=0.02)

    ax[1, 0].set_xlabel(r'$\mathcal{U}^{eq}_{xx}-1$', labelpad=0.05)
    ax[1, 0].set_ylabel(r'$\sigma^{U^{eq}}_{xx}$', labelpad=0.02)
    ax[1, 1].set_xlabel(r'$\mathcal{U}^{eq}_{xy}$', labelpad=0.05)
    ax[1, 1].set_ylabel(r'$\sigma^{U^{eq}}_{xy}$', labelpad=0.02)
    if F_P:
        ax[1, 2].set_xlabel(r'$\mathcal{U}^{eq}_{yx}$', labelpad=0.05)
        ax[1, 2].set_ylabel(r'$\sigma^{U^{eq}}_{yx}$', labelpad=0.02)
        ax[1, 3].set_xlabel(r'$\mathcal{U}^{eq}_{yy}-1$', labelpad=0.05)
        ax[1, 3].set_ylabel(r'$\sigma^{U^{eq}}_{yy}$', labelpad=0.02)
    else:
        ax[1, 2].set_xlabel(r'$\mathcal{U}^{eq}_{yy}-1$', labelpad=0.05)
        ax[1, 2].set_ylabel(r'$\sigma^{U^{eq}}_{yy}$', labelpad=0.02)

    # Add legends to plots that have both true and predicted data
    if (sigma_F_pred is not None) or (sigma_U_pred is not None):
        for i in range(2):
            # for j in range(4 if i == 0 else 3):
            ax[i, 0].legend()

    plt.tight_layout()
    plt.savefig(f"{savename}.pdf", bbox_inches='tight', format='pdf')
    if png:
        plt.savefig(f"{savename}.png", bbox_inches='tight', format='png', dpi=300)


    # Plot 3x1 U-U curves with the different components:
    if plot_U_U:
        # Columns: (U_00 vs U_01), (U_00 vs U_11), (U_01 vs U_11)
        fig_u, ax_u = plt.subplots(1, 3, figsize=(10, 3.5))

        for b in range(batch_size):
            if color_converged:
                if np.allclose(sigma_F_true[b, 0], 0, atol=1e-3) and np.allclose(sigma_U_eq_true[b, 0], 0, atol=1e-3):
                    color = 'tab:red'   # not converged
                else:
                    color = 'tab:green' # converged
            else:
                color = colours[b % len(colours)]

            U00 = U_eq_true[b, :, 0, 0]
            U01 = U_eq_true[b, :, 1, 0]
            U11 = U_eq_true[b, :, 1, 1]

            ax_u[0].plot(U00, U01, '-', color=color, alpha=0.9)
            ax_u[1].plot(U00, U11, '-', color=color, alpha=0.9)
            ax_u[2].plot(U01, U11, '-', color=color, alpha=0.9)

        ax_u[0].set_xlabel(r'$\mathcal{U}_{xx}-1$')
        ax_u[0].set_ylabel(r'$\mathcal{U}_{xy}$')

        ax_u[1].set_xlabel(r'$\mathcal{U}_{xx}-1$')
        ax_u[1].set_ylabel(r'$\mathcal{U}_{yy}-1$')

        ax_u[2].set_xlabel(r'$\mathcal{U}_{xy}$')
        ax_u[2].set_ylabel(r'$\mathcal{U}_{yy}-1$')

        # for j in range(3):
        #     ax_u[j].grid(True, alpha=0.3)
        #     try:
        #         ax_u[j].set_aspect('equal', adjustable='box')
        #     except Exception:
        #         pass

        fig_u.tight_layout()
        fig_u.savefig(f"{savename}_U_pairs.pdf", bbox_inches='tight', format='pdf')
        if png:
            fig_u.savefig(f"{savename}_U_pairs.png", bbox_inches='tight', format='png', dpi=300)

    plt.close()

    return fig, ax


# if main
if __name__ == '__main__':
    """
    Main function that replicates the original behavior of the script.
    This loads data and creates the visualization.
    """
    # Load data
    from data_utils import LDDataset_PK1, LDDataset

    settings = {
        # 'data_path': f'../data/vfrac_ratio_smallmu_biggerdomain/literature_matprops/mixed_t100',
        # 'data_path': f'../data/vfrac_ratio_smallmu_biggerdomain/10runs_literature_matprops/mixed_t50_merged',
        # 'data_path': f'../data/vfrac_ratio_smallmu_biggerdomain/triplets_1/mixed_t5_seed1',
        # 'data_path': f'../data/vfrac_ratio_smallmu_biggerdomain/lit_props_constant/mixed_t50_seed0',
        # 'data_path': f'../data/vfrac_ratio_smallmu_biggerdomain/triplets_0/mixed_t50_seed0',
        # 'data_path': f'../data/woodchip_conhull_batch/mixed_t50_seed0',
        # 'data_path': f'../data/vary_vfrac_ratio/vary_vfrac_ratio/mixed_t50_merged',
        # 'data_path': f'../data/uniaxial_v2/uniaxial',
        # 'data_path': f'../data/uniaxial_debug/uniaxial',
        # 'data_path': f'../data/woodchip/woodchip_periodic_relax_simple_100_compression',
        # 'data_path': f'../datasets/vfrac_ratio_smallmu_biggerdomain/shuffled_3430',
        # 'data_path': f'../datasets/no_geo/nfib25_0.0000_2.00_mesh384_t50',
        # 'data_path': f'../data/deposition/v1/mixed_t50_merged',
        # 'data_path': f'../data/deposition/v5_bigrve/mixed_t50_merged',
        # 'data_path': f'../data/deposition/v10/mixed_t50_merged',
        'data_path': f'../data/deposition/filler/dataset_combi_v4v6/mixed_t50_merged',
        'seq_length': 50,
    }
    settings['matdata_path'] = settings['data_path'] + '_matparam.data'
    dataset = LDDataset_PK1(settings['data_path'], settings['seq_length'], mat_file=settings['matdata_path'])
    # dataset = LDDataset_PK1(settings['data_path'], settings['seq_length']) #, mat_file=settings['matdata_path'])

    all_indices = np.arange(dataset.F.shape[0])
    # test_indices = all_indices[350:]
    test_indices = all_indices[:]
    testset = dataset.get_subset(test_indices)

    # num_samples = 686 # 384  #8

    # full_DF = testset['F'] #[-num_samples:]
    # full_sigma = testset['sig_F'] #[-num_samples:]
    # true_strain = testset['x'] #[-num_samples:]
    # true_stress = testset['sig_U_eq_unnorm'] #[-num_samples:]
    #
    # # print(f"full_DF: {full_DF[0]}")
    #
    #
    # # # To compare with a different dataset, we load that dataset and pass it as the predicted values
    # # # alternative_file = f'../data/vfrac_ratio_smallmu_biggerdomain/uniaxial_1/uniaxial_tension_t50'
    # # alternative_file = f'../data/woodchip/woodchip_periodic_relax_simple_80_compression'
    # # dataset_alternative = LDDataset_PK1(alternative_file, settings['seq_length']) #, mat_file=f'../data/vfrac_ratio_smallmu_biggerdomain/uniaxial_1/uniaxial_tension_t50_matparam.data')
    # # testset_alternative = dataset_alternative.get_subset(test_indices)
    # # sigma_F_pred_alternative = testset_alternative['sig_F'][:]
    # # pred_stress_alternative = testset_alternative['sig_U_eq_unnorm'][:]
    # #
    # # # print(f"full_DF: {full_DF}")
    # # # print(f"F alternative: {testset_alternative['F']}")
    # # # assert np.allclose(full_DF, testset_alternative['F']), "F's don't match"
    #
    # plot_stress_strain_curves(
    #     F_true=full_DF,
    #     sigma_F_true=full_sigma,
    #     U_eq_true=true_strain,
    #     sigma_U_eq_true=true_stress,
    #     # sigma_U_pred=pred_stress_alternative,
    #     # savename=no_geo_file + '_curves',
    #     savename=settings['data_path'] + '_curves',
    #     color_converged=True,
    #     plot_U_U=True,
    #     png=False
    # )

    # dataset = LDDataset(settings['data_path'], settings['seq_length'], mat_file=settings['matdata_path'])
    # dataset = LDDataset(settings['data_path'], settings['seq_length'], mat_file=settings['matdata_path'], mat_features=['mix'])
    dataset = LDDataset(settings['data_path'], settings['seq_length'], mat_file=settings['matdata_path'], mat_features=['fil_frac'])
    # dataset = LDDataset(settings['data_path'], settings['seq_length'])
    testset = dataset.get_subset(test_indices)
    # plot_PK2_curves(
    #     E_true=testset['x'],
    #     PK2_true=testset['PK2_eq_unnorm'],
    #     savename=settings['data_path'] + '_PK2_curves',
    #     png=True
    # )

    plot_PK2_curves(
        E_true=testset['x'],
        PK2_true=testset['PK2_eq_unnorm'],
        savename=settings['data_path'] + '_PK2_curves',
        color_variable=testset['m'][:, 0],  # color by 'mix' material parameter
        # color_label='mix',
        color_label='Pellet fraction',
        png=True
    )