"""
Train a single surrogate model.

Edit the hyperparameters in this file directly, and run it to train a model & visualize its predictions.
Switch between mode = 'train' and mode = 'test'.

"""
import os
import random
import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')  # force jax to use cpu; which showed to be faster

from trainer import Trainer
from data_utils import LDDataset, Config, save_settings, tensor_to_matrix
from HyPRNN import create_shared_hyper_prnn_model
from StandardNN import create_nn_model
from material_params import FUNGI_MU, FUNGI_LAMBDA

import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use(['science', 'bright'])
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
rc('text', usetex=True)

mode = 'train'
# mode = 'test'

plot_uni = True  # plot uniaxial tension/compression curves per mix value after test

mix_label = 'fil_frac' # 'mix'

settings = {
    'savefolder': f'../trained_models/deposition/example/',

    # 'data_path': f'../data/vary_all/mixed_t50_mergedv2',
    # 'matdata_path': f'../data/vary_all/mixed_t50_mergedv2_matparam.data',

    'data_path': f'../data/deposition/filler/dataset_combi_v4v6/mixed_t50_merged',
    'matdata_path': f'../data/deposition/filler/dataset_combi_v4v6/mixed_t50_merged_matparam.data',

    'seq_length': 50,
    'train_samples': 256,   # 128
    'train_batch_size': 2,
    'num_samples': 512,
    # 'num_samples': 6144,
    'val_test_samples': 128,
    # 'val_test_samples': 2048,

    'model_type': 'hyprnn',
    # 'model_type': 'nn',

    # 'encoder_type': 'NonLinear',
    'encoder_type': 'Linear',

    'decoder_type': 'HyperSparseLayer',

    'norm_stresses': True,
    'stress_scaling_feature': None,  #'mu',  # None  # e.g. 'mu': divide stresses by this mat param before normalization to equalize magnitudes

    'mat_points': 6,

    'lr_schedule_steps': 10000,          # interval over which learning rate is scheduled. Ideally finish around this # of update steps. Linked to epochs through num batches
    'max_epochs': 3000,  # 2000
    'warmup_epochs': 0,
    'base_lr': 1e-3,   # 1e-2                 # base learning rate (this is the maximum, both warmup and decay afterwards are lower)
    'min_lr_factor': 1, # 5e-2,               # minimum lr: scheduler goes from base_lr to base_lr * min_lr_factor (e.g. 0.01 -> 0.0005)
    'patience': 50,                     # early stopping epochs
    'interval': 1,                      # interval for which we compute validation loss (once every x epochs)

    'feature_dim': 3,
    'verbose': True,
    'seed': 0
}

if settings['model_type'] == 'nn':
    # settings['mat_parameters'] = ['vfrac', 'ratio']  # ['mu', 'vfrac', 'ratio']
    # settings['norm_matparams'] = [True, True]  # [True, True, True]  # For NN
    # settings['mat_parameters'] = ['mu', 'vfrac', 'ratio']
    # settings['norm_matparams'] = [True, True, True]  # For NN
    settings['mat_parameters'] = ['mu']
    settings['norm_matparams'] = [True]  # For NN
    settings['nn_hidden_sizes'] = [64, 64, 64]
    settings['nn_activation'] = 'sigmoid'
    settings['nn_bias'] = True
if settings['model_type'] == 'hyprnn':
    # settings['mat_parameters'] = ['mu', 'lambda', 'vfrac', 'ratio']
    # settings['norm_matparams'] = [False, False, True, True]
    # settings['shared_micro_features'] = [2, 3]  # vfrac, ratio feed the shared hypernet
    # settings['mat_micro_features'] = [0, 1]      # mu, lambda go directly to material model
    # settings['hyper_hidden_sizes'] = (8, 8, 8)
    # settings['hyper_activation'] = 'sigmoid'
    # settings['encoder_n_layers'] = 1
    # settings['encoder_activation'] = 'sigmoid'
    # # encoder/decoder micro_features are set automatically by create_shared_hyper_prnn_model

    # Mixture
    settings['mat_parameters'] = ['mu', 'lambda', mix_label]
    settings['norm_matparams'] = [False, False, True]
    settings['shared_micro_features'] = [2]
    settings['mat_micro_features'] = [0, 1]
    # settings['hyper_hidden_sizes'] = (8, 8, 8)  # (8,)
    settings['hyper_hidden_sizes'] = (8,)
    settings['hyper_activation'] = 'sigmoid'


# Create save directory if it doesn't exist
if not os.path.exists(settings['savefolder']):
   os.makedirs(settings['savefolder'], exist_ok=True)
   print(f"Created a new save directory: {settings['savefolder']}")

# Load data
dataset = LDDataset(settings['data_path'], seq_length=settings['seq_length'], num_samples=settings['num_samples'], mat_file=settings['matdata_path'], mat_features=settings['mat_parameters'], norm_stresses=settings['norm_stresses'], norm_matparams=settings['norm_matparams'], stress_scaling_feature=settings['stress_scaling_feature'])


# fix test & validation set
all_indices = np.arange(settings['num_samples'])
# np.random.shuffle(all_indices)

train_indices = all_indices[:int(-2*settings['val_test_samples'])]
val_indices = all_indices[int(-2*settings['val_test_samples']):int(-settings['val_test_samples'])]
test_indices = all_indices[int(-settings['val_test_samples']):]
print(f"Train/Val/Test sizes: {train_indices.shape[0]}/{val_indices.shape[0]}/{test_indices.shape[0]}")

valset = dataset.get_subset(val_indices)
testset = dataset.get_subset(test_indices)

# Setup Keys and Seed
seed = settings['seed']
seed += 1
key = jax.random.PRNGKey(seed)
np.random.seed(seed)
random.seed(seed)

# Create random training subset
cur_indices = train_indices.copy()
np.random.shuffle(cur_indices)    # optionally turn off shuffling to force same dataset
cur_indices = cur_indices[:settings['train_samples']]
trainset = dataset.get_subset(cur_indices)

# Compute stress_scaling_index for PRNN models (index of scaling feature in mat_parameters)
stress_scaling_index = None
if settings['stress_scaling_feature'] is not None and settings['mat_parameters'] is not None:
    if settings['stress_scaling_feature'] in settings['mat_parameters']:
        stress_scaling_index = settings['mat_parameters'].index(settings['stress_scaling_feature'])
        # PRNN models divide internal micro-stresses by this feature's raw value,
        # so it must NOT be normalized (otherwise scale would be wrong)
        if isinstance(settings['norm_matparams'], list) and settings['norm_matparams'][stress_scaling_index]:
            print(f"WARNING: stress_scaling_feature '{settings['stress_scaling_feature']}' is normalized "
                  f"(norm_matparams[{stress_scaling_index}]=True). PRNN internal scaling requires un-normalized values.")

if settings['model_type'] == 'hyprnn':
    model, params, material = create_shared_hyper_prnn_model(
        n_micro_raw=len(settings['mat_parameters']),
        shared_micro_features=settings['shared_micro_features'],
        n_matpts=settings['mat_points'],
        encoder_type=settings['encoder_type'],  # 'DefGrad' or 'LLT'
        stress_normalizer=dataset.stress_normalizer,
        mat_m_feats=settings['mat_micro_features'],
        random_key=key,
        shared_hidden_mult=settings.get('hyper_hidden_mult', 2),
        shared_output_mult=settings.get('hyper_output_mult', 4),
        hidden_dim=settings.get('hidden_dim', 8),
        hyper_hidden_sizes=settings.get('hyper_hidden_sizes', None),
        hyper_activation=settings.get('hyper_activation', 'sigmoid'),
        encoder_n_layers=settings.get('encoder_n_layers', 3),
        encoder_activation=settings.get('encoder_activation', 'sigmoid'),
        stress_scaling_index=stress_scaling_index,
    )
elif settings['model_type'] == 'nn':
    model, params, material = create_nn_model(random_key=key, hidden_sizes=settings['nn_hidden_sizes'], num_mat_features=len(settings['mat_parameters']), activation=settings['nn_activation'], use_bias=settings['nn_bias'])
lr_config = Config(warmup_epochs=settings['warmup_epochs'], schedule_steps=settings['lr_schedule_steps'], steps_per_epoch=settings['train_samples'] / settings['train_batch_size'], base_learning_rate = settings['base_lr'], min_lr_factor = settings['min_lr_factor'] )

train_handler = Trainer(model, params, material=material, lr_config=lr_config, random_key=key, out_norm_factor=1)
# Note that the out_norm_factor doesn't work, as we scale the components individually. So after computing MSE, we can't "factor" it back.

if mode == 'train':
    # Train the model
    train_handler.train(trainset, valset, test_data=testset, **settings)

    # Save Model
    train_handler.save(settings['savefolder'] + 'params')

    # Save dataset normalizers
    dataset.saveDataparams(settings['savefolder'] + 'normparams')

    # Save Settings
    save_settings(settings, f"{settings['savefolder']}settings")

    # Plot loss curves
    train_losses, val_losses, test_losses = train_handler.get_losses()
    epochs = np.arange(len(train_losses))

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(epochs, train_losses, label='Train', color=colours[0])
    ax.plot(epochs, val_losses, label='Validation', color=colours[1])
    ax.plot(epochs, test_losses, label='Test', color=colours[2])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.savefig(f"{settings['savefolder']}loss_curves.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Loss curves saved to {settings['savefolder']}loss_curves.png")

elif mode == 'test':
    from plot_LD_data import plot_PK2_curves
    # Load model
    params = train_handler.load(settings['savefolder'] + 'params') #, set_best_params=False)
    # params = train_handler.load(settings['savefolder'] + 'samples32_run0') #, set_best_params=False)
    # params = train_handler.load(settings['savefolder'] + 'samples512_run1') #, set_best_params=False)

    # Evaluate on test set
    test_denorm = dataset.stress_normalizer.make_denormalizer(testset.get('stress_scale'))
    test_loss = train_handler.eval_step_L1(train_handler._state, testset, test_denorm, material)
    test_loss_rel = train_handler.eval_relative_norm(train_handler._state, testset, test_denorm, material)
    test_loss_L2 = train_handler.eval_step_L2(train_handler._state, testset, test_denorm, material)
    print(f"Test loss (L1): {test_loss}, L2: {test_loss_L2}, relative: {test_loss_rel}")

    # Save test losses to file (consistent with train_multiconfig_batch.py)
    np.savetxt(f"{settings['savefolder']}test_losses.txt", np.array([[test_loss]]), delimiter=',')
    np.savetxt(f"{settings['savefolder']}test_losses_rel.txt", np.array([[test_loss_rel]]), delimiter=',')
    np.savetxt(f"{settings['savefolder']}test_losses_L2.txt", np.array([[test_loss_L2]]), delimiter=',')

    print(f"Plotting figures...")
    def process_stress_prediction(cur_dataset, sample, train_handler, params, material, dataset, settings, plot_savename, png):
        E_true = cur_dataset['x'][sample]

        input_E = E_true.reshape(1, settings['seq_length'], 2, 2)
        if dataset.mat_params:
            mat_params = cur_dataset['m'][sample].reshape(1, len(settings['mat_parameters']))
        else:
            mat_params = None
        predicted_stresses_tensor = train_handler._model.apply({'params': params['params']}, input_E, material, mat_params)

        # denormalize predicted stresses (undo global norm + per-sample stress scaling)
        sample_scale = cur_dataset['stress_scale'][sample].reshape(1) if 'stress_scale' in cur_dataset else None
        predicted_stresses_tensor = dataset.stress_normalizer.denormalize(predicted_stresses_tensor, scale=sample_scale)
        predicted_stresses = tensor_to_matrix(predicted_stresses_tensor)

        # # rotate stresses
        # re_oriented_stresses = jax.vmap(lambda q, s: q @ s @ q.T)(cur_dataset['Q'][sample], predicted_stresses[0])

        plot_PK2_curves(E_true, cur_dataset['PK2_eq_unnorm'][sample], predicted_stresses[0], savename=plot_savename, png=png)
        return

    # Plotting test curve(s):
    for idx in range(5):
        sample = idx + 1    # doing -0 does not give desired behavior
        process_stress_prediction( testset, -sample, train_handler, params, material, dataset, settings, f"{settings['savefolder']}test_curve_norm_-{sample}", png=True )

    if plot_uni and mix_label in settings.get('mat_parameters', []):
    # if plot_uni and 'fil_frac' in settings.get('mat_parameters', []):
        import jax.numpy as jnp
        mix_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        # mix_values = [-10, 0.5, 10]
        mu_val = FUNGI_MU
        lambda_val = FUNGI_LAMBDA

        # E = (F^T F - I) / 2 for uniaxial F stretched along `comp`:
        # E_cc = (lam^2 - 1) / 2, all other components = 0
        def make_E_seq(lambdas, comp):
            E = np.zeros((settings['seq_length'], 2, 2))
            E[:, comp, comp] = (lambdas ** 2 - 1.0) / 2.0
            return E

        lambdas_tension = np.linspace(1.0, 1.5, settings['seq_length'])
        lambdas_compression = np.linspace(1.0, 0.7, settings['seq_length'])

        mix_idx = settings['mat_parameters'].index(mix_label)
        mu_idx = settings['mat_parameters'].index('mu')
        lambda_idx = settings['mat_parameters'].index('lambda')

        cmap = plt.cm.viridis
        mix_norm = plt.Normalize(0.0, 1.0)

        # Plot uniaxial response for x (comp=0) and y (comp=1) directions separately
        for comp, dir_label in [(0, 'x'), (1, 'y')]:
            E_tension = make_E_seq(lambdas_tension, comp)
            E_compression = make_E_seq(lambdas_compression, comp)

            fig, ax = plt.subplots(figsize=(3.5, 2.5))
            for mix in mix_values:
                raw_mat = np.zeros((1, len(settings['mat_parameters'])))
                raw_mat[0, mu_idx] = mu_val
                raw_mat[0, lambda_idx] = lambda_val
                raw_mat[0, mix_idx] = mix
                print(f"raw_mat: {raw_mat}")
                mat_norm = dataset.M_normalizer.normalize(jnp.array(raw_mat))
                print(f"mat norm: {mat_norm}")
                color = cmap(mix_norm(mix))

                for E_seq in [E_tension, E_compression]:
                    input_E = jnp.array(E_seq).reshape(1, settings['seq_length'], 2, 2)
                    pred = train_handler._model.apply(
                        {'params': params['params']}, input_E, material, mat_norm
                    )
                    pred = dataset.stress_normalizer.denormalize(pred, scale=None)
                    pred = tensor_to_matrix(pred)
                    pred = np.array(pred[0])  # (seq_length, 2, 2)
                    ax.plot(E_seq[:, comp, comp], pred[:, comp, comp], color=color, lw=1.5, ls='-')

            comp_lbl = comp + 1  # 1-indexed for labels
            ax.set_xlabel(rf'$E_{{{comp_lbl}{comp_lbl}}}$')
            ax.set_ylabel(rf'$S_{{{comp_lbl}{comp_lbl}}}$')
            ax.axhline(0, color='k', lw=0.5, ls=':')
            ax.axvline(0, color='k', lw=0.5, ls=':')
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=mix_norm)
            fig.colorbar(sm, ax=ax, label=mix_label, shrink=0.8)
            plt.tight_layout()
            uni_path = f"{settings['savefolder']}uniaxial_mix_curves_{dir_label}.pdf"
            fig.savefig(uni_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"Uniaxial curves ({dir_label}) saved to {uni_path}")
