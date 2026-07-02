"""For a shared dataset, compare truth vs. predictions of nonlinear PRNN, linear PRNN, and NN
for the median-error test sample (ranked by nonlinear PRNN error)."""
import os
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_platform_name', 'cpu')

from trainer import Trainer
from data_utils import LDDataset, Config, load_settings, tensor_to_matrix
from LDprnn_shared_hyper import create_shared_hyper_prnn_model
from LDnn import create_nn_model

import matplotlib.pyplot as plt
from matplotlib import rc
try:
    plt.style.use(['science', 'bright'])
    rc('text', usetex=True)
except:
    print("not using custom colors")
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

training_samples_array = [2, 4, 8, 12, 16, 24, 32, 64, 96, 128, 192, 256, 512]

# ================== Settings ==================
# nonlin_folder   = '../trained_models/train_vary_all/prnn_nonlin_1L8_6m_sigmoid/'
nonlin_folder   = '../trained_models/train_vary_all_v2/prnn_nonlin_1L8_12m_sigmoid/'
nonlin_samples  = 512

linear_folder   = '../trained_models/train_vary_all_v2/prnn_lin_1L_6m/'
linear_samples  = 32

linear512_folder  = '../trained_models/train_vary_all_v2/prnn_lin_1L_6m/'
linear512_samples = 512

nn32_folder    = '../trained_models/train_vary_all_v2/nn_64_4/'
nn32_samples   = 32

# nn_folder    = '../trained_models/train_vary_all_v2/nn_64_3/'
nn_folder    = '../trained_models/train_vary_all_v2/nn_64_4/'
nn_samples   = 512

y_lims = [[-2, 0.1], [-0.1, 2.2], [-0.1, 1.7]]  # None

def median_run(folder, n_samples):
    """Return the run name (e.g. 'samples512_run3') whose test loss is closest to the median."""
    losses = np.loadtxt(f"{folder}test_losses.txt", delimiter=',')
    row = training_samples_array.index(n_samples)
    row_losses = losses[row]
    median_run_idx = int(np.argsort(row_losses)[len(row_losses) // 2])
    print(f"  {folder}: samples={n_samples}, run{median_run_idx} (loss={row_losses[median_run_idx]:.4f})")
    return f"samples{n_samples}_run{median_run_idx}"


nonlin_run    = median_run(nonlin_folder,   nonlin_samples)
linear_run    = median_run(linear_folder,   linear_samples)
linear512_run = median_run(linear512_folder, linear512_samples)
nn_run        = median_run(nn_folder,       nn_samples)
nn32_run      = median_run(nn32_folder,       nn32_samples)

plot_folder = nonlin_folder + 'statistics/'
# plot_folder = nn32_folder + 'statistics/'
os.makedirs(plot_folder, exist_ok=True)

eval_set = 'test'  # 'test' or 'val'


def load_model_and_predict(folder, run, eval_indices):
    """Load a trained model and its own dataset subset, return denormalized predictions.

    Each model builds its own testset so that mat_parameters and normalization are correct
    for that specific model (e.g. NN may use fewer material features than the PRNN).
    Returns predictions and the model's own testset (for y_true / x access if needed).
    """
    settings = load_settings(f"{folder}{run}_settings")
    key = jax.random.PRNGKey(settings['seed'])

    dataset = LDDataset(
        settings['data_path'], seq_length=settings['seq_length'],
        num_samples=settings['num_samples'],
        mat_file=settings['matdata_path'], mat_features=settings['mat_parameters'],
        norm_stresses=settings['norm_stresses'], norm_matparams=settings['norm_matparams'],
    )
    subset = dataset.get_subset(eval_indices)

    model_type = settings.get('model_type', 'shared_prnn')
    if model_type == 'nn':
        model, params, material = create_nn_model(
            random_key=key,
            hidden_sizes=tuple(settings['nn_hidden_sizes']),
            num_mat_features=len(settings['mat_parameters']),
            activation=settings['nn_activation'],
            use_bias=settings['nn_bias'],
        )
    elif model_type == 'shared_prnn':
        model, params, material = create_shared_hyper_prnn_model(
            random_key=key,
            n_micro_raw=len(settings['mat_parameters']),
            shared_micro_features=settings['shared_micro_features'],
            n_matpts=settings['mat_points'],
            encoder_type=settings['encoder_type'],
            shared_hidden_mult=settings.get('hyper_hidden_mult', 2),
            shared_output_mult=settings.get('hyper_output_mult', 4),
            hidden_dim=settings.get('hidden_dim', 8),
            stress_normalizer=dataset.stress_normalizer,
            mat_m_feats=settings['mat_micro_features'],
            hyper_hidden_sizes=settings.get('hyper_hidden_sizes', None),
            hyper_activation=settings.get('hyper_activation', 'sigmoid'),
            encoder_n_layers=settings.get('encoder_n_layers', 3),
            encoder_activation=settings.get('encoder_activation', 'softplus'),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    lr_config = Config(
        warmup_epochs=settings['warmup_epochs'],
        schedule_steps=settings['lr_schedule_steps'],
        steps_per_epoch=settings['train_samples'] / settings['train_batch_size'],
        base_learning_rate=settings['base_lr'],
        min_lr_factor=settings['min_lr_factor'],
    )
    trainer = Trainer(model, params, material=material, lr_config=lr_config, random_key=key, out_norm_factor=1)
    params = trainer.load(f"{folder}{run}")

    y_pred_norm = model.apply(
        {'params': params['params']}, subset['x'], material, micro_params=subset.get('m', None)
    )
    return dataset.stress_normalizer.denormalize(y_pred_norm), subset


# ================== Load primary (nonlinear PRNN) dataset and testset ==================
# Used for y_true, mask, and strain (x) — shared across all models.
primary_settings = load_settings(f"{nonlin_folder}{nonlin_run}_settings")
primary_dataset = LDDataset(
    primary_settings['data_path'], seq_length=primary_settings['seq_length'],
    num_samples=primary_settings['num_samples'],
    mat_file=primary_settings['matdata_path'], mat_features=primary_settings['mat_parameters'],
    norm_stresses=primary_settings['norm_stresses'], norm_matparams=primary_settings['norm_matparams'],
)

val_test_samples = int(primary_settings['val_test_samples'])
all_indices = np.arange(primary_settings['num_samples'])
if eval_set == 'val':
    eval_indices = all_indices[-2 * val_test_samples:-val_test_samples]
else:
    eval_indices = all_indices[-val_test_samples:]
primary_testset = primary_dataset.get_subset(eval_indices)
print(f"{eval_set} set: {len(eval_indices)} samples")

y_true = primary_dataset.stress_normalizer.denormalize(primary_testset['t'])
mask = primary_testset['mask']

# ================== Get predictions from all three models ==================
print("Loading nonlinear PRNN...")
y_nonlin, _ = load_model_and_predict(nonlin_folder, nonlin_run, eval_indices)

print("Loading linear PRNN (32 samples)...")
y_linear, _ = load_model_and_predict(linear_folder, linear_run, eval_indices)

print("Loading linear PRNN (512 samples)...")
y_linear512, _ = load_model_and_predict(linear512_folder, linear512_run, eval_indices)

print("Loading NN...")
y_nn, _ = load_model_and_predict(nn_folder, nn_run, eval_indices)

print("Loading NN 32...")
y_nn32, _ = load_model_and_predict(nn32_folder, nn32_run, eval_indices)

# ================== Compute per-sample errors for all models ==================
def sample_errors(y_pred):
    return np.array(jnp.sum(jnp.abs(y_pred - y_true) * mask, axis=(1, 2)) / jnp.sum(mask, axis=(1, 2)))

errors_nonlin   = sample_errors(y_nonlin)
errors_linear   = sample_errors(y_linear)
errors_linear512 = sample_errors(y_linear512)
errors_nn       = sample_errors(y_nn)
errors_nn32       = sample_errors(y_nn32)

n_test = len(eval_indices)
median_idx = int(np.argsort(errors_nonlin)[n_test // 2])
print(f"Median-error sample index: {median_idx}  (nonlin error={errors_nonlin[median_idx]:.6f})")

def rank_label(name, errors, idx):
    """'Name, 0.0783, 413/512' — rank = number of samples with strictly lower error."""
    rank = int(np.sum(errors < errors[idx]))
    return f'{name}, {errors[idx]:.4f}, {rank}/{n_test}'

# ================== Extract sample data ==================
E_mat = primary_testset['x']
y_true_mat   = tensor_to_matrix(y_true)
y_nonlin_mat = tensor_to_matrix(y_nonlin)
y_linear_mat     = tensor_to_matrix(y_linear)
y_linear512_mat  = tensor_to_matrix(y_linear512)
y_nn_mat     = tensor_to_matrix(y_nn)
y_nn32_mat     = tensor_to_matrix(y_nn32)

mat_params = primary_settings['mat_parameters']
mp = np.array(primary_dataset.M[eval_indices[median_idx]])
mu_val = mp[mat_params.index('mu')]
vf_val = mp[mat_params.index('vfrac')]
r_val  = mp[mat_params.index('ratio')]
true_label = rf'True ($\mu$={mu_val:.2f}, $V_f$={vf_val:.2f}, $r$={r_val:.2f})'

E_s      = np.array(E_mat[median_idx])
true_s   = np.array(y_true_mat[median_idx])
nonlin_s = np.array(y_nonlin_mat[median_idx])
linear_s    = np.array(y_linear_mat[median_idx])
linear512_s = np.array(y_linear512_mat[median_idx])
nn_s     = np.array(y_nn_mat[median_idx])
nn32_s     = np.array(y_nn32_mat[median_idx])

# ================== Plot ==================
components = [(0, 0, 'xx'), (1, 1, 'yy'), (0, 1, 'xy')]

fig, axes = plt.subplots(1, 3, figsize=(8, 2.2), sharey=False)

model_styles = [
    (nonlin_s,    rank_label(f'Nonlinear PRNN ({nonlin_samples})', errors_nonlin,    median_idx), '--', colours[0]),
    (linear_s,    rank_label(f'Linear PRNN ({linear_samples})',    errors_linear,    median_idx), ':',  colours[1]),
    (linear512_s, rank_label(f'Linear PRNN ({linear512_samples})', errors_linear512, median_idx), '--', colours[1]),
    (nn32_s,      rank_label(f'NN ({nn32_samples})',            errors_nn32,      median_idx), ':',  colours[2]),
    (nn_s,        rank_label(f'NN ({nn_samples})',                 errors_nn,        median_idx), '--', colours[2]),
]

for ax, (i, j, label) in zip(axes, components):
    e_comp = E_s[:, i, j] - E_s[0, i, j]
    true_comp = true_s[:, i, j] - true_s[0, i, j]

    ax.plot(e_comp, true_comp, 'k-', linewidth=1.2, label=true_label, zorder=5)
    for pred_s, lname, ls, c in model_styles:
        pred_comp = pred_s[:, i, j] - pred_s[0, i, j]
        ax.plot(e_comp, pred_comp, ls, color=c, linewidth=1.0, label=lname)

    ax.set_xlabel(rf'$E_{{{label}}}$')
    ax.set_ylabel(rf'$S_{{{label}}}$')
    ax.minorticks_off()

for i, ax in enumerate(axes):
    if y_lims is not None:
        ax.set_ylim(y_lims[i])

# Single shared legend below the figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8,
           bbox_to_anchor=(.5, .9))

plt.tight_layout()
plt.savefig(f"{plot_folder}median_curves_{eval_set}.pdf", bbox_inches='tight', format='pdf')
plt.close()
print(f"Saved to {plot_folder}median_curves_{eval_set}.pdf")
