"""Visualize uniaxial dataset (create_uniaxial_data.py) with PRNN predictions.

Subplots left-to-right: mu, vfrac, ratio variation.
Tension + compression shown; solid = RVE, dashed = PRNN nonlin.
RVE mesh thumbnails are shown above each subplot.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
plt.style.use(['science', 'bright'])
from matplotlib import rc
rc('text', usetex=True)

jax.config.update('jax_platform_name', 'cpu')

from data_utils import LDDataset, load_settings, Config, tensor_to_matrix
from LDprnn_shared_hyper import create_shared_hyper_prnn_model
from trainer import Trainer

# ============================================================
# PRNN model settings
# ============================================================
training_samples_array = [2, 4, 8, 12, 16, 24, 32, 64, 96, 128, 192, 256, 512]
nonlin_folder = '../trained_models/train_vary_all_v2/prnn_nonlin_1L8_12m_sigmoid/'
nonlin_samples = 512


def median_run(folder, n_samples):
    """Return the run name whose test loss is closest to the median."""
    losses = np.loadtxt(f"{folder}test_losses.txt", delimiter=',')
    row = training_samples_array.index(n_samples)
    row_losses = losses[row]
    median_run_idx = int(np.argsort(row_losses)[len(row_losses) // 2])
    print(f"  {folder}: samples={n_samples}, run{median_run_idx} (loss={row_losses[median_run_idx]:.4f})")
    return f"samples{n_samples}_run{median_run_idx}"


run = median_run(nonlin_folder, nonlin_samples)
settings = load_settings(f"{nonlin_folder}{run}_settings")
key = jax.random.PRNGKey(settings['seed'])

# Training dataset (normalizers only)
train_dataset = LDDataset(
    settings['data_path'], seq_length=settings['seq_length'],
    num_samples=settings['num_samples'],
    mat_file=settings['matdata_path'], mat_features=settings['mat_parameters'],
    norm_stresses=settings['norm_stresses'], norm_matparams=settings['norm_matparams'],
)

# ============================================================
# Load uniaxial dataset
# ============================================================
data_base = '../data/uniaxial_v2/uniaxial'
seq_length = 50

dataset = LDDataset(
    data_base, seq_length,
    mat_file=data_base + '_matparam.data',
    mat_features=settings['mat_parameters'],
)
data = dataset.get_all_batches()

# Sample layout (from create_uniaxial_data.py):
#   tension:     0=median, 2=nfib_min, 4=nfib_max, 6=ratio_min, 8=ratio_max, 10=mu_min, 12=mu_max
#   compression: tension_idx + 1

# ============================================================
# PRNN model + predictions
# ============================================================
model, params_init, material = create_shared_hyper_prnn_model(
    random_key=key,
    n_micro_raw=len(settings['mat_parameters']),
    shared_micro_features=settings['shared_micro_features'],
    n_matpts=settings['mat_points'],
    encoder_type=settings['encoder_type'],
    shared_hidden_mult=settings.get('hyper_hidden_mult', 2),
    shared_output_mult=settings.get('hyper_output_mult', 4),
    hidden_dim=settings.get('hidden_dim', 8),
    stress_normalizer=train_dataset.stress_normalizer,
    mat_m_feats=settings['mat_micro_features'],
    hyper_hidden_sizes=settings.get('hyper_hidden_sizes', None),
    hyper_activation=settings.get('hyper_activation', 'sigmoid'),
    encoder_n_layers=settings.get('encoder_n_layers', 3),
    encoder_activation=settings.get('encoder_activation', 'softplus'),
)
lr_config = Config(
    warmup_epochs=settings['warmup_epochs'],
    schedule_steps=settings['lr_schedule_steps'],
    steps_per_epoch=settings['train_samples'] / settings['train_batch_size'],
    base_learning_rate=settings['base_lr'],
    min_lr_factor=settings['min_lr_factor'],
)
trainer = Trainer(model, params_init, material=material, lr_config=lr_config, random_key=key, out_norm_factor=1)
params = trainer.load(f"{nonlin_folder}{run}")

M_norm = train_dataset.M_normalizer.normalize(dataset.M)
pred_norm = model.apply({'params': params['params']}, data['x'], material, micro_params=M_norm)
PK2_pred = tensor_to_matrix(train_dataset.stress_normalizer.denormalize(pred_norm), symmetric=True)

PK2_gt = data['PK2_eq_unnorm']   # (N, T, 2, 2)
E_data  = data['x']              # (N, T, 2, 2)

# ============================================================
# Mesh loading
# ============================================================
mesh_folder = '../meshes/vfrac_ratio_big/'

sample_to_mesh = {
    0:  f'{mesh_folder}nfib30_0.0000_1.75/rve_0.msh',
    2:  f'{mesh_folder}nfib10_0.0000_1.75/rve_0.msh',
    4:  f'{mesh_folder}nfib70_0.0000_1.75/rve_0.msh',
    6:  f'{mesh_folder}nfib30_0.0000_1.00/rve_0.msh',
    8:  f'{mesh_folder}nfib30_0.0000_2.50/rve_0.msh',
    10: f'{mesh_folder}nfib30_0.0000_1.75/rve_0.msh',   # mu only → same geometry
    12: f'{mesh_folder}nfib30_0.0000_1.75/rve_0.msh',
}

_mesh_cache = {}


def load_gmsh_mesh(msh_file):
    """Load a GMSH .msh file; return (node_coords, tri_elem_nodes, phys_tags)."""
    import gmsh
    gmsh.initialize()
    gmsh.open(msh_file)
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coords = node_coords.reshape(-1, 3)[:, :2]
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    triangles, phys_tags = [], []
    for dim, phys_tag in gmsh.model.getPhysicalGroups(dim=2):
        for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag):
            elem_types, _, node_tags_list = gmsh.model.mesh.getElements(dim, entity)
            for etype, ntags in zip(elem_types, node_tags_list):
                if int(etype) == 2:
                    for e in ntags.reshape(-1, 3):
                        triangles.append([tag_to_idx[int(n)] for n in e])
                        phys_tags.append(phys_tag)
    gmsh.finalize()
    return coords, np.array(triangles), np.array(phys_tags)


def get_mesh(msh_file):
    if msh_file not in _mesh_cache:
        try:
            _mesh_cache[msh_file] = load_gmsh_mesh(msh_file)
        except Exception as exc:
            print(f"  Warning: could not load mesh {msh_file}: {exc}")
            _mesh_cache[msh_file] = None
    return _mesh_cache[msh_file]


def draw_mesh_in_ax(mesh_ax, msh_file, label=None, label_color='k', matrix_color='#4CAF50'):
    """Draw RVE mesh into mesh_ax; optionally add a value label as axis title."""
    mesh = get_mesh(msh_file)
    if mesh is None:
        mesh_ax.axis('off')
        return
    coords, elems, tags = mesh
    material_colors = {1: matrix_color, 2: '#8B4513'}
    pc = PolyCollection(coords[elems],
                        facecolors=[material_colors.get(int(t), '#888888') for t in tags],
                        edgecolors='none')
    mesh_ax.add_collection(pc)
    mesh_ax.autoscale()
    mesh_ax.set_aspect('equal')
    mesh_ax.axis('off')
    if label is not None:
        mesh_ax.set_title(label, fontsize=10, color=label_color, pad=2)


# ============================================================
# Colors: 3 discrete values sampled from each colormap
# ============================================================
def discrete_colors(cmap_name, n=3, vmin=0.15, vmax=0.85):
    cmap = plt.get_cmap(cmap_name)
    return [cmap(vmin + (vmax - vmin) * i / (n - 1)) for i in range(n)]

# mu lines/labels use the same greens as the mesh matrix colors
mu_colors = ['#A5D6A7', '#4CAF50', '#1B5E20']
vf_colors = discrete_colors('viridis')
r_colors  = discrete_colors('plasma')

# ============================================================
# Figure layout: top row = mesh thumbnails, bottom row = plots
# ============================================================
fig = plt.figure(figsize=(8, 3.0))

gs = GridSpec(2, 3, figure=fig,
              height_ratios=[1.0, 2.0],
              hspace=0.08,
              wspace=0.2,
              left=0.09, right=0.98, top=0.97, bottom=0.13)

# Bottom row: stress-strain subplots
ax_plots = [fig.add_subplot(gs[1, col]) for col in range(3)]

# Top row mesh areas
# mu (col 0): 3 meshes, each with a different matrix color to indicate mu variation
gs_mu_top = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 0], wspace=0.15)
ax_mu_mesh = [fig.add_subplot(gs_mu_top[0, j]) for j in range(3)]

# vfrac (col 1): 3 meshes spread left (low) to right (high)
gs_vf_top = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 1], wspace=0.15)
ax_vf_mesh = [fig.add_subplot(gs_vf_top[0, j]) for j in range(3)]

# ratio (col 2): 3 meshes spread left (low) to right (high)
gs_r_top = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 2], wspace=0.15)
ax_r_mesh = [fig.add_subplot(gs_r_top[0, j]) for j in range(3)]

# ============================================================
# Draw mesh thumbnails
# ============================================================
mu_idx_in_params  = settings['mat_parameters'].index('mu')
vf_idx_in_params  = settings['mat_parameters'].index('vfrac')
r_idx_in_params   = settings['mat_parameters'].index('ratio')

# mu: 3 identical geometry meshes, matrix color varies light→default→dark green
mu_matrix_colors = ['#A5D6A7', '#4CAF50', '#1B5E20']  # light / default / dark green
for mesh_ax, t_idx, label_color, matrix_color in zip(
        ax_mu_mesh, [10, 0, 12], mu_colors, mu_matrix_colors):
    mu_val = float(dataset.M[t_idx, mu_idx_in_params])
    draw_mesh_in_ax(mesh_ax, sample_to_mesh[t_idx], f'$\\mu={mu_val:.2f}$',
                    label_color, matrix_color)

# vfrac: 3 meshes with Vf labels
for j, (mesh_ax, t_idx, color) in enumerate(zip(ax_vf_mesh, [2, 0, 4], vf_colors)):
    vf_val = float(dataset.M[t_idx, vf_idx_in_params])
    draw_mesh_in_ax(mesh_ax, sample_to_mesh[t_idx], f'$V_f={vf_val:.2f}$', color)

# ratio: 3 meshes with ratio labels
for j, (mesh_ax, t_idx, color) in enumerate(zip(ax_r_mesh, [6, 0, 8], r_colors)):
    r_val = float(dataset.M[t_idx, r_idx_in_params])
    draw_mesh_in_ax(mesh_ax, sample_to_mesh[t_idx], f'$r={r_val:.2f}$', color)

# ============================================================
# Stress-strain plots
# ============================================================
subplot_configs = [
    ('mu',    [10, 0, 12], mu_colors),
    ('vfrac', [2,  0, 4],  vf_colors),
    ('ratio', [6,  0, 8],  r_colors),
]

for col, (subplot_ax, (param_name, tension_indices, colors)) in enumerate(
        zip(ax_plots, subplot_configs)):

    subplot_ax.axhline(0, color='lightgray', linewidth=0.6, zorder=0)
    subplot_ax.axvline(0, color='lightgray', linewidth=0.6, zorder=0)

    for i, t_idx in enumerate(tension_indices):
        c_idx = t_idx + 1
        color = colors[i]

        # Tension
        E_t = np.array(E_data[t_idx, :, 0, 0])
        subplot_ax.plot(E_t, np.array(PK2_gt[t_idx,   :, 0, 0]), '-',  color=color)
        # subplot_ax.plot(E_t, np.array(PK2_pred[t_idx, :, 0, 0]), '--', color=color)

        # Compression
        E_c = np.array(E_data[c_idx, :, 0, 0])
        subplot_ax.plot(E_c, np.array(PK2_gt[c_idx,   :, 0, 0]), '-',  color=color)
#         subplot_ax.plot(E_c, np.array(PK2_pred[c_idx, :, 0, 0]), '--', color=color)

    subplot_ax.set_xlabel(r'$E_{xx}$', labelpad=1)
    subplot_ax.set_ylabel(r'$S_{xx}$')
    subplot_ax.minorticks_off()

# RVE / PRNN nonlin legend on first subplot only
# legend_handles = [
#     Line2D([0], [0], linestyle='-',  color='k', linewidth=1.0, label='RVE'),
#     Line2D([0], [0], linestyle='--', color='k', linewidth=1.0, label='PRNN nonlin'),
# ]
# ax_plots[0].legend(handles=legend_handles, fontsize=7, loc='lower right')

# ============================================================
# Save
# ============================================================
os.makedirs(nonlin_folder + 'statistics/', exist_ok=True)
# save_path = nonlin_folder + 'statistics/uniaxial_pred'
save_path = nonlin_folder + 'statistics/uniaxial_pred_noprnn'
plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved to {save_path}.pdf")
