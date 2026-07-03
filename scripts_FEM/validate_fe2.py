"""
Diagnostic: compare RVE (FE²) vs PRNN stress response on identical load paths.

Generates an RVE mesh, runs the full RVE solve along a load path, runs the
PRNN surrogate on the same path, and optionally overlays original training data.
Plots F-PK1 curves for all 4 tensor components.
"""
import os
import sys
os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use(['science', 'bright'])
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
rc('text', usetex=True)

from mpi4py import MPI
from scripts_materials.rve_material import RVEMaterial
from scripts_materials.rve_mesher import RVEMeshConfig
from scripts_materials.prnn_material import PRNNMaterial, complete_update_pk2, _convert_to_float64
from scripts_surrogates.data_utils import LDDataset, load_settings, create_Q_matrix, matrix_to_tensor, tensor_to_matrix, load_params
from scripts_surrogates.HyPRNN import HyPRNN
from scripts_surrogates.StandardNN import StandardNN
from material_params import WOOD_E, WOOD_NU, FUNGI_MU, FUNGI_LAMBDA


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_training_sample(data_path, matparam_path, sample_idx):
    """Load a single load path from the training dataset.

    Returns F_path (T,2,2), PK1_path (T,2,2), and a dict of material params.
    """
    F_all = np.load(f"{data_path}_F.npy")       # (N, T, 2, 2)
    PK1_all = np.load(f"{data_path}_PK1.npy")   # (N, T, 2, 2)

    if F_all.ndim == 3:
        F_all = F_all[np.newaxis]
        PK1_all = PK1_all[np.newaxis]

    F_path = F_all[sample_idx]
    PK1_path = PK1_all[sample_idx]

    # Read matparam file for this sample
    import pandas as pd
    df = pd.read_csv(matparam_path, sep=' ')
    row = df.iloc[sample_idx]

    micro = {
        'nfib': int(row['nfib']),
        'vfrac': float(row['vfrac']),
        'theta': float(row['angle']),
        'ratio': float(row['ratio']),
        'mu': float(row['mu']),
        'lambda': float(row['lambda']),
    }
    return F_path, PK1_path, micro


def generate_linear_load_path(F_target, n_steps=50):
    """Linear ramp from I to F_target."""
    I = np.eye(2)
    return np.array([I + (t + 1) / n_steps * (F_target - I) for t in range(n_steps)])


def _flat_to_matrix_batch(X):
    """(T,4) with [11,22,12,21] -> (T,2,2)"""
    T = X.shape[0]
    M = np.zeros((T, 2, 2))
    M[:, 0, 0] = X[:, 0]
    M[:, 1, 1] = X[:, 1]
    M[:, 0, 1] = X[:, 2]
    M[:, 1, 0] = X[:, 3]
    return M


def load_tension_results(npz_path, f_model_key, ip_idx=0):
    """Load matched F-PK1 path from validate_tension output for a specific model and IP.

    Only returns F and PK1 from the same model, so the pairs are consistent.

    Returns F_path (T,2,2), PK1_ref (T,2,2), micro dict.
    """
    data = np.load(npz_path, allow_pickle=True)
    micro = data['micro_params'].item()

    F_flat = data[f'{f_model_key}_F'][:, ip_idx, :]      # (T, 4)
    PK1_flat = data[f'{f_model_key}_PK1'][:, ip_idx, :]  # (T, 4)

    F_path = _flat_to_matrix_batch(F_flat)
    PK1_ref = _flat_to_matrix_batch(PK1_flat)

    return F_path, PK1_ref, micro


def _mesh_geo_file(geo_file):
    """Mesh a .geo file with gmsh and return the .msh path and domain_size."""
    import re
    import subprocess

    # Extract domain_size from Rectangle definition in .geo
    with open(geo_file) as f:
        geo_text = f.read()
    match = re.search(r'Rectangle\(\d+\)\s*=\s*\{[^,]+,[^,]+,[^,]+,\s*([^,]+)', geo_text)
    if not match:
        raise ValueError(f"Could not extract domain size from {geo_file}")
    domain_size = float(match.group(1))

    # Mesh the .geo file
    msh_file = geo_file.replace('.geo', '.msh')
    subprocess.run(
        ['/home/joep/Programs/gmsh-4.13.1-Linux64/bin/gmsh2', geo_file, '-format', 'msh22', '-2'],
        capture_output=True, check=True
    )

    # Remove $PhysicalNames section (same as create_rve.py)
    with open(msh_file, 'r') as f:
        lines = f.readlines()
    with open(msh_file, 'w') as f:
        skip = False
        for line in lines:
            if '$PhysicalNames' in line:
                skip = True
            elif '$EndPhysicalNames' in line:
                skip = False
                continue
            if not skip:
                f.write(line)

    return msh_file, domain_size


def run_rve(F_path, micro, meshsize=0.0125, seed=0, geo_file=None):
    """Run the full RVE solve along F_path. Returns PK1 (T,2,2).

    If geo_file is provided, uses that existing .geo file instead of generating a new mesh.
    """
    lambda_val = micro['lambda']
    mu_val = micro['mu']
    E_w, nu_w = WOOD_E, WOOD_NU

    material_props = {
        'fungi': {'mu': mu_val, 'lambda_': lambda_val, 'tag': 1},
        'wood': {'E': E_w, 'nu': nu_w, 'tag': 2},
    }

    if geo_file is not None:
        print(f"  Using existing .geo file: {geo_file}")
        msh_file, domain_size = _mesh_geo_file(geo_file)
    else:
        config = RVEMeshConfig(
            vfrac=micro['vfrac'],
            aspect_ratio=micro['ratio'],
            angle=micro['theta'],
            seed=seed,
            meshsize=meshsize,
        )

        savefolder = '../results/validate_fe2_meshes'
        # Clean up previous mesh files
        import glob
        if os.path.exists(savefolder):
            for f in glob.glob(f"{savefolder}/*.geo") + glob.glob(f"{savefolder}/*.msh"):
                os.remove(f)
        os.makedirs(savefolder, exist_ok=True)
        msh_file, domain_size = config.create_mesh_file(savefolder=savefolder, mesh_id=seed)

    rve = RVEMaterial(
        material_properties=material_props,
        mesh_file=msh_file,
        domain_size=domain_size,
        comm=MPI.COMM_SELF,
    )

    T = F_path.shape[0]
    PK1_rve = np.zeros((T, 2, 2))

    for t in range(T):
        P, _, converged = rve.update_stress(F_path[t], monitor_base_solve=False)
        if not converged:
            print(f"  RVE did not converge at step {t}, F={F_path[t]}")
        PK1_rve[t] = P

    return PK1_rve


def run_prnn(F_path, micro, prnn_model_loc):
    """Run the PRNN surrogate along F_path. Returns PK1 (T,2,2)."""
    settings = load_settings(f"{prnn_model_loc}_settings")

    proxyData = LDDataset.__new__(LDDataset)
    proxyData.loadDataparams(f"{prnn_model_loc}_normparams")
    proxyData.stress_normalizer = _convert_to_float64(proxyData.stress_normalizer)
    proxyData.M_normalizer = _convert_to_float64(proxyData.M_normalizer)

    params = _convert_to_float64(load_params(prnn_model_loc))

    model_type = settings.get('model_type', 'hyprnn')
    if model_type == 'nn':
        model = StandardNN(
            n_outputs=3,
            hidden_sizes=tuple(settings['nn_hidden_sizes']),
            activation=settings.get('nn_activation', 'sigmoid'),
            use_bias=settings.get('nn_bias', True),
        )
    else:
        use_multiple_materials = isinstance(settings['mat_points'], list)
        n_matpts = tuple(settings['mat_points']) if use_multiple_materials else settings['mat_points']
        model = HyPRNN(
            n_matpts=n_matpts,
            n_outputs=3,
            encoder_type=settings['encoder_type'],
            stress_normalizer=proxyData.stress_normalizer,
            material_micro_features=tuple(settings['mat_micro_features']),
            shared_micro_features=tuple(settings.get('shared_micro_features', [])),
            hidden_dim=settings.get('hidden_dim', 8),
            w_max=settings.get('w_max', 1.25),
            hyper_hidden_sizes=tuple(settings['hyper_hidden_sizes']) if settings.get('hyper_hidden_sizes') else None,
            hyper_activation=settings.get('hyper_activation', 'sigmoid'),
            encoder_n_layers=settings.get('encoder_n_layers', 3),
            encoder_activation=settings.get('encoder_activation', 'softplus'),
            use_multiple_materials=use_multiple_materials,
        )

    import scripts_materials.neohooke as neohooke

    # Build Q matrix
    theta = micro['theta']
    Q = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                    [jnp.sin(theta),  jnp.cos(theta)]], dtype=jnp.float64)

    # Build micro_params vector
    mat_param_keys = settings['mat_parameters']
    micro_vals = {
        'mu': micro['mu'],
        'lambda': micro['lambda'],
        'vfrac': micro['vfrac'],
        'ratio': micro['ratio'],
    }
    if isinstance(mat_param_keys, list):
        micro_params_raw = jnp.array([micro_vals[k] for k in mat_param_keys], dtype=jnp.float64)
    else:
        micro_params_raw = jnp.array([micro_vals[mat_param_keys]], dtype=jnp.float64)

    micro_params_norm = proxyData.M_normalizer.normalize(micro_params_raw[None, :])[0]

    material = neohooke.create_material(lambda_=micro['lambda'], mu_=micro['mu'])

    T = F_path.shape[0]
    PK1_prnn = np.zeros((T, 2, 2))

    for t in range(T):
        F_vec = matrix_to_tensor(jnp.array(F_path[t], dtype=jnp.float64))
        pk1_vec = complete_update_pk2(
            model, F_vec, material, proxyData, params, Q, micro_params_norm
        )
        PK1_prnn[t] = np.array(tensor_to_matrix(pk1_vec))

    return PK1_prnn


def to_flat(T_2x2):
    """(T,2,2) -> (T,4) in [11,22,12,21] ordering."""
    return np.stack([T_2x2[:, 0, 0], T_2x2[:, 1, 1],
                     T_2x2[:, 0, 1], T_2x2[:, 1, 0]], axis=-1)


def plot_comparison(F_path, results, output_path, micro=None):
    """Plot F vs PK1 for each component. results is a dict {label: PK1 (T,2,2)}."""
    F_flat = to_flat(F_path)
    labels_F = [r'$F_{11}$', r'$F_{22}$', r'$F_{12}$', r'$F_{21}$']
    labels_P = [r'$P_{11}$', r'$P_{22}$', r'$P_{12}$', r'$P_{21}$']
    linestyles = ['-', '--', '-.', ':']

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for comp, ax in enumerate(axes.ravel()):
        for i, (label, PK1) in enumerate(results.items()):
            P_flat = to_flat(PK1)
            ax.plot(F_flat[:, comp], P_flat[:, comp],
                    linestyles[i % len(linestyles)],
                    color=colours[i % len(colours)],
                    label=label, linewidth=1.5)
        ax.set_xlabel(labels_F[comp])
        ax.set_ylabel(labels_P[comp])
        ax.legend(fontsize=7)

    title = "FE2 vs PRNN comparison"
    if micro:
        title += f"\n$V_f$={micro['vfrac']:.2f}, $r$={micro['ratio']:.2f}, " \
                 fr"$\theta$={np.degrees(micro['theta']):.1f}, $\mu$={micro['mu']:.3f}"
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
    plt.savefig(f"{output_path}.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved plot to {output_path}.pdf")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    output_folder = "../results/validate_fe2"
    os.makedirs(output_folder, exist_ok=True)

    # ── Paths (edit these) ───────────────────────────────────────────────
    prnn_model_loc = '../trained_models/train_vfrac_ratio/prnn_lin_3L_6m/samples512_run0'

    # ── Source selection (uncomment ONE) ─────────────────────────────────

    # Option 1: Load path from validate_tension output
    tension_npz = '../results/validate_tension/tension_results.npz'
    tension_model_key = 'fe2'  # which model's F path to replay ('prnn' or 'fe2')
    tension_ip = 0              # which integration point

    # Option 2: Load path from training data
    training_data_path = None  # '../data/vary_vfrac_ratio/vary_vfrac_ratio/mixed_t50_merged'
    training_matparam_path = f'{training_data_path}_matparam.data' if training_data_path else None
    sample_idx = 0

    # Option 3: Manual F target (used if both above are None)
    micro = {
        'nfib': 50,
        'vfrac': 0.3,
        'theta': 0.0,
        'ratio': 2.0,
        'mu': FUNGI_MU,
        'lambda': FUNGI_LAMBDA,
    }

    # ── RVE mesh options ───────────────────────────────────────────────
    # Set geo_file to use an existing .geo file from a previous FE² run instead of generating a new mesh
    geo_file = '../results/validate_fe2/rve_0.geo'  # None to generate fresh mesh

    # ── Load path ────────────────────────────────────────────────────────
    rve_meshsize = 0.0125
    rve_seed = 0
    PK1_ref = None  # reference PK1 from tension/training run
    suffix = ''

    if tension_npz is not None and os.path.exists(tension_npz):
        print(f"Loading tension results: {tension_npz}, model='{tension_model_key}', IP={tension_ip}")
        F_path, PK1_ref, micro = load_tension_results(tension_npz, tension_model_key, tension_ip)
        suffix = f'_tension_{tension_model_key}_ip{tension_ip}'
        print(f"  micro params: {micro}")
        print(f"  F_path shape: {F_path.shape}, F11 range: [{F_path[:,0,0].min():.6f}, {F_path[:,0,0].max():.6f}]")
    elif training_data_path is not None:
        print(f"Loading training sample {sample_idx} from {training_data_path}")
        F_path, PK1_ref, micro = load_training_sample(
            training_data_path, training_matparam_path, sample_idx
        )
        suffix = f'_train_sample{sample_idx}'
        print(f"  micro params: {micro}")
        print(f"  F_path shape: {F_path.shape}, range: [{F_path.min():.4f}, {F_path.max():.4f}]")
    else:
        F_target = np.eye(2) + np.array([[.025, .0003], [.0008, -0.008]])
        F_path = generate_linear_load_path(F_target, n_steps=10)
        suffix = '_manual'

    # ── Run RVE ──────────────────────────────────────────────────────────
    print("Running RVE solve...")
    PK1_rve = run_rve(F_path, micro, meshsize=rve_meshsize, seed=rve_seed, geo_file=geo_file)

    # ── Run PRNN ─────────────────────────────────────────────────────────
    print("Running PRNN inference...")
    PK1_prnn = run_prnn(F_path, micro, prnn_model_loc)

    # ── Plot ─────────────────────────────────────────────────────────────
    results = {}
    if PK1_ref is not None:
        ref_label = f'{tension_model_key} (tension)' if tension_npz else 'Training data'
        results[ref_label] = PK1_ref
    results['RVE (fresh)'] = PK1_rve
    results['PRNN (fresh)'] = PK1_prnn

    plot_comparison(F_path, results, f"{output_folder}/comparison{suffix}", micro)

    # Print max relative difference
    mask = np.abs(PK1_rve) > 1e-10
    if mask.any():
        rel_diff = np.abs(PK1_prnn[mask] - PK1_rve[mask]) / np.abs(PK1_rve[mask])
        print(f"Max relative |PRNN - RVE| / |RVE|: {rel_diff.max():.4f}")
        print(f"Mean relative |PRNN - RVE| / |RVE|: {rel_diff.mean():.4f}")
