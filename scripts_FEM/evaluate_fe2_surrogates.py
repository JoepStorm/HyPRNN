"""Evaluate surrogate stress predictions on the FE^2 strain trajectories
saved by validate_bending.py, to compare against the FE^2 validation/test
loss reported during training.
"""
import os
import sys
os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp
from material_params import FUNGI_LAMBDA

from scripts_materials.prnn_material import PRNNMaterial
from validate_bending import MODELS, MODEL_LABELS
from scripts_surrogates.plot_LD_data import plot_PK2_curves


def _vec_to_mat(v):
    """(..., 4) with component ordering [11, 22, 12, 21] -> (..., 2, 2)."""
    out = np.zeros(v.shape[:-1] + (2, 2))
    out[..., 0, 0] = v[..., 0]
    out[..., 1, 1] = v[..., 1]
    out[..., 0, 1] = v[..., 2]
    out[..., 1, 0] = v[..., 3]
    return out


def _compute_E_PK2(F_vec, PK1_vec):
    """Green-Lagrange E and PK2 stress from flat F, PK1. Returns (..., 2, 2)."""
    Fm = _vec_to_mat(F_vec)
    Pm = _vec_to_mat(PK1_vec)
    I = np.eye(2)
    E = 0.5 * (np.einsum('...ji,...jk->...ik', Fm, Fm) - I)
    Finv = np.linalg.inv(Fm)
    PK2 = np.einsum('...ij,...jk->...ik', Finv, Pm)
    return E, PK2


def _plot_E_PK2_paths(F_true, PK1_true, results, output_folder, max_paths=30):
    """Plot E-PK2 trajectories: true-only, then one figure per surrogate."""
    E_true, PK2_true = _compute_E_PK2(F_true, PK1_true)
    # (n_steps, n_qp, 2, 2) -> (n_qp, n_steps, 2, 2) so QPs are the batch axis.
    E_true = np.transpose(E_true, (1, 0, 2, 3))
    PK2_true = np.transpose(PK2_true, (1, 0, 2, 3))

    n_qp = E_true.shape[0]
    idx = np.linspace(0, n_qp - 1, min(max_paths, n_qp)).astype(int)
    E_sub = E_true[idx]
    PK2_sub = PK2_true[idx]

    plot_PK2_curves(E_true=E_sub, PK2_true=PK2_sub,
                    savename=os.path.join(output_folder, 'E_PK2_true'), png=True)

    for key, r in results.items():
        _, PK2_pred = _compute_E_PK2(F_true, r['PK1_pred'])
        PK2_pred = np.transpose(PK2_pred, (1, 0, 2, 3))[idx]
        plot_PK2_curves(E_true=E_sub, PK2_true=PK2_sub, PK2_pred=PK2_pred,
                        savename=os.path.join(output_folder, f'E_PK2_{key}'), png=True)
    print(f"Saved E-PK2 path plots to {output_folder}")


def evaluate(npz_path, output_folder=None):
    data = np.load(npz_path)
    F_true = data['fe2_F']        # (n_steps, n_qp, 4)
    PK1_true = data['fe2_PK1']    # (n_steps, n_qp, 4)
    n_steps, n_qp, _ = F_true.shape

    micro_variables = {
        'vfrac':  jnp.asarray(data['micro_vfrac']),
        'theta':  jnp.asarray(data['micro_theta']),
        'ratio':  jnp.asarray(data['micro_ratio']),
        'mu':     jnp.asarray(data['micro_mu']),
        'lambda': jnp.asarray(FUNGI_LAMBDA * np.ones(n_qp)),
    }

    F_flat = F_true.reshape(-1, 4)
    PK1_flat = PK1_true.reshape(-1, 4)
    stress_norm = np.sqrt(np.mean(PK1_flat ** 2))

    print(f"Loaded {npz_path}: {n_steps} steps x {n_qp} qp, "
          f"RMS(PK1_true) = {stress_norm:.4e}")

    surrogate_keys = [k for k, cfg in MODELS.items() if cfg.get('use_surrogate')]
    results = {}

    for key in surrogate_keys:
        cfg = MODELS[key]
        print(f"\n--- {key} ({MODEL_LABELS.get(key, key)}) ---")
        loc = cfg['prnn_model_loc']
        mat = PRNNMaterial(f"{loc}_settings", f"{loc}_normparams", loc,
                           micro_variables)

        PK1_pred = np.zeros_like(PK1_true)
        for s in range(n_steps):
            F_s = jnp.asarray(F_true[s])
            stress, _, _ = mat.constitutive_update_vectorized(F_s, {})
            PK1_pred[s] = np.asarray(stress)

        diff = PK1_pred.reshape(-1, 4) - PK1_flat
        me = float(np.mean(np.abs(diff)))
        mse = float(np.mean(diff ** 2))
        rmse = float(np.sqrt(mse))
        rel_rmse = rmse / stress_norm
        comp_rmse = np.sqrt(np.mean(diff ** 2, axis=0))
        print(f"  ME (L1)     = {me:.4e}")
        print(f"  MSE         = {mse:.4e}")
        print(f"  RMSE        = {rmse:.4e}")
        print(f"  rel. RMSE   = {rel_rmse:.4e}")
        print(f"  per-comp RMSE (P11,P22,P12,P21) = "
              + ", ".join(f"{v:.3e}" for v in comp_rmse))
        results[key] = {
            'PK1_pred': PK1_pred, 'mse': mse, 'rmse': rmse,
            'rel_rmse': rel_rmse, 'comp_rmse': comp_rmse,
        }

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        out = {'F_true': F_true, 'PK1_true': PK1_true}
        for k, r in results.items():
            out[f'{k}_PK1_pred'] = r['PK1_pred']
            out[f'{k}_mse'] = r['mse']
            out[f'{k}_rel_rmse'] = r['rel_rmse']
        np.savez(f"{output_folder}/surrogate_eval.npz", **out)
        print(f"\nSaved predictions to {output_folder}/surrogate_eval.npz")
        _plot_E_PK2_paths(F_true, PK1_true, results, output_folder)

    return results


if __name__ == "__main__":
    folder = "../results/bending_model_comparison/v4/"
    evaluate(f"{folder}/comparison_data.npz", output_folder=folder)
