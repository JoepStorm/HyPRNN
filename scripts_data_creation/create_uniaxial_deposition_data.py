"""Run uniaxial RVE simulations on YADE-deposited 2D wet meshes.

For each (mix, seed, loading direction) combination, runs a uniaxial
tension + compression simulation using the periodic GMSH meshes generated
in the yade/ workflow.

The single varying material parameter is 'mix' (fraction of small chips):
  0.0 = all large, 0.25, 0.50, 0.75, 1.0 = all small.

Usage:
    cd scripts_surrogates
    python create_uniaxial_deposition_data.py
    python create_uniaxial_deposition_data.py --shrink_factor 1.0 --mesh_dir ../yade/generated_meshes_2D_wet
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts_materials.rve_material import RVEMaterial
from material_params import WOOD_E, WOOD_NU, FUNGI_MU, FUNGI_LAMBDA

# ── Fixed material properties ─────────────────────────────────────────────────
MATERIAL_PROPS = {
    'fungi': {'mu': FUNGI_MU, 'lambda_': FUNGI_LAMBDA,  'tag': 1},
    'wood':  {'E': WOOD_E,      'nu': WOOD_NU, 'tag': 2},
}

# ── Loading ───────────────────────────────────────────────────────────────────
TIMESTEPS  = 50
DISP_INCR  = 0.01

LOADING_DIRECTIONS = [
    (np.array([1., 0., 0.]),  'tension'),
    (np.array([-1., 0., 0.]), 'compression'),
]

# mix value encoded in directory name: small{int(mix*100):03d}_large{...}
MIX_VALUES = [0.0, 0.25, 0.50, 0.75, 1.0]

def config_name(mix):
    fs, fl = int(mix * 100), int((1 - mix) * 100)
    return f"small{fs:03d}_large{fl:03d}"


def build_samples(mesh_dir, shrink_factor):
    """Return list of (mix, seed, msh_file, load_dir, label)."""
    samples = []
    for mix in MIX_VALUES:
        cdir = mesh_dir / config_name(mix)
        if not cdir.is_dir():
            print(f"  Warning: {cdir} not found, skipping mix={mix}")
            continue
        msh_files = sorted(cdir.glob(f"seed*_shrink{shrink_factor}.msh"))
        for msh in msh_files:
            seed = int(msh.stem.split('_')[0].replace('seed', ''))
            for load_dir, load_label in LOADING_DIRECTIONS:
                label = f"mix{mix:.2f}_seed{seed}_{load_label}"
                samples.append((mix, seed, msh, load_dir, label))
    return samples


def main(mesh_dir, shrink_factor, output_dir):
    mesh_dir   = Path(mesh_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = build_samples(mesh_dir, shrink_factor)
    n_samples = len(samples)
    if n_samples == 0:
        print("No mesh files found. Check --mesh_dir and --shrink_factor.")
        sys.exit(1)

    print(f"Total samples: {n_samples}")
    for i, (mix, seed, msh, _, label) in enumerate(samples):
        print(f"  {i:3d}: {label}  mesh={msh.name}")

    # File paths
    prefix        = output_dir / "uniaxial_deposition"
    f_path        = f"{prefix}_F.npy"
    pk1_path      = f"{prefix}_PK1.npy"
    cauchy_path   = f"{prefix}_cauchy.npy"
    matparam_path = f"{prefix}_matparam.data"

    # Write matparam file
    with open(matparam_path, 'w') as f:
        f.write('mix seed load_sign mu lambda\n')
        for mix, seed, _, load_dir, label in samples:
            load_sign = int(load_dir[0])
            f.write(f'{mix:.2f} {seed} {load_sign} {MATERIAL_PROPS["fungi"]["mu"]} {MATERIAL_PROPS["fungi"]["lambda_"]}\n')

    # Load or create result arrays
    start_sample = 0
    if os.path.exists(f_path) and os.path.exists(pk1_path) and os.path.exists(cauchy_path):
        F_all              = np.load(f_path)
        stresses_PK1_all   = np.load(pk1_path)
        stresses_cauchy_all = np.load(cauchy_path)
        if F_all.shape[0] == n_samples:
            for i in range(n_samples):
                if np.allclose(F_all[i], 0):
                    start_sample = i
                    break
            else:
                start_sample = n_samples
            print(f"Resuming from sample {start_sample}/{n_samples}")
        else:
            print("Shape mismatch, starting fresh")
            F_all = stresses_PK1_all = stresses_cauchy_all = None
    else:
        F_all = stresses_PK1_all = stresses_cauchy_all = None

    if F_all is None:
        F_all               = np.zeros((n_samples, TIMESTEPS, 2, 2))
        stresses_PK1_all    = np.zeros((n_samples, TIMESTEPS, 2, 2))
        stresses_cauchy_all = np.zeros((n_samples, TIMESTEPS, 2, 2))

    if start_sample >= n_samples:
        print("All samples already complete.")
        sys.exit(0)

    # Run simulations
    failed_samples = []
    n_sub = 8

    for sample_id in range(start_sample, n_samples):
        mix, seed, msh_file, load_dir, label = samples[sample_id]

        F_id    = np.eye(2) + np.array([[load_dir[0], 0.], [0., load_dir[2]]])
        F_min_I = F_id - np.eye(2)

        print(f"\nSample {sample_id}/{n_samples} [{label}]")

        rve = RVEMaterial(mesh_file=str(msh_file),
                          material_properties=MATERIAL_PROPS,
                          visualize=False)

        F_prev = np.eye(2)

        for t in range(TIMESTEPS):
            F_macro = np.eye(2) + (t + 1) * DISP_INCR * F_min_I
            PK1, cauchy, converged = rve.update_stress(F_macro, monitor_base_solve=False)

            if not converged:
                print(f"  Timestep {t} failed, trying {n_sub} sub-increments...")
                delta_F = F_macro - F_prev
                sub_converged = True
                for sub_step in range(n_sub):
                    alpha = (sub_step + 1) / n_sub
                    F_sub = F_prev + alpha * delta_F
                    PK1, cauchy, converged = rve.update_stress(F_sub,
                                                               monitor_base_solve=False)
                    if not converged:
                        print(f"    Sub-increment {sub_step + 1} failed")
                        sub_converged = False
                        break
                if sub_converged:
                    print(f"  Sub-incrementing succeeded for timestep {t}")
                    converged = True

            if converged:
                F_all[sample_id, t]               = F_macro
                stresses_PK1_all[sample_id, t]    = PK1
                stresses_cauchy_all[sample_id, t] = cauchy
                F_prev = F_macro.copy()
            else:
                print(f"  Simulation failed at timestep {t}")
                failed_samples.append((sample_id, t, label))
                # Shift converged steps to end, fill beginning with undeformed
                F_all[sample_id, TIMESTEPS - t:]               = F_all[sample_id, :t]
                stresses_PK1_all[sample_id, TIMESTEPS - t:]    = stresses_PK1_all[sample_id, :t]
                stresses_cauchy_all[sample_id, TIMESTEPS - t:] = stresses_cauchy_all[sample_id, :t]
                F_all[sample_id, :TIMESTEPS - t]               = np.eye(2)
                stresses_PK1_all[sample_id, :TIMESTEPS - t]    = 0.
                stresses_cauchy_all[sample_id, :TIMESTEPS - t] = 0.
                break

        # Save after each sample for crash recovery
        np.save(f_path, F_all)
        np.save(pk1_path, stresses_PK1_all)
        np.save(cauchy_path, stresses_cauchy_all)

    print(f"\nDone. Failed samples: {failed_samples}")
    print(f"Results saved to {prefix}_*.npy / _matparam.data")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mesh_dir",      type=str,   default="../yade/generated_meshes_2D_wet/v1")
    p.add_argument("--shrink_factor", type=float, default=0.85)
    p.add_argument("--output_dir",    type=str,   default="../data/uniaxial_deposition/v1")
    args = p.parse_args()
    main(args.mesh_dir, args.shrink_factor, args.output_dir)
