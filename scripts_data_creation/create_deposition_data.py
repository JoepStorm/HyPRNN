"""Generate mixed-loading RVE data from YADE-deposited 2D filler meshes.

Reads meshes from a single folder: coords_2D_500_{fil_frac}.msh with fil_frac
in {0.0, 0.1, ..., 0.9}. For each sample, randomly draws a filler fraction and a
random loading direction, then runs an RVE simulation. Matches the format of
create_mixed_data.py so the output can be used with the same data pipeline.

Matparam columns: mu  lambda  fil_frac

Usage:
    - Set TOTAL_SAMPLES: number of samples to generate, and other settings. Then:
    python create_deposition_data.py          # seed=1
    python create_deposition_data.py 3        # seed=3 (batch job)

In the paper, 16 seeds each generate a subset of the data, which are then merged using data/merge_datasets.py

Some samples will fail. In that case, the output will say "Done. Failed: [#numbers failed].
If none fail, output = "Done. Failed []"
"""

import os
import sys
import random
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts_materials.rve_material import RVEMaterial
from material_params import WOOD_E, WOOD_NU, FUNGI_MU, FUNGI_LAMBDA

# Job seed (controls both sampling and reproducibility)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
print(f"Seed: {seed}")
np.random.seed(seed)
random.seed(seed)

# # On cluster only: starting at the same time causes errors
# import time
# time.sleep(2*seed)

# Settings
TIMESTEPS    = 50
DISP_INCR    = 0.01
TOTAL_SAMPLES = 1
XY_FACTOR    = 1.0        # scale off-diagonal F terms (1.0 = full shear)
LAMBDA_VAL   = FUNGI_LAMBDA       # fixed matrix bulk parameter
MU_RANGE     = [FUNGI_MU, FUNGI_MU]     # if minimum=maximum  -> fixed mu.
MIX_VALUES   = [0.0, 0.25, 0.50, 0.75, 1.0]
SHRINK_FACTOR = 0.85

MESH_DIR   = Path("../yade/data/dataset_combi_v4v6")
# OUTPUT_DIR = Path(f"../data/RVE/deposition/filler/dataset_combi_v4v6/")
OUTPUT_DIR = Path(f"../data/deposition/example/")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

base_filename = f"mixed_t{TIMESTEPS}_seed{seed}"
f_path        = OUTPUT_DIR / f"{base_filename}_F.npy"
pk1_path      = OUTPUT_DIR / f"{base_filename}_PK1.npy"
matparam_path = OUTPUT_DIR / f"{base_filename}_matparam.data"

MATERIAL_PROPS_TEMPLATE = {
    'wood': {'E': WOOD_E, 'nu': WOOD_NU, 'tag': 2},
}


# Helpers
FIL_FRACS = [round(0.1 * i, 1) for i in range(10)]   # 0.0, 0.1, ..., 0.9


def mesh_path(fil_frac):
    return MESH_DIR / f"coords_2D_500_{fil_frac:.1f}.msh"


def available_fracs():
    """Filler fractions whose mesh exists in MESH_DIR."""
    return [f for f in FIL_FRACS if mesh_path(f).exists()]

def randVector():
    """Uniform random unit vector on S^2."""
    theta = random.uniform(0, 2 * np.pi)
    z0    = random.uniform(-1, 1)
    return np.array([np.sqrt(1 - z0**2) * np.cos(theta),
                     np.sqrt(1 - z0**2) * np.sin(theta),
                     z0])


def is_valid_F(F):
    det = np.linalg.det(F)
    if det <= 0 or det < 0.01 or det > 100:
        return False
    eigenvals = np.linalg.eigvals(F.T @ F)
    return np.all(eigenvals > 0) and np.all(np.sqrt(eigenvals) < 10)


# Sample plan
available = available_fracs()
if not available:
    print(f"No meshes found in {MESH_DIR}. Check MESH_DIR.")
    sys.exit(1)

start_sample = 0
if matparam_path.exists():
    print(f"Found existing matparams: {matparam_path}")
    rows = np.loadtxt(matparam_path, skiprows=1, comments='#')
    if rows.ndim == 1:
        rows = rows[np.newaxis, :]
    # columns: mu lambda fil_frac
    mu_samples   = rows[:, 0]
    frac_samples = rows[:, 2]
    samples = len(mu_samples)
    print(f"Loaded {samples} samples")
else:
    print(f"Creating new sample plan: {matparam_path}")
    mu_samples   = np.random.uniform(MU_RANGE[0], MU_RANGE[1], TOTAL_SAMPLES)
    frac_samples = np.array([random.choice(available) for _ in range(TOTAL_SAMPLES)])
    samples = TOTAL_SAMPLES

    with open(matparam_path, 'w') as f:
        f.write('mu lambda fil_frac\n')
        for i in range(samples):
            f.write(f"{mu_samples[i]:.6f} {LAMBDA_VAL} {frac_samples[i]:.2f}\n")

# Result arrays
if f_path.exists() and pk1_path.exists():
    F_all               = np.load(f_path)
    stresses_PK1_all    = np.load(pk1_path)
    if F_all.shape[0] == samples:
        for i in range(samples):
            if np.allclose(F_all[i], 0):
                start_sample = i
                break
        else:
            start_sample = samples
        print(f"Continuing from sample {start_sample}/{samples}")
    else:
        print("Shape mismatch, starting fresh")
        F_all = stresses_PK1_all = None
else:
    F_all = stresses_PK1_all = None

if F_all is None:
    F_all               = np.zeros((samples, TIMESTEPS, 2, 2))
    stresses_PK1_all    = np.zeros((samples, TIMESTEPS, 2, 2))

if start_sample >= samples:
    print("All samples complete.")
    sys.exit(0)

# Simulation loop
failed_samples = []

for sample_id in range(start_sample, samples):
    random.seed(seed * 100000 + sample_id)

    mu        = float(mu_samples[sample_id])
    fil_frac  = float(frac_samples[sample_id])
    meshfile  = str(mesh_path(fil_frac))

    # Random loading direction
    F_valid = False
    while not F_valid:
        d = randVector()
        F_id = np.eye(2) + np.array([[d[0], d[1] * XY_FACTOR],
                                     [d[1] * XY_FACTOR, d[2]]])
        F_valid = is_valid_F(F_id)

    F_min_I = F_id - np.eye(2)
    print(f"Sample {sample_id}/{samples}: fil_frac={fil_frac:.1f}, mu={mu:.4f}, "
          f"mesh={Path(meshfile).name}")

    material_props = {
        **MATERIAL_PROPS_TEMPLATE,
        'fungi': {'mu': mu, 'lambda_': LAMBDA_VAL, 'tag': 1},
    }
    rve = RVEMaterial(mesh_file=meshfile,
                      material_properties=material_props,
                      visualize=False)

    F_prev = np.eye(2)
    n_sub  = 8

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
                converged = True

        if converged:
            F_all[sample_id, t]               = F_macro
            stresses_PK1_all[sample_id, t]    = PK1
            F_prev = F_macro.copy()
        else:
            print(f"  Simulation failed at timestep {t}")
            failed_samples.append((sample_id, t, fil_frac, mu))
            F_all[sample_id, TIMESTEPS - t:]               = F_all[sample_id, :t]
            stresses_PK1_all[sample_id, TIMESTEPS - t:]    = stresses_PK1_all[sample_id, :t]
            F_all[sample_id, :TIMESTEPS - t]               = np.eye(2)
            stresses_PK1_all[sample_id, :TIMESTEPS - t]    = 0.
            break

    np.save(f_path, F_all)
    np.save(pk1_path, stresses_PK1_all)

print(f"\nDone. Failed: {failed_samples}")
