"""Generate mixed-loading RVE data from YADE-deposited 2D wet meshes.

For each sample, randomly draws a (mix, seed, mu) combination and a random
loading direction, then runs an RVE simulation.  Matches the format of
createMixedData.py so the output can be used with the same data pipeline.

Matparam columns: mu  lambda  mix

Usage:
    python createDepositionData.py          # seed=1
    python createDepositionData.py 3        # seed=3 (batch job)
"""

import os
import sys
import random
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts_materials.RVE_material import RVEMaterial
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
TOTAL_SAMPLES = 32
XY_FACTOR    = 1.0        # scale off-diagonal F terms (1.0 = full shear)
LAMBDA_VAL   = FUNGI_LAMBDA       # fixed matrix bulk parameter
MU_RANGE     = [FUNGI_MU, FUNGI_MU]
MIX_VALUES   = [0.0, 0.25, 0.50, 0.75, 1.0]
SHRINK_FACTOR = 0.85

# Set True to only try loading every mesh (no simulation, ignores INVALID_MESHES)
# and print the (mix, seed) pairs that fail. Script exits after the check.
CHECK_MESHES_ONLY = False

# MESH_DIR   = Path("../yade/generated_meshes_2D_wet/v1")
# OUTPUT_DIR = Path(f"../data/deposition/v1/p3")
# MESH_DIR   = Path("../yade/generated_meshes_2D_wet/v5_bigrve")
# OUTPUT_DIR = Path(f"../data/deposition/v5_bigrve/")
MESH_DIR   = Path("../yade/generated_meshes_2D_wet/v10")
OUTPUT_DIR = Path(f"../data/deposition/v10/")
# Meshes with invalid PBC (unequal boundary DOFs) — exclude from pool.
# Add entries as (mix, seed) tuples, e.g. (0.0, 2) or (0.75, 4).
# INVALID_MESHES = {
#     (0.0, 12),
#     (0.25, 16),
#     (0.5, 17),
#     (0.5, 6),
#     (0.75, 11),
#     (0.75, 19),
#     (1.0, 9),
# }

INVALID_MESHES = {
(0.25, 0),
(0.25, 2),
(0.25, 4),
(0.25, 5),
(0.25, 6),
(0.5, 4),
(0.5, 9),
(0.75, 1),
(0.75, 9),
(1.0, 6),
(1.0, 7),
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

base_filename = f"mixed_t{TIMESTEPS}_seed{seed}"
f_path        = OUTPUT_DIR / f"{base_filename}_F.npy"
pk1_path      = OUTPUT_DIR / f"{base_filename}_PK1.npy"
matparam_path = OUTPUT_DIR / f"{base_filename}_matparam.data"

MATERIAL_PROPS_TEMPLATE = {
    'wood': {'E': WOOD_E, 'nu': WOOD_NU, 'tag': 2},
}


# Helpers
def mesh_seed_from_path(p):
    """Extract integer seed from e.g. seed3_shrink0.85.msh → 3."""
    return int(Path(p).stem.split('_')[0].replace('seed', ''))


def mesh_path(mix, mesh_seed):
    return (MESH_DIR / f"small{int(mix*100):03d}_large{int((1-mix)*100):03d}"
            / f"seed{mesh_seed}_shrink{SHRINK_FACTOR}.msh")


def available_meshes():
    """Return {mix: [msh_path, ...]} for all meshes that exist, excluding INVALID_MESHES."""
    meshes = {}
    for mix in MIX_VALUES:
        paths = sorted(MESH_DIR.glob(
            f"small{int(mix*100):03d}_large{int((1-mix)*100):03d}"
            f"/seed*_shrink{SHRINK_FACTOR}.msh"))
        paths = [p for p in paths
                 if (mix, mesh_seed_from_path(p)) not in INVALID_MESHES]
        if paths:
            meshes[mix] = paths
    return meshes


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
mesh_pool = available_meshes()
if not mesh_pool:
    print(f"No meshes found in {MESH_DIR}. Check MESH_DIR and SHRINK_FACTOR.")
    sys.exit(1)
available_mixes = list(mesh_pool.keys())

# ---------------------------------------------------------------------------
# Mesh validation mode: try to load every mesh, collect failures, then exit.
# No simulation, no loading steps — purely RVEMaterial construction.
# ---------------------------------------------------------------------------
if CHECK_MESHES_ONLY:
    print("\n=== Mesh check mode: loading all meshes, no simulation ===")
    check_material_props = {
        **MATERIAL_PROPS_TEMPLATE,
        'fungi': {'mu': MU_RANGE[0], 'lambda_': LAMBDA_VAL, 'tag': 1},
    }
    failed_meshes = []
    checked = 0
    for mix_val in MIX_VALUES:
        paths = sorted(MESH_DIR.glob(
            f"small{int(mix_val*100):03d}_large{int((1-mix_val)*100):03d}"
            f"/seed*_shrink{SHRINK_FACTOR}.msh"))
        for p in paths:
            mseed = mesh_seed_from_path(p)
            checked += 1
            try:
                RVEMaterial(mesh_file=str(p),
                            material_properties=check_material_props,
                            visualize=False)
                print(f"  OK   mix={mix_val:.2f} seed={mseed}")
            except Exception as e:
                print(f"  FAIL mix={mix_val:.2f} seed={mseed}: "
                      f"{type(e).__name__}: {e}")
                failed_meshes.append((mix_val, mseed))

    print(f"\nChecked {checked} meshes, {len(failed_meshes)} failed.")
    print("Failed (mix, seed) pairs:")
    for entry in failed_meshes:
        print(f"  {entry},")
    sys.exit(0)

start_sample = 0
if matparam_path.exists():
    print(f"Found existing matparams: {matparam_path}")
    rows = np.loadtxt(matparam_path, skiprows=1, comments='#')
    if rows.ndim == 1:
        rows = rows[np.newaxis, :]
    # columns: mu lambda mix mesh_seed
    mu_samples        = rows[:, 0]
    mix_samples       = rows[:, 2]
    mesh_seed_samples = rows[:, 3].astype(int)
    samples = len(mu_samples)
    print(f"Loaded {samples} samples")
else:
    print(f"Creating new sample plan: {matparam_path}")
    mu_samples        = np.random.uniform(MU_RANGE[0], MU_RANGE[1], TOTAL_SAMPLES)
    mix_samples       = np.array([random.choice(available_mixes) for _ in range(TOTAL_SAMPLES)])
    mesh_seed_samples = np.array([mesh_seed_from_path(random.choice(mesh_pool[m])) for m in mix_samples])
    samples = TOTAL_SAMPLES

    with open(matparam_path, 'w') as f:
        f.write('mu lambda mix mesh_seed\n')
        for i in range(samples):
            f.write(f"{mu_samples[i]:.6f} {LAMBDA_VAL} {mix_samples[i]:.2f}"
                    f" {mesh_seed_samples[i]}\n")

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
    mix       = float(mix_samples[sample_id])
    meshfile  = str(mesh_path(mix, mesh_seed_samples[sample_id]))

    # Random loading direction
    F_valid = False
    while not F_valid:
        d = randVector()
        F_id = np.eye(2) + np.array([[d[0], d[1] * XY_FACTOR],
                                     [d[1] * XY_FACTOR, d[2]]])
        F_valid = is_valid_F(F_id)

    F_min_I = F_id - np.eye(2)
    print(f"Sample {sample_id}/{samples}: mix={mix:.2f}, mu={mu:.4f}, "
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
            failed_samples.append((sample_id, t, mix, mu))
            F_all[sample_id, TIMESTEPS - t:]               = F_all[sample_id, :t]
            stresses_PK1_all[sample_id, TIMESTEPS - t:]    = stresses_PK1_all[sample_id, :t]
            F_all[sample_id, :TIMESTEPS - t]               = np.eye(2)
            stresses_PK1_all[sample_id, :TIMESTEPS - t]    = 0.
            break

    np.save(f_path, F_all)
    np.save(pk1_path, stresses_PK1_all)

print(f"\nDone. Failed: {failed_samples}")
