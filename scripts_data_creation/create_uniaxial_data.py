import os
import numpy as np
import sys
from scripts_materials.rve_material import RVEMaterial
from material_params import WOOD_E, WOOD_NU, FUNGI_MU, FUNGI_LAMBDA

theta = np.pi / 4 * 0 / 3  # angle in radians
nfibs_vfrac_40 = 50  # Number of fibers giving vfrac = 40%

results_name = f'../data/uniaxial_v2'
try:
    os.makedirs(results_name, exist_ok=True)
except Exception as e:
    print(f"Could not create data folder: {e}. Maybe it already exists?")

timesteps = 50
dispIncr = 0.01
lambda_val = FUNGI_LAMBDA
mesh_folder = f"../meshes/vfrac_ratio_big/"

# ================== Parameter combinations ==================
# min/median/max for each variable, varied independently
num_fib_array = [10, 20, 30, 40, 50, 60, 70]
shape_array = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
mu_range = [FUNGI_MU / 5, FUNGI_MU * 5]

nfib_min, nfib_med, nfib_max = num_fib_array[0], num_fib_array[len(num_fib_array) // 2], num_fib_array[-1]
ratio_min, ratio_med, ratio_max = shape_array[0], shape_array[len(shape_array) // 2], shape_array[-1]
mu_min, mu_med, mu_max = mu_range[0], FUNGI_MU, mu_range[1]

# 7 combinations: 1 baseline (all median) + 2 per variable (min, max) × 3 variables
# Each row: (nfib, ratio, mu, label)
param_combinations = [
    (nfib_med,  ratio_med,  mu_med,  'median_all'),
    (nfib_min,  ratio_med,  mu_med,  'nfib_min'),
    (nfib_max,  ratio_med,  mu_med,  'nfib_max'),
    (nfib_med,  ratio_min,  mu_med,  'ratio_min'),
    (nfib_med,  ratio_max,  mu_med,  'ratio_max'),
    (nfib_med,  ratio_med,  mu_min,  'mu_min'),
    (nfib_med,  ratio_med,  mu_max,  'mu_max'),
]

# 2 loading directions: tension (+x) and compression (-x)
loading_directions = [
    (np.array([1., 0., 0.]),  'tension'),
    (np.array([-1., 0., 0.]), 'compression'),
]

# Build flat list of all 14 samples: (nfib, ratio, mu, loading_dir, label)
samples_list = [
    (nfib, ratio, mu, direction, f'{param_label}_{load_label}')
    for (nfib, ratio, mu, param_label) in param_combinations
    for (direction, load_label) in loading_directions
]
n_samples = len(samples_list)
print(f"Total samples: {n_samples}")
for i, (nfib, ratio, mu, _, label) in enumerate(samples_list):
    print(f"  {i:2d}: {label}  nfib={nfib}, ratio={ratio}, mu={mu:.4f}")

# ================== File paths ==================
f_path      = f"{results_name}/uniaxial_F.npy"
pk1_path    = f"{results_name}/uniaxial_PK1.npy"
cauchy_path = f"{results_name}/uniaxial_cauchy.npy"
matparam_path = f'{results_name}/uniaxial_matparam.data'

# Write matparam file
with open(matparam_path, 'w') as f:
    f.write('nfib vfrac angle ratio mu lambda loading\n')
    for nfib, ratio, mu, direction, label in samples_list:
        cur_vfrac = 0.4 / nfibs_vfrac_40 * nfib
        load_sign = int(direction[0])
        f.write(f'{nfib} {cur_vfrac:.4f} {theta:.4f} {ratio:.2f} {mu:.5f} {lambda_val} {load_sign}  # {label}\n')

# Load or create result arrays, find where to resume
start_sample = 0
if os.path.exists(f_path) and os.path.exists(pk1_path) and os.path.exists(cauchy_path):
    F_all = np.load(f_path)
    stresses_PK1_all = np.load(pk1_path)
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
        print(f"Shape mismatch, starting fresh")
        F_all = np.zeros((n_samples, timesteps, 2, 2))
        stresses_PK1_all = np.zeros((n_samples, timesteps, 2, 2))
        stresses_cauchy_all = np.zeros((n_samples, timesteps, 2, 2))
else:
    F_all = np.zeros((n_samples, timesteps, 2, 2))
    stresses_PK1_all = np.zeros((n_samples, timesteps, 2, 2))
    stresses_cauchy_all = np.zeros((n_samples, timesteps, 2, 2))

if start_sample >= n_samples:
    print("All samples already complete, nothing to do.")
    sys.exit(0)

# ================== Run RVE simulations ==================
failed_samples = []

for sample_id in range(start_sample, n_samples):
    nfib, ratio, mu, load_dir, label = samples_list[sample_id]
    nfib = int(nfib)
    cur_vfrac = 0.4 / nfibs_vfrac_40 * nfib

    mesh_name = f'nfib{nfib}_{theta:.4f}_{ratio:.2f}'
    meshfile = f"{mesh_folder}{mesh_name}/rve_0.msh"

    # Pure uniaxial in x-direction (F12 = F21 = 0, F22 = 0)
    F_id = np.eye(2) + np.array([[load_dir[0], 0.], [0., load_dir[2]]])
    F_min_I = F_id - np.eye(2)

    print(f"Sample {sample_id}/{n_samples} [{label}]: nfib={nfib}, ratio={ratio}, mu={mu:.4f}, F_dir={load_dir}")

    material_props = {
        'fungi': {'mu': mu, 'lambda_': lambda_val, 'tag': 1},
        'wood':  {'E': WOOD_E, 'nu': WOOD_NU, 'tag': 2},
    }

    rve = RVEMaterial(mesh_file=meshfile, material_properties=material_props, visualize=False)

    F_prev = np.eye(2)
    sample_failed = False
    n_sub_increments = 8

    for t in range(timesteps):
        F_macro = np.eye(2) + (t + 1) * dispIncr * F_min_I
        PK1, cauchy, converged = rve.update_stress(F_macro, monitor_base_solve=False)

        if not converged:
            print(f"  Timestep {t} failed, trying {n_sub_increments} sub-increments...")
            delta_F = F_macro - F_prev
            sub_converged = True
            for sub_step in range(n_sub_increments):
                alpha = (sub_step + 1) / n_sub_increments
                F_sub = F_prev + alpha * delta_F
                PK1, cauchy, converged = rve.update_stress(F_sub, monitor_base_solve=False)
                if not converged:
                    print(f"    Sub-increment {sub_step + 1} failed")
                    sub_converged = False
                    break
            if sub_converged:
                print(f"  Sub-incrementing succeeded for timestep {t}")
                converged = True

        if converged:
            F_all[sample_id, t] = F_macro
            stresses_PK1_all[sample_id, t] = PK1
            stresses_cauchy_all[sample_id, t] = cauchy
            F_prev = F_macro.copy()
        else:
            print(f"  Simulation failed at timestep {t}")
            failed_samples.append([sample_id, t, nfib, ratio, mu, label])
            sample_failed = True

            # Shift converged steps to the end, fill beginning with undeformed state
            F_all[sample_id, timesteps - t:] = F_all[sample_id, :t]
            stresses_PK1_all[sample_id, timesteps - t:] = stresses_PK1_all[sample_id, :t]
            stresses_cauchy_all[sample_id, timesteps - t:] = stresses_cauchy_all[sample_id, :t]
            F_all[sample_id, :timesteps - t] = np.eye(2)
            stresses_PK1_all[sample_id, :timesteps - t] = 0.
            stresses_cauchy_all[sample_id, :timesteps - t] = 0.
            break

    # Save after each sample for crash recovery
    np.save(f_path, F_all)
    np.save(pk1_path, stresses_PK1_all)
    np.save(cauchy_path, stresses_cauchy_all)

print(f"\nDone. Failed samples: {failed_samples}")
