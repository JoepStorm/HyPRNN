import os
import numpy as np
import random
import sys
from scripts_materials.RVE_material import RVEMaterial

# Default settings
seed = 1

sys_args = sys.argv
if len(sys_args) > 1:  # There is a parameter given, so part of batch job.
    seed = int(sys_args[1])
    print(f"Job with seed: {seed}")
else:
    print(f"No parameter given to python script - using seed = {seed}")

theta = np.pi / 4 * 0 / 3  # angle in radians. 2/3 = 30, 4/3=60, 6/3=90, 8/3=120 10/3=150
nfibs_vfrac_40 = 50  # Number of fibers giving vfrac = 40% Old: 25.


def lambda_mu_from_E_nu(E, nu):
    mu = E / (2 * (1 + nu))
    lambda_val = E * nu / ((1 + nu) * (1 - 2 * nu))
    return lambda_val, mu


def E_nu_from_mu_lambda(mu, lambda_):
    E = mu * (3 * lambda_ + 2 * mu) / (lambda_ + mu)
    nu = lambda_ / (2 * (lambda_ + mu))
    return E, nu


# results_name = f'../data/RVE/vfrac_ratio_smallmu_biggerdomain/10runs_{seed}'
# results_name = f'../data/RVE/vfrac_ratio_smallmu_biggerdomain/10runs_literature_matprops'
# results_name = f'../data/RVE/vfrac_ratio_smallmu_biggerdomain/lit_props_constant'
# results_name = f'../data/RVE/vary_mu/vary_mu'
# results_name = f'../data/RVE/vary_vfrac_ratio/vary_vfrac_ratio'
results_name = f'../data/RVE/vary_all/vary_all'
# results_name = f'../data/RVE/vfrac_ratio_E/{seed}'
try:
    os.makedirs(results_name, exist_ok=True)
except Exception as e:
    print(f"Could not create data folder: {e}. Maybe it already exists?")

pro_file_name = f'{results_name}/cur_settings.pro'
timesteps = 50
dispIncr = 0.01
keep_failed_runs = True
materials = ['wood', 'fungi']
lambda_val = 0.62
# lambda_val = 0.15e3

base_filename = f'mixed_t{timesteps}_seed{seed}'
mesh_folder = f"../meshes/micro/vfrac_ratio_big/"
max_retries = 0  # 40
xy_factor = 1.0  # 0.5 # scale non-diagonal F terms with this factor to avoid excessive shear
gen_mat_params = True
np.random.seed(seed)
random.seed(seed)

random_sampling = True  # True: uniform random, False: grid + repeat


def is_valid_deformation_gradient(F):
    det_F = np.linalg.det(F)
    if det_F <= 0:
        return False

    # Check for extreme deformation
    if det_F < 0.01 or det_F > 100:
        return False

    # Check principal stretches
    C = F.T @ F
    eigenvals = np.linalg.eigvals(C)
    if np.any(eigenvals <= 0) or np.any(np.sqrt(eigenvals) > 10):
        return False

    return True


def randVector():
    """
    Computing random load vector: A 3D unit vector with random 'direction'
    Based on answer https://math.stackexchange.com/questions/44689/how-to-find-a-random-axis-or-unit-vector-in-3d
    (𝑥,𝑦,𝑧)=(sqrt(1−𝑧^2)cos𝜃,sqrt(1−𝑧^2)sin𝜃,𝑧)
    """
    theta = random.uniform(0, 2 * np.pi)
    z0 = random.uniform(-1, 1)
    out = np.empty(3)
    out[0] = np.sqrt(1 - z0 ** 2) * np.cos(theta)
    out[1] = np.sqrt(1 - z0 ** 2) * np.sin(theta)
    out[2] = z0
    return out


"""
Create loadpath
"""
f_path = f"{results_name}/{base_filename}_F.npy"
pk1_path = f"{results_name}/{base_filename}_PK1.npy"
cauchy_path = f"{results_name}/{base_filename}_cauchy.npy"
matparam_path = f'{results_name}/{base_filename}_matparam.data'

# Load existing matparams or create new ones
start_sample = 0
if os.path.exists(matparam_path):
    print(f"Found existing matparams: {matparam_path}")
    # Parse existing matparams to reconstruct fib_shape_mat_combinations
    with open(matparam_path, 'r') as f:
        lines = f.readlines()[1:]  # skip header
    # Columns: nfib(0) vfrac(1) angle(2) ratio(3) mu(4) lambda(5)
    # fib_shape_mat_combinations needs: nfib, ratio, mu
    parsed = np.loadtxt(lines)
    fib_shape_mat_combinations = parsed[:, [0, 3, 4]]
    samples = fib_shape_mat_combinations.shape[0]
    print(f"Loaded {samples} samples from existing matparams")
else:
    print(f"No existing matparams, creating new ones: {matparam_path}")

    if random_sampling:
        # # vary mu only
        # num_fib_array = [40]
        # shape_array = [2.0]
        # mu_range = [0.51 / 5, .51 * 5]

        # # vary all except mu
        # num_fib_array = [10, 20, 30, 40, 50, 60, 70]
        # shape_array = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
        # mu_range = [0.51, 0.51]

        # vary all
        num_fib_array = [10, 20, 30, 40, 50, 60, 70]
        shape_array = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
        mu_range = [0.51 / 5, .51 * 5]

        total_samples = 512
        fib_shape_mat_combinations = np.column_stack([
            np.random.choice(num_fib_array, total_samples),
            np.random.choice(shape_array, total_samples),
            np.random.uniform(mu_range[0], mu_range[1], total_samples),
        ])
    else:
        num_fib_array = [40]
        shape_array = [2.0]
        mu_vals = [0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5]
        fib_shape_mat_combinations = np.array(np.meshgrid(num_fib_array, shape_array, mu_vals)).T.reshape(-1, 3)
        num_runs_per_setting = 490
        fib_shape_mat_combinations = np.repeat(fib_shape_mat_combinations, num_runs_per_setting, axis=0)

    samples = fib_shape_mat_combinations.shape[0]

    with open(matparam_path, 'w') as f:
        f.write('nfib vfrac angle ratio mu lambda\n')
        for i in range(samples):
            nfib = int(fib_shape_mat_combinations[i, 0])
            cur_vfrac = 0.4 / nfibs_vfrac_40 * nfib
            ratio = fib_shape_mat_combinations[i, 1]
            mu = fib_shape_mat_combinations[i, 2]
            f.write(f'{nfib} {cur_vfrac} {theta} {ratio:.2f} {mu} {lambda_val}\n')

# Load or create result arrays, find where to continue
if os.path.exists(f_path) and os.path.exists(pk1_path) and os.path.exists(cauchy_path):
    F_all = np.load(f_path)
    stresses_PK1_all = np.load(pk1_path)
    stresses_cauchy_all = np.load(cauchy_path)

    if F_all.shape[0] == samples:
        for i in range(samples):
            if np.allclose(F_all[i], 0):
                start_sample = i
                break
        else:
            start_sample = samples  # All done
        print(f"Continuing from sample {start_sample}/{samples}")
    else:
        print(f"Shape mismatch ({F_all.shape[0]} vs {samples}), starting fresh")
        F_all = np.zeros((samples, timesteps, 2, 2))
        stresses_PK1_all = np.zeros((samples, timesteps, 2, 2))
        stresses_cauchy_all = np.zeros((samples, timesteps, 2, 2))
else:
    F_all = np.zeros((samples, timesteps, 2, 2))
    stresses_PK1_all = np.zeros((samples, timesteps, 2, 2))
    stresses_cauchy_all = np.zeros((samples, timesteps, 2, 2))

if start_sample >= samples:
    print("All samples already complete, nothing to do.")
    sys.exit(0)

failed_samples = []

sample_id = 0
for sample_id in range(start_sample, samples):
    # Per-sample seed ensures reproducibility even when continuing from a crash
    random.seed(seed * 100000 + sample_id)

    nfib = int(fib_shape_mat_combinations[sample_id, 0])
    cur_vfrac = 0.4 / nfibs_vfrac_40 * nfib  # assumes the void sizes staying constant such that #(nfibs_vfrac_40) fibers = 40%.
    ratio = fib_shape_mat_combinations[sample_id, 1]
    mu = fib_shape_mat_combinations[sample_id, 2]

    mesh_name = f'nfib{nfib}_{theta:.4f}_{ratio:.2f}'
    meshfile = f"{mesh_folder}{mesh_name}/rve_0.msh"

    """
    Create new load paths files
    """
    F_valid = False
    while not F_valid:
        # Generate a 3-component vector with length 1 in random direction
        random_dir = randVector()
        # We sample in U, keeping R=Identity. That way we cover the full possible space the surrogate will see.
        # In practice this just means F=U while ensuring that the off-diagonal terms are the same
        F_id = np.eye(2) + np.array(
            [[random_dir[0], random_dir[1] * xy_factor], [random_dir[1] * xy_factor, random_dir[2]]])
        F_valid = is_valid_deformation_gradient(F_id)
    print(
        f"Trying: F = {F_id} for sample_id: {sample_id}/{samples} with nfib: {nfib}, mu: {mu}, ratio: {ratio}, mesh: {mesh_name}")

    # if sample_id < 2:
    #     continue

    # Set F-I to the BC's
    F_min_I = F_id - np.eye(2)

    # transfer theta to fibdir for Bonet:
    fibdir = [np.cos(theta), np.sin(theta)]

    material_props = {
        # 'fungi': {'mu': mu, 'lambda_': 150., 'tag': 1},
        'fungi': {'mu': mu, 'lambda_': lambda_val, 'tag': 1},
        # 'fungi': {'E': 1.3, 'nu': 0.3, 'tag': 1},           # E based on paper Stochastic continuum model for mycelium-based bio-foam.
        'wood': {'E': 13.e3, 'nu': 0.375, 'tag': 2}
        # 'wood': {'mu': 4.727e3, 'lambda_': 14.182e3, 'tag': 2}       # equivalent to E=13e3, nu=0.375
    }

    # Initialize the RVE material model
    rve = RVEMaterial(
        mesh_file=meshfile,
        material_properties=material_props,
        visualize=False,
    )

    n_sub_increments = 8  # Number of sub-increments for retry
    F_prev = np.eye(2)
    sample_failed = False

    for t in range(timesteps):
        F_macro = np.eye(2) + (t + 1) * dispIncr * F_min_I
        PK1, cauchy, converged = rve.update_stress(F_macro, monitor_base_solve=False)

        if not converged:
            # Try sub-incrementing before giving up
            print(f"Timestep {t} failed, trying {n_sub_increments} sub-increments...")
            delta_F = F_macro - F_prev
            sub_converged = True

            for sub_step in range(n_sub_increments):
                alpha = (sub_step + 1) / n_sub_increments
                F_sub = F_prev + alpha * delta_F
                PK1, cauchy, converged = rve.update_stress(F_sub, monitor_base_solve=False)

                if not converged:
                    print(f"  Sub-increment {sub_step + 1}/{n_sub_increments} failed")
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
            print(f"Simulation failed for timestep {t}")
            failed_samples.append([sample_id, t, nfib, float(ratio), float(mu)])
            sample_failed = True

            # Move converged steps to the end of the arrays
            F_all[sample_id, timesteps - t:] = F_all[sample_id, :t]
            stresses_PK1_all[sample_id, timesteps - t:] = stresses_PK1_all[sample_id, :t]
            stresses_cauchy_all[sample_id, timesteps - t:] = stresses_cauchy_all[sample_id, :t]

            # Fill the beginning of F_all with identity matrices
            F_all[sample_id, :timesteps - t] = np.eye(2)
            stresses_PK1_all[sample_id, :timesteps - t] = np.zeros(2)
            stresses_cauchy_all[sample_id, :timesteps - t] = np.zeros(2)

            # Move on to the next sample.
            break

    # Save after each sample for crash recovery
    np.save(pk1_path, stresses_PK1_all)
    np.save(cauchy_path, stresses_cauchy_all)
    np.save(f_path, F_all)
