"""
Validate optimized ring grading: run with PRNN surrogate, then with FE² RVEs, and compare.
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.style.use(['science', 'bright'])
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
from matplotlib import rc
rc('text', usetex=True)

from graded_ring_pressure import GradedRingSimulation, plot_stress_comparison
from optimize_ring import RingOptimizer


def run_and_store(sim, output_folder, output_name):
    """Run simulation with adaptive stepping, storing nodal coords, F, and PK1 at each step."""
    sim._setup_material_and_problem()
    sim._setup_solver()

    ref_coords = sim.V.tabulate_dof_coordinates()[:, :2]
    coords_history = [ref_coords.copy()]
    F_identity = np.tile([1, 1, 0, 0], (sim.num_ips, 1))
    F_history = [F_identity]
    vm_peak_history = [0.0]
    # Full history including failed steps (for debugging)
    F_all = [F_identity.copy()]
    PK1_all = [np.zeros_like(F_identity)]
    load_history = [0.0]
    converged_history = [True]

    from dolfinx import io
    from mpi4py import MPI

    micro_funcs = sim._create_micro_variable_functions()
    sim._create_stress_strain_functions()
    sim._update_stress_strain_functions()
    stress_strain_funcs = list(sim._S_funcs.values()) + list(sim._E_funcs.values())

    cur_step_size = sim.step_size_init
    cur_load = 0.0
    converged_steps = 0
    converged = True

    with io.VTXWriter(MPI.COMM_WORLD, f"{output_folder}/{output_name}.bp",
                       [sim.u] + micro_funcs + stress_strain_funcs) as xf:
        xf.write(0.0)

        while abs(cur_load) < sim.max_load - 1e-10:
            if converged:
                sim.u_pre.x.array[:] = sim.u.x.array[:]
                converged_steps += 1
                cur_load += cur_step_size
            else:
                sim.u.x.array[:] = sim.u_pre.x.array[:]
                sim._destroy_problem()
                sim._setup_solver()
                print(f"Load {cur_load + cur_step_size:.5f} did not converge, reducing step size.")
                cur_step_size *= sim.step_size_factor
                cur_load += cur_step_size

            if abs(cur_step_size) < abs(sim.step_size_min):
                print("Step size too small, stopping simulation.")
                break

            sim.load_magnitude.value = cur_load

            try:
                sim.problem.solve()
                converged = sim.problem.solver.getConvergedReason() > 0
                nr_iters = sim.problem.solver.getIterationNumber()
                print(f"Step {converged_steps}: load={cur_load:.5f}, converged={converged}, iters={nr_iters}")
            except Exception:
                print("Error while solving loop.")
                converged = False

            # Store F and PK1 regardless of convergence (for debugging failed steps)
            F_grad = sim.qmap.gradients["F"]
            F_vals = sim.qmap.get_gradient_vals(F_grad, sim.qmap.cells)
            pk1_vals = sim.qmap.fluxes["PK1"].x.array.reshape(-1, 4)
            F_all.append(F_vals.copy())
            PK1_all.append(pk1_vals.copy())
            load_history.append(cur_load)
            converged_history.append(converged)

            if converged:
                u_array = sim.u.x.array.reshape(-1, 2)
                coords_history.append((ref_coords + u_array).copy())

                F_history.append(F_vals.copy())
                vm = sim.get_von_mises_at_qp()
                vm_peak_history.append(float(vm.max()))

                if hasattr(sim, '_S_funcs'):
                    sim._update_stress_strain_functions()
                xf.write(converged_steps)
            else:
                cur_load -= cur_step_size

    # Save per-IP load paths (including failed steps) for debugging
    ip_data = {
        'F_all': np.array(F_all),               # [n_attempts+1, num_ips, 4]
        'PK1_all': np.array(PK1_all),           # [n_attempts+1, num_ips, 4]
        'load_all': np.array(load_history),      # [n_attempts+1]
        'converged_all': np.array(converged_history),  # [n_attempts+1]
        'qp_coords': sim.qp_coords,             # [num_ips, 2]
    }
    # Store per-IP material properties for standalone RVE replay
    if hasattr(sim, 'micro_variables'):
        for key, val in sim.micro_variables.items():
            ip_data[f'micro_{key}'] = np.asarray(val)
    ip_path = f"{output_folder}/{output_name}_ip_paths.npz"
    np.savez(ip_path, **ip_data)
    print(f"Saved per-IP load paths to {ip_path}")

    return np.array(coords_history), np.array(F_history), np.array(vm_peak_history)


def run_comparison(
    results_file,
    n_ctrl=2,
    optimize_mu=True,
    macro_meshsize=0.03,
    R_inner=0.2,
    R_outer=0.5,
    prnn_model_loc="../trained_models/train_vary_all_v2/prnn_nonlin_1L8_12m_sigmoid/samples512_run1",
    rve_meshsize=0.0125,
    output_folder="../results/graded_ring/validate",
    max_load=1.0,
    step_size_init=1.0,
):
    """Load optimized grading and run both PRNN and RVE simulations."""
    os.makedirs(output_folder, exist_ok=True)

    # Load optimized parameters
    data = np.load(results_file)
    x_best = data['x_best']
    print(f"Loaded optimized parameters from {results_file}")
    print(f"  x_best = {x_best}")

    # --- PRNN simulation ---
    print("\n" + "="*60)
    print("Running PRNN simulation")
    print("="*60)

    sim_prnn = GradedRingSimulation(
        macro_meshsize=macro_meshsize,
        R_inner=R_inner,
        R_outer=R_outer,
        use_surrogate=True,
        output_folder=output_folder + "/",
        prnn_model_loc=prnn_model_loc,
    )
    sim_prnn.max_load = max_load
    sim_prnn.step_size_init = step_size_init

    # Apply optimized grading via the optimizer's interpolation
    opt = RingOptimizer(
        output_folder=output_folder,
        prnn_model_loc=prnn_model_loc,
        macro_meshsize=macro_meshsize,
        R_inner=R_inner,
        R_outer=R_outer,
        n_ctrl=n_ctrl,
        optimize_mu=optimize_mu,
    )
    # Use optimizer's interpolation on the PRNN sim
    vfrac_ctrl, r_combi_ctrl, mu_ctrl = opt._unpack(x_best)
    r_qp_prnn = np.sqrt(sim_prnn.qp_coords[:, 0]**2 + sim_prnn.qp_coords[:, 1]**2)
    r_ctrl = np.linspace(R_inner, R_outer, n_ctrl)
    sim_prnn.apply_radial_grading(
        np.interp(r_qp_prnn, r_ctrl, vfrac_ctrl),
        np.interp(r_qp_prnn, r_ctrl, r_combi_ctrl),
        np.interp(r_qp_prnn, r_ctrl, mu_ctrl),
    )

    prnn_coords, prnn_F, prnn_vm_peak = run_and_store(sim_prnn, output_folder, 'ring_prnn')
    prnn_vm = sim_prnn.get_von_mises_at_qp()
    prnn_du = sim_prnn.get_nodal_displacements()
    print(f"PRNN: {prnn_coords.shape[0]} configurations stored")

    # --- RVE simulation ---
    print("\n" + "="*60)
    print("Running RVE (FE2) simulation")
    print("="*60)

    rve_config = {
        'meshsize': rve_meshsize,
        'verbose': True,
        'plot_meshes': False,
        'reuse_meshes': False,
    }

    sim_rve = GradedRingSimulation(
        macro_meshsize=macro_meshsize,
        R_inner=R_inner,
        R_outer=R_outer,
        use_surrogate=False,
        rve_config=rve_config,
        output_folder=output_folder + "/",
        prnn_model_loc=prnn_model_loc,
    )
    sim_rve.max_load = max_load
    sim_rve.step_size_init = step_size_init

    # Apply the same optimized grading
    r_qp_rve = np.sqrt(sim_rve.qp_coords[:, 0]**2 + sim_rve.qp_coords[:, 1]**2)
    sim_rve.apply_radial_grading(
        np.interp(r_qp_rve, r_ctrl, vfrac_ctrl),
        np.interp(r_qp_rve, r_ctrl, r_combi_ctrl),
        np.interp(r_qp_rve, r_ctrl, mu_ctrl),
    )

    rve_coords, rve_F, rve_vm_peak = run_and_store(sim_rve, output_folder, 'ring_rve')
    rve_vm = sim_rve.get_von_mises_at_qp()
    rve_du = sim_rve.get_nodal_displacements()
    print(f"RVE: {rve_coords.shape[0]} configurations stored")

    # Save comparison data
    mesh_cells = sim_prnn.V.mesh.geometry.dofmap
    qp_coords = sim_prnn.qp_coords
    np.savez(f"{output_folder}/ring_comparison.npz",
             prnn_coords=prnn_coords, rve_coords=rve_coords,
             prnn_F=prnn_F, rve_F=rve_F,
             prnn_vm_peak=prnn_vm_peak, rve_vm_peak=rve_vm_peak,
             mesh_cells=mesh_cells, qp_coords=qp_coords,
             x_best=x_best)
    print(f"\nSaved to {output_folder}/ring_comparison.npz")

    # Stress comparison
    print(f"\n{'='*60}")
    print("Peak von Mises stress comparison")
    print(f"{'='*60}")
    print(f"  {'Step':>6}  {'PRNN':>10}  {'RVE':>10}  {'Rel. err':>10}")
    n_common = min(len(prnn_vm_peak), len(rve_vm_peak))
    for i in range(1, n_common):
        rel_err = abs(prnn_vm_peak[i] - rve_vm_peak[i]) / rve_vm_peak[i] if rve_vm_peak[i] > 1e-12 else 0.0
        print(f"  {i:>6}  {prnn_vm_peak[i]:>10.4f}  {rve_vm_peak[i]:>10.4f}  {rel_err:>10.2%}")
    print(f"  {'Final':>6}  {prnn_vm_peak[n_common-1]:>10.4f}  {rve_vm_peak[n_common-1]:>10.4f}")
    print(f"{'='*60}")

    plot_stress_comparison(sim_prnn, prnn_vm, prnn_du, rve_vm, rve_du,
                          output_folder=output_folder, disp_scale=1,
                          labels=('PRNN', 'RVE'))
    plot_trajectories(prnn_coords, rve_coords, output_folder, mesh_cells)
    plot_displacement_error(prnn_coords, rve_coords, output_folder)

    return prnn_coords, rve_coords, prnn_F, rve_F


def plot_trajectories(prnn_coords, rve_coords, output_folder, mesh_cells=None):
    """Plot nodal displacement trajectories comparing PRNN and RVE."""
    n_dofs = prnn_coords.shape[1]
    fig, ax = plt.subplots(figsize=(4, 4))

    if mesh_cells is not None:
        ref_coords = prnn_coords[0]
        for cell in mesh_cells:
            cell_closed = list(cell) + [cell[0]]
            xs = [ref_coords[n, 0] for n in cell_closed]
            ys = [ref_coords[n, 1] for n in cell_closed]
            ax.plot(xs, ys, color='gray', linewidth=0.5, alpha=0.3, zorder=1)

    # skip = max(1, n_dofs // 100)
    skip = 1
    for i in range(0, n_dofs, skip):
        ax.plot(prnn_coords[:, i, 0], prnn_coords[:, i, 1], '-', c=colours[0], alpha=1.0, linewidth=0.8)
        ax.plot(rve_coords[:, i, 0], rve_coords[:, i, 1], '--', c=colours[1], alpha=1.0, linewidth=0.8)

    ax.scatter(prnn_coords[-1, ::skip, 0], prnn_coords[-1, ::skip, 1], c=colours[0], s=8, label='PRNN', zorder=5)
    ax.scatter(rve_coords[-1, ::skip, 0], rve_coords[-1, ::skip, 1], c=colours[1], s=8, label='RVE', zorder=5)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_folder}/ring_trajectories.pdf", bbox_inches='tight')
    plt.savefig(f"{output_folder}/ring_trajectories.png", bbox_inches='tight', dpi=300)
    print(f"Saved: {output_folder}/ring_trajectories")
    plt.close()


def plot_displacement_error(prnn_coords, rve_coords, output_folder):
    """Plot per-step displacement error (L2 norm) between PRNN and RVE."""
    n_common = min(prnn_coords.shape[0], rve_coords.shape[0])
    errors = []
    for i in range(1, n_common):
        diff = prnn_coords[i] - rve_coords[i]
        rve_disp = rve_coords[i] - rve_coords[0]
        rve_norm = np.linalg.norm(rve_disp)
        if rve_norm > 1e-12:
            errors.append(np.linalg.norm(diff) / rve_norm)
        else:
            errors.append(0.0)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(range(1, n_common), errors, 'o-', color=colours[0], markersize=4)
    ax.set_xlabel('Load step')
    ax.set_ylabel('Relative displacement error')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/ring_displacement_error.pdf", bbox_inches='tight')
    print(f"Saved: {output_folder}/ring_displacement_error.pdf")
    plt.close()


if __name__ == "__main__":
    run_comparison(
        results_file="../results/graded_ring/optimized/v4/graded/optimization_results.npz",
        n_ctrl=4,
        optimize_mu=True,
        macro_meshsize=0.03,  # 0.03
        rve_meshsize=0.0125,      # 0.0125?
        prnn_model_loc="../trained_models/train_vary_all_v2/prnn_nonlin_1L8_12m_sigmoid/samples512_run1",
        output_folder="../results/graded_ring/validate/v4/graded_debug_centraldiff",
        max_load=3.0,
        step_size_init=.2,
    )
