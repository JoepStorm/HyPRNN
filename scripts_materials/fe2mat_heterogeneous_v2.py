"""
Heterogeneous Parallel FE² Material Interface

Parallel FE² material where each quadrature point can have a different RVE
microstructure. Meshes are pre-generated globally once and passed to workers.
"""
import os
import glob
import traceback
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Pipe
from material_params import WOOD_MU, WOOD_LAMBDA, FUNGI_MU, FUNGI_LAMBDA
from mpi4py import MPI
from dolfinx.common import Timer
from dolfinx_materials.generic import Material, DataManager

from scripts_materials.rve_mesher import RVEMeshConfig
from scripts_materials.RVE_material import RVEMaterial


def _generate_single_mesh(args):
    """Generate a single RVE mesh and validate it. Returns file path for workers."""
    gauss_idx, config, material_properties, reuse_meshes = args
    savefolder = '../results/fe2_meshes'

    # Reuse path: skip regeneration if the mesh file from a previous run exists.
    # Caller is responsible for ensuring settings haven't changed.
    if reuse_meshes:
        msh_file = f"{savefolder}/rve_{gauss_idx}.msh"
        if os.path.exists(msh_file):
            a = config.base_radius * config.aspect_ratio
            b = config.base_radius / config.aspect_ratio
            domain_size = np.sqrt(config.num_fibers * np.pi * a * b / config.vfrac)
            rve = RVEMaterial(
                material_properties=material_properties,
                mesh_file=msh_file,
                domain_size=domain_size,
                comm=MPI.COMM_SELF
            )
            ndofs = len(rve.u_tilde.x.array)
            return (gauss_idx, ndofs, msh_file, domain_size)

    print(f"Creating mesh for idx: {gauss_idx}")
    for attempt in range(20):
        try:
            msh_file, domain_size = config.create_mesh_file(savefolder=savefolder, mesh_id=gauss_idx)
            # Validate mesh by creating RVE (checks periodic BCs, etc.)
            rve = RVEMaterial(
                material_properties=material_properties,
                mesh_file=msh_file,
                domain_size=domain_size,
                comm=MPI.COMM_SELF
            )
            ndofs = len(rve.u_tilde.x.array)
            return (gauss_idx, ndofs, msh_file, domain_size)
        except Exception as e:
            for ext in ('msh', 'geo'):
                path = f"{savefolder}/rve_{gauss_idx}.{ext}"
                if os.path.exists(path):
                    os.remove(path)
            print(f"Failed to create mesh for Gauss point {gauss_idx}, retrying (attempt {attempt+1})... \n material_properties: {material_properties}")
            print(f"  Real exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            config.seed += 1 + attempt
    raise RuntimeError(f"Failed to create mesh for Gauss point {gauss_idx} after 20 tries.")


# Worker Function (Runs permanently in background)
def _persistent_worker_loop(conn, worker_id):
    """
    Continuous loop for a worker process.
    1. Receives initialization data (meshes).
    2. Enters loop to solve RVEs upon request.
    """

    # Local cache for this specific worker
    rve_cache = {}

    # --- Initialization Phase ---
    # Receive the batch of RVE data assigned to this worker
    # Expects: { local_idx: (msh_file, domain_size, material_props, global_gauss_idx) }
    init_data = conn.recv()

    for local_idx, (msh_file, domain_size, mat_props, _) in init_data.items():
        # Load mesh directly from .msh file (preserves GMSH periodic structure)
        rve = RVEMaterial(
            material_properties=mat_props,
            mesh_file=msh_file,
            domain_size=domain_size,
            comm=MPI.COMM_SELF,
        )
        rve_cache[local_idx] = rve

    # Acknowledge readiness
    conn.send(True)

    # --- Integration Loop ---
    n_sub_increments = 8  # Number of sub-increments for retry  (delta F / n)

    while True:
        # Wait for command from main process
        command = conn.recv()

        if command is None:
            # Sentinel to exit
            break

        # Command is a batch of inputs: { local_idx: (F_matrix, F_prev, u_init) }
        inputs = command
        results = {}

        for local_idx, (F_target, F_prev, u_init) in inputs.items():
            rve = rve_cache[local_idx]

            # Apply previous state if available
            if u_init is not None:
                rve.u_tilde.x.array[:] = u_init

            # Solve with error handling
            try:
                P_matrix, _, C_full = rve.update_full(F_target, monitor_base_solve=False)
                u_new = rve.u_tilde.x.array.copy()
                results[local_idx] = (P_matrix, C_full, u_new, None)  # None = no error
            except Exception as e:
                # Try sub-incrementing before giving up
                try:
                    # Reset to initial state
                    if u_init is not None:
                        rve.u_tilde.x.array[:] = u_init

                    delta_F = F_target - F_prev
                    for sub_step in range(n_sub_increments):
                        alpha = (sub_step + 1) / n_sub_increments
                        F_sub = F_prev + alpha * delta_F

                        if sub_step < n_sub_increments - 1:
                            # Intermediate steps: stress only (faster)
                            P_matrix, _, converged = rve.update_stress(F_sub, monitor_base_solve=False)
                            if not converged:
                                raise RuntimeError(f"Sub-increment {sub_step+1}/{n_sub_increments} failed")
                        else:
                            # Final step: full solve with tangent
                            P_matrix, _, C_full = rve.update_full(F_sub, monitor_base_solve=False)

                    u_new = rve.u_tilde.x.array.copy()
                    results[local_idx] = (P_matrix, C_full, u_new, None)
                    print(f"Sub-incrementing succeeded to obtain convergence.")
                except Exception as sub_e:
                    # Sub-incrementing also failed
                    error_msg = f"Worker {worker_id}, local_idx {local_idx}: {type(e).__name__}: {e} (sub-increment also failed: {sub_e})"
                    results[local_idx] = (None, None, None, error_msg)

        # Send batch results back
        conn.send(results)

    # Cleanup
    conn.close()


class VariableDOFStateManager:
    """Manages RVE state for quadrature points with different DOF counts."""

    def __init__(self, ngauss: int):
        self.ngauss = ngauss
        self.ndofs_list = [0] * ngauss
        self.u_converged = [None] * ngauss
        self.u_trial = [None] * ngauss
        self.F_converged = [np.eye(2) for _ in range(ngauss)]

    def initialize_gauss_point(self, gauss_idx: int, ndofs: int):
        """Initialize storage for a specific Gauss point."""
        self.ndofs_list[gauss_idx] = ndofs
        self.u_converged[gauss_idx] = np.zeros(ndofs)
        self.u_trial[gauss_idx] = np.zeros(ndofs)

    def commit(self, F_new_list: list = None):
        """Accept trial state as converged."""
        for i in range(self.ngauss):
            if self.u_trial[i] is not None:
                self.u_converged[i] = self.u_trial[i].copy()
            if F_new_list is not None and F_new_list[i] is not None:
                self.F_converged[i] = F_new_list[i].copy()

    def revert(self):
        """Revert to last converged state."""
        for i in range(self.ngauss):
            if self.u_converged[i] is not None:
                self.u_trial[i] = self.u_converged[i].copy()

    def get_initial_guess(self, gauss_idx: int) -> np.ndarray:
        return self.u_trial[gauss_idx]

    def get_F_prev(self, gauss_idx: int) -> np.ndarray:
        return self.F_converged[gauss_idx]

    def store_solution(self, gauss_idx: int, u: np.ndarray):
        self.u_trial[gauss_idx] = u.copy()

class HeterogeneousFE2Material(Material):
    """
    Parallel FE² Material with per-quadrature-point microstructure.

    Each quadrature point can have a different RVE based on local volume fraction,
    fiber orientation, and aspect ratio. Meshes are created once and cached in
    workers for reuse across timesteps. Material parameters (mu, lambda) are also graded.
    """

    def __init__(
        self,
        graded_variables: dict,
        n_workers: int = None,
        rve_config: dict = None,
    ):
        super().__init__()
        self.graded_variables = graded_variables

        # Required graded variables
        required = ['vfrac', 'ratio', 'theta', 'mu_fungi', 'lambda_fungi', 'mu_wood', 'lambda_wood']
        for key in required:
            if key not in self.graded_variables:
                raise ValueError(f"Required graded variable '{key}' is missing from graded_variables.")

        self.material_properties = {}  # Empty for macro quadrature compatibility
        self.n_workers = n_workers if n_workers is not None else mp.cpu_count()
        self.meshsize = rve_config.get('meshsize', 0.0125) if rve_config else 0.0125
        self.verbose = rve_config.get('verbose', False) if rve_config else False
        self.plot_meshes = rve_config.get('plot_meshes', False) if rve_config else False
        self.gauss_coords = rve_config.get('gauss_coords', None) if rve_config else None
        self.reuse_meshes = rve_config.get('reuse_meshes', False) if rve_config else False

        # Build material properties for all Gauss points
        ngauss = len(graded_variables['vfrac'])
        self._material_properties_list = []
        for i in range(ngauss):
            props = {
                'fungi': {
                    'mu': float(graded_variables['mu_fungi'][i]),
                    'lambda_': float(graded_variables['lambda_fungi'][i]),
                    'tag': 1
                },
                'wood': {
                    'mu': float(graded_variables['mu_wood'][i]),
                    'lambda_': float(graded_variables['lambda_wood'][i]),
                    'tag': 2
                }
            }
            self._material_properties_list.append(props)

        self.workers = []
        self.pipes = []
        self.worker_assignments = [] # Maps global_gauss_idx -> (worker_idx, local_idx)

        print(f"HeterogeneousFE2Material: Using {self.n_workers} workers")

    @property
    def gradients(self):
        return {"F": 4}

    @property
    def fluxes(self):
        return {"PK1": 4}

    @property
    def internal_state_variables(self):
        return {}

    def set_data_manager(self, ngauss: int):
        """Initialize data storage and generate all meshes in parallel."""
        self.data_manager = DataManager(self, ngauss)
        self._state_manager = VariableDOFStateManager(ngauss)

        # Create worker pool
        ctx = mp.get_context('spawn')
        self._pool = ctx.Pool(processes=self.n_workers)

        # Prepare work items for parallel mesh generation
        work_items = []
        for i in range(ngauss):
            config = RVEMeshConfig(
                vfrac=float(self.graded_variables['vfrac'][i]),
                aspect_ratio=float(self.graded_variables['ratio'][i]),
                angle=float(self.graded_variables['theta'][i]),
                seed=i,
                meshsize=self.meshsize
            )
            work_items.append((i, config, self._material_properties_list[i], self.reuse_meshes))

        # Generate all meshes in parallel
        # Using map with chunking for deterministic worker assignment (better cache locality)
        if self.reuse_meshes:
            print(f"Reusing previously generated meshes (skipping deletion and regeneration).")
        else:
            print(f"Removing previous meshes...")
            for f in glob.glob("../results/fe2_meshes/*.geo") + glob.glob("../results/fe2_meshes/*.msh"):
                os.remove(f)

        print(f"Generating {ngauss} RVE meshes in parallel ({self.n_workers} workers)...")
        self._msh_files = [None] * ngauss
        self._domain_sizes = [None] * ngauss

        completed = 0
        for gauss_idx, ndofs, msh_file, domain_size in self._pool.imap_unordered(_generate_single_mesh, work_items):
            self._msh_files[gauss_idx] = msh_file
            self._domain_sizes[gauss_idx] = domain_size
            self._state_manager.initialize_gauss_point(gauss_idx, ndofs)
            completed += 1
            if self.verbose and completed % self.n_workers == 0:
                print(f"  Generated {completed}/{ngauss}")

        print(f"Generated {ngauss} meshes")
        self._pool.close()
        self._pool.join()

        if self.plot_meshes:
            self.plot_micro_meshes()

        # Distribute RVEs to Persistent Workers
        print(f"Initializing {self.n_workers} persistent workers...")

        # Split Gauss points into chunks for workers
        chunk_size = int(np.ceil(ngauss / self.n_workers))
        self.worker_assignments = [None] * ngauss

        worker_payloads = []

        for w_idx in range(self.n_workers):
            start = w_idx * chunk_size
            end = min(start + chunk_size, ngauss)

            # Prepare data dict for this worker
            worker_data = {}
            for global_idx in range(start, end):
                local_idx = global_idx - start # Simple local indexing

                # Store mapping for later retrieval
                self.worker_assignments[global_idx] = (w_idx, local_idx)

                worker_data[local_idx] = (
                    self._msh_files[global_idx],
                    self._domain_sizes[global_idx],
                    self._material_properties_list[global_idx],
                    global_idx
                )
            worker_payloads.append(worker_data)

        # Launch Processes
        # Initialize worker 0 first so FFCx JIT cache is populated before
        # the rest start compiling (avoids cache file race conditions that have caused errors).
        ctx = mp.get_context('spawn')
        for w_idx in range(self.n_workers):
            parent_conn, child_conn = Pipe()
            p = ctx.Process(
                target=_persistent_worker_loop,
                args=(child_conn, w_idx)
            )
            p.daemon = True  # workers exit when main exits
            p.start()
            self.workers.append(p)
            self.pipes.append(parent_conn)

            # Send Initialization Data (mesh data only transferred here)
            parent_conn.send(worker_payloads[w_idx])

            # Wait for the first worker to finish (warms JIT cache on disk)
            if w_idx == 0:
                ready = self.pipes[0].recv()
                if not ready:
                    raise RuntimeError("Worker 0 failed to initialize")
                # print("Worker 0 initialized (JIT cache warmed), starting remaining workers...")

        # Wait for remaining workers to finish initialization
        for pipe in self.pipes[1:]:
            ready = pipe.recv()
            if not ready:
                raise RuntimeError("Worker failed to initialize")

        print("Workers initialized and meshes cached.")


    def _load_mesh_data_for_plotting(self):
        """Load mesh data from .msh files for plotting."""
        import gmsh
        from dolfinx.io import gmsh as gmshio

        mesh_data_list = []
        for msh_file in self._msh_files:
            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 0)
            gmsh.open(msh_file)
            msh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=2)
            gmsh.finalize()

            mesh = msh_data.mesh
            cell_tags = msh_data.cell_tags
            topology = mesh.topology
            topology.create_connectivity(topology.dim, 0)
            c_to_v = topology.connectivity(topology.dim, 0)
            num_cells = topology.index_map(topology.dim).size_local

            mesh_data_list.append({
                'coords': mesh.geometry.x.copy(),
                'cells': np.array([c_to_v.links(i) for i in range(num_cells)]),
                'tag_values': cell_tags.values.copy(),
                'tag_indices': cell_tags.indices.copy(),
            })
        return mesh_data_list

    def plot_micro_meshes(self, output_folder='../results/micro_meshes', ncols=4, scale=0.1):
        """Plot all generated micro meshes."""
        if not self._msh_files or self._msh_files[0] is None:
            raise RuntimeError("No meshes to plot. Call set_data_manager first.")

        print("Plotting all micro meshes..")
        mesh_data_list = self._load_mesh_data_for_plotting()

        if self.gauss_coords is not None:
            from scripts_FEM.plotting_utils import plot_micro_meshes_at_gauss_points
            plot_micro_meshes_at_gauss_points(
                mesh_data_list, self.gauss_coords, output_folder, scale=scale)
        else:
            from scripts_FEM.plotting_utils import plot_micro_meshes_grid
            plot_micro_meshes_grid(mesh_data_list, output_folder, ncols=ncols)

    def _flat_to_matrix(self, F_flat: np.ndarray) -> np.ndarray:
        return np.array([[F_flat[0], F_flat[2]], [F_flat[3], F_flat[1]]])

    def _matrix_to_flat(self, P_mat: np.ndarray) -> np.ndarray:
        return np.array([P_mat[0, 0], P_mat[1, 1], P_mat[0, 1], P_mat[1, 0]])

    def integrate(self, gradients: np.ndarray, dt: float = 0.0):
        """Integrate constitutive response at all Gauss points in parallel."""
        with Timer("HeterogeneousFE2: Constitutive integration"):
            return self._integrate_impl(gradients, dt)

    def _integrate_impl(self, gradients: np.ndarray, dt: float = 0.0):
        ngauss = gradients.shape[0]

        # Store current F values for commit later
        self._current_F = [None] * ngauss

        # 1. Prepare Batches
        # We group inputs by worker_idx to send in 1 go
        worker_inputs = [{} for _ in range(self.n_workers)]

        for i in range(ngauss):
            w_idx, local_idx = self.worker_assignments[i]

            F_matrix = self._flat_to_matrix(gradients[i])
            F_prev = self._state_manager.get_F_prev(i)
            u_init = self._state_manager.get_initial_guess(i)
            self._current_F[i] = F_matrix

            worker_inputs[w_idx][local_idx] = (F_matrix, F_prev, u_init)

        # Send Tasks
        for w_idx in range(self.n_workers):
            if worker_inputs[w_idx]: # If worker has work
                self.pipes[w_idx].send(worker_inputs[w_idx])

        # Collect Results
        stresses = np.zeros((ngauss, 4))
        tangents = np.zeros((ngauss, 4, 4))
        failed_points = []

        for w_idx in range(self.n_workers):
            if worker_inputs[w_idx]:
                results = self.pipes[w_idx].recv()

                # Map back to global arrays
                for local_idx, res in results.items():
                    P_matrix, C_full, u_new, error = res

                    chunk_size = int(np.ceil(ngauss / self.n_workers))
                    global_idx = w_idx * chunk_size + local_idx

                    if error is not None:
                        # RVE solve failed - record failure
                        failed_points.append((global_idx, error))
                        # Use zero stress/tangent for failed point
                        stresses[global_idx] = np.zeros(4)
                        tangents[global_idx] = np.zeros((4, 4))
                        break
                    else:
                        stresses[global_idx] = self._matrix_to_flat(P_matrix)
                        tangents[global_idx] = C_full
                        self._state_manager.store_solution(global_idx, u_new)

        # If any RVE failed, raise exception to signal macro solver to reduce timestep
        if failed_points:
            error_msgs = [f"Gauss point {idx}: {err}" for idx, err in failed_points]
            raise RuntimeError(
                f"RVE solve failed at {len(failed_points)} point(s):\n  " +
                "\n  ".join(error_msgs)
            )

        self.data_manager.s1.fluxes[:] = stresses
        return stresses, np.zeros((ngauss, 0)), tangents

    def update_state(self):
        """Accept current trial state as converged."""
        self.data_manager.update()
        self._state_manager.commit(self._current_F if hasattr(self, '_current_F') else None)

    def revert_state(self):
        """Revert to last converged state."""
        self.data_manager.revert()
        self._state_manager.revert()

    def close(self):
        # Send shutdown signal
        for pipe in self.pipes:
            try:
                pipe.send(None)
            except:
                pass

        # Wait for workers to finish cleanly, force terminate if needed
        for p in self.workers:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)

        self.workers = []
        self.pipes = []

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


if __name__ == "__main__":
    print("Testing HeterogeneousFE2Material...")

    ngauss = 4

    # Define graded variables per Gauss point (geometry + material parameters)
    graded_variables = {
        'vfrac': np.array([0.2 + 0.1 * (i % 3) for i in range(ngauss)]),
        'ratio': np.ones(ngauss) * 1.25,
        'theta': np.array([np.pi / 6 * (i % 2) for i in range(ngauss)]),
        # Material parameters (Lamé constants) per phase
        # 'mu_fungi': np.ones(ngauss) * 38.46,      # E=100, nu=0.3 -> mu=38.46
        # 'lambda_fungi': np.ones(ngauss) * 57.69,   # E=100, nu=0.3 -> lambda=57.69
        # 'mu_wood': np.ones(ngauss) * 384.6,       # E=1000, nu=0.3 -> mu=384.6
        # 'lambda_wood': np.ones(ngauss) * 576.9,    # E=1000, nu=0.3 -> lambda=576.9
        'mu_fungi': np.ones(ngauss) * FUNGI_MU,
        'lambda_fungi': np.ones(ngauss) * FUNGI_LAMBDA,
        'mu_wood': np.ones(ngauss) * WOOD_MU,
        'lambda_wood': np.ones(ngauss) * WOOD_LAMBDA,
    }

    rve_config = {
        'meshsize': 0.0125,  # 0.04
        'verbose': True,
        'plot_meshes': False,
    }

    material = HeterogeneousFE2Material(
        graded_variables=graded_variables,
        n_workers=2,
        rve_config = rve_config,
    )

    material.set_data_manager(ngauss)

    material.plot_micro_meshes('../results/micro_meshes/fe2mat_test')


    # Test multiple timesteps with small incremental loading
    print("\nTimestep 1 (eps=0.01)...")
    F_test = np.tile([1.01, 0.99, 0.0, 0.0], (ngauss, 1))
    stresses1, _, _ = material.integrate(F_test)

    print("\nTimestep 2 (eps=0.02)...")
    F_test = np.tile([1.02, 0.98, 0.0, 0.0], (ngauss, 1))
    stresses2, _, _ = material.integrate(F_test)

    print(f"\nStresses timestep 1: {stresses1[0]}")
    print(f"Stresses timestep 2: {stresses2[0]}")

    material.close()
    print("\nDone!")
