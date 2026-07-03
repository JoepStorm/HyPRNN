"""
Compare multiple material models on the 3-point bending simulation with
a manually prescribed grading:

  * horizontal fibers at the top/bottom, vertical at mid-height,
  * vfrac high at the bottom, low at the top,
  * mu low on the outside of the beam, high on the inside (symmetric).

For each model we store nodal coordinates and (F, PK1) at every converged
step, write VTX output, and produce:
  * a visualization of the grading (explicit ellipses),
  * a single plot with the deformations of all models overlaid,
  * a FE^2 stress map (ground truth) and one error map per surrogate,
  * a timing / call-count summary using dolfinx.common.Timer.
        Note that the 1st model has extra overhead. For a fair comparison, do not count the first model being run.
"""
import os
import sys
import time
os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
from matplotlib.tri import Triangulation
plt.style.use(['science', 'bright'])
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, mesh
from dolfinx.common import Timer, timing
from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.solvers import NonlinearMaterialProblem
from dolfinx_materials.utils import create_quadrature_functionspace

from scripts_materials.prnn_material import PRNNMaterial
from scripts_materials.fe2_material import HeterogeneousFE2Material
from material_params import WOOD_MU, WOOD_LAMBDA, FUNGI_LAMBDA


# -- Simulation --------------------------------------------------------------

class ThreePointBendingSimulation:
    """3-point bending of a half-beam (symmetry plane at x = L/2).

    Pin support at (0, 0); symmetry BC (ux=0) on the right edge; downward
    traction strip at the top-right corner. Material is set through
    ``self.micro_variables`` before ``_setup_material_and_problem``.
    """

    def __init__(self,
                 beam_length=2.0,
                 beam_height=0.3,
                 macro_meshsize=0.05,
                 use_surrogate=True,
                 rve_config=None,
                 output_folder='../results/',
                 prnn_model_loc=''):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.dim = 2
        self.beam_length = beam_length
        self.beam_height = beam_height
        self.macro_meshsize = macro_meshsize
        self.use_surrogate = use_surrogate
        self.rve_config = rve_config
        self.output_folder = output_folder
        self.prnn_model_loc = prnn_model_loc

        self._setup_mesh()
        self._setup_function_spaces()
        self._setup_boundary_conditions()

        WJ = create_quadrature_functionspace(self.domain, self.deg_quad, self.order)
        self.qp_coords = WJ.tabulate_dof_coordinates()[:, :2]

    def _setup_mesh(self):
        half_length = self.beam_length / 2
        nx = max(int(half_length / self.macro_meshsize), 4)
        ny = max(int(self.beam_height / self.macro_meshsize), 2)
        self.domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [np.array([0.0, 0.0]), np.array([half_length, self.beam_height])],
            [nx, ny],
            cell_type=mesh.CellType.triangle,
        )
        self.dim = self.domain.topology.dim
        self.domain.topology.create_connectivity(self.dim - 1, self.dim)
        print(f"Beam mesh (half, symmetry): "
              f"{self.domain.topology.index_map(self.dim).size_global} elements")

    def _setup_function_spaces(self):
        self.order = 1
        self.deg_quad = 2 * (self.order - 1)
        self.V = fem.functionspace(self.domain, ("P", self.order, (2,)))
        self.num_elems = self.V.dofmap.list.shape[0]
        self.ips_per_elem = 1 if self.order == 1 else 3
        self.num_ips = self.num_elems * self.ips_per_elem

    def _setup_boundary_conditions(self):
        half_length = self.beam_length / 2
        support_tol = self.macro_meshsize * 0.6
        load_half_width = self.beam_length * 0.05

        def left_support(x):
            return np.logical_and(np.isclose(x[1], 0.0),
                                  np.isclose(x[0], 0.0, atol=support_tol))

        def right_edge(x):
            return np.isclose(x[0], half_length)

        left_verts = mesh.locate_entities_boundary(self.domain, 0, left_support)
        left_dofs_x = fem.locate_dofs_topological(self.V.sub(0), 0, left_verts)
        left_dofs_y = fem.locate_dofs_topological(self.V.sub(1), 0, left_verts)
        right_verts = mesh.locate_entities_boundary(self.domain, 0, right_edge)
        right_dofs_x = fem.locate_dofs_topological(self.V.sub(0), 0, right_verts)

        self.bcs = [
            fem.dirichletbc(0.0, left_dofs_x, self.V.sub(0)),
            fem.dirichletbc(0.0, left_dofs_y, self.V.sub(1)),
            fem.dirichletbc(0.0, right_dofs_x, self.V.sub(0)),
        ]

        fdim = self.domain.topology.dim - 1

        def top_right_facets(x):
            return np.logical_and(
                np.isclose(x[1], self.beam_height),
                np.abs(x[0] - half_length) < load_half_width,
            )

        load_facets = mesh.locate_entities_boundary(self.domain, fdim, top_right_facets)
        load_tags = np.ones(len(load_facets), dtype=np.int32)
        sorted_idx = np.argsort(load_facets)
        self.facet_tags = mesh.meshtags(self.domain, fdim,
                                        load_facets[sorted_idx], load_tags[sorted_idx])
        self.ds_load = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tags)
        self.load_magnitude = fem.Constant(self.domain, PETSc.ScalarType(0.0))

    def _tensor_to_vector_2D(self, T):
        return ufl.as_vector([T[0, 0], T[1, 1], T[0, 1], T[1, 0]])

    def _F(self, u):
        return self._tensor_to_vector_2D(ufl.Identity(self.dim) + ufl.grad(u))

    def _dF(self, u, v):
        return ufl.derivative(self._F(u), u, v)

    def _setup_material_and_problem(self):
        self.du = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        self.u = fem.Function(self.V, name="u")
        self.u_pre = fem.Function(self.V, name="u")

        if not self.use_surrogate:
            full_micro = {
                'vfrac': self.micro_variables['vfrac'],
                'theta': self.micro_variables['theta'],
                'ratio': self.micro_variables['ratio'],
                'mu_fungi': self.micro_variables['mu'],
                'lambda_fungi': self.micro_variables['lambda'],
                'mu_wood': jnp.ones(self.num_ips) * WOOD_MU,
                'lambda_wood': jnp.ones(self.num_ips) * WOOD_LAMBDA,
            }
            self.rve_config['gauss_coords'] = self.qp_coords
            self.material = HeterogeneousFE2Material(graded_variables=full_micro,
                                                     rve_config=self.rve_config)
        else:
            self.material = PRNNMaterial(f"{self.prnn_model_loc}_settings",
                                         f"{self.prnn_model_loc}_normparams",
                                         f"{self.prnn_model_loc}",
                                         self.micro_variables)

        self.qmap = QuadratureMap(self.domain, self.deg_quad, self.material)
        self.qmap.register_gradient("F", self._F(self.u))

        P = self.qmap.fluxes["PK1"]
        traction = ufl.as_vector([0.0, self.load_magnitude])
        self.Res = (ufl.dot(P, self._dF(self.u, self.v)) * self.qmap.dx
                    - ufl.dot(traction, self.v) * self.ds_load(1))
        self.Jac = self.qmap.derivative(self.Res, self.u, self.du)

    def _setup_solver(self):
        petsc_options = {
            "snes_monitor": None,
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_atol": 1e-5,
            "snes_max_it": 50,
            "ksp_type": "gmres",
            "ksp_rtol": 1e-8,
            "pc_type": "ilu",
            "pc_factor_levels": 1,
        }
        self.problem = NonlinearMaterialProblem(
            self.qmap, self.Res, self.u, bcs=self.bcs, J=self.Jac,
            petsc_options_prefix="BENDING", petsc_options=petsc_options,
        )

    def _destroy_problem(self):
        if hasattr(self, 'problem'):
            for attr in ('_snes', '_A', '_b', '_x', '_P_mat'):
                obj = getattr(self.problem, attr, None)
                if obj is not None:
                    obj.destroy()
            del self.problem

    def _create_micro_variable_functions(self):
        V_DG0 = fem.functionspace(self.domain, ("DG", 0))
        funcs = []
        for name, values in self.micro_variables.items():
            f = fem.Function(V_DG0, name=name)
            f.x.array[:] = np.asarray(values)
            funcs.append(f)
        return funcs


# -- Model definitions --------------------------------------------------------

MODEL_LABELS = {
    'prnn_1': 'PRNN TEST',
    'prnn': 'Linear PRNN (32)',
    'prnn_512': 'Linear PRNN (512)',
    'prnn_nonlin': 'Nonlinear PRNN (512)',
    'nn': 'NN (512)',
    'nn_32': 'NN (32)',
    # 'fe2': r'FE$^2$',
    'fe2': r'True',
}
MODEL_COLORS = {
    'prnn_1': colours[5],
    'prnn': colours[1],
    'prnn_512': colours[4],
    'prnn_nonlin': colours[0],
    'nn': colours[2],
    'nn_32': colours[3],
    # 'fe2': colours[3],
    'fe2': 'black',
}


def _model_color(key, i=0):
    return MODEL_COLORS.get(key, colours[i % len(colours)])


MODEL_LABELS_SHORT = {
    'prnn_1': 'Lin.PRNN\n(1)',
    'prnn': 'Lin.PRNN\n(32)',
    'prnn_512': 'Lin.PRNN\n(512)',
    'prnn_nonlin': 'Nonlin.PRNN',
    'nn': 'NN (512)',
    'nn_32': 'NN (32)',
    'fe2': r'FE$^2$',
}

MODELS = {
    'prnn_1': {
        'use_surrogate': True,
        # 'prnn_model_loc': '../trained_models/train_vfrac_ratio/prnn_lin_3L_6m/samples32_run0',
        'prnn_model_loc': '../trained_models/train_vary_all_v2/prnn_lin_1L_6m/samples32_run2',
        'timer_name': 'PRNN: Constitutive integration',
    },
    'prnn': {
        'use_surrogate': True,
        # 'prnn_model_loc': '../trained_models/train_vfrac_ratio/prnn_lin_3L_6m/samples32_run0',
        'prnn_model_loc': '../trained_models/train_vary_all_v2/prnn_lin_1L_6m/samples32_run2',
        'timer_name': 'PRNN: Constitutive integration',
    },
    'prnn_512': {
        'use_surrogate': True,
        # 'prnn_model_loc': '../trained_models/train_vfrac_ratio/prnn_lin_3L_6m/samples512_run0',
        'prnn_model_loc': '../trained_models/train_vary_all_v2/prnn_lin_1L_6m/samples512_run0',
        'timer_name': 'PRNN: Constitutive integration',
    },
    'prnn_nonlin': {
        'use_surrogate': True,
        # 'prnn_model_loc': '../trained_models/train_vfrac_ratio/prnn_nonlin_1L8_6m_sigmoid/samples512_run0',
        'prnn_model_loc': '../trained_models/train_vary_all_v2/prnn_nonlin_1L8_12m_sigmoid//samples512_run9',
        'timer_name': 'PRNN: Constitutive integration',
    },
    'nn': {
        'use_surrogate': True,
#         'prnn_model_loc': '../trained_models/train_vfrac_ratio/nn_64_3/samples512_run0',
        'prnn_model_loc': '../trained_models/train_vary_all_v2/nn_64_4/samples512_run8',
        'timer_name': 'PRNN: Constitutive integration',
    },
    'nn_32': {
        'use_surrogate': True,
        'prnn_model_loc': '../trained_models/train_vary_all_v2/nn_64_4/samples32_run7',
        'timer_name': 'PRNN: Constitutive integration',
    },
    'fe2': {
        'use_surrogate': False,
        'rve_config': {
            'meshsize': 0.0125,
            'verbose': False,
            'plot_meshes': False,
        },
        'timer_name': 'HeterogeneousFE2: Constitutive integration',
    },
}


# -- Grading ------------------------------------------------------------------

def apply_manual_grading(sim):
    """Prescribe a grading on sim's quadrature points.

    Half-beam: x in [0, L/2] (symmetry plane at x=L/2, support at x=0),
               y in [0, H]   (bottom at y=0, top at y=H).
    """
    L = sim.beam_length
    H = sim.beam_height
    half_L = L / 2
    x = sim.qp_coords[:, 0]
    y = sim.qp_coords[:, 1]

    # Fiber orientation: three equal horizontal bands.
    # bottom third: horizontal (theta=0), middle third: vertical (pi/2),
    # top third: horizontal (theta=0).
    in_mid = (y >= H / 3) & (y < 2 * H / 3)
    theta = np.where(in_mid, np.pi / 2, 0.0)

    # Aspect ratio: 2.25 in top/bottom bands, 1.25 in the middle.
    ratio = np.where(in_mid, 1.25, 2.25)

    # Volume fraction: high at bottom (y=0), low at top (y=H).
    vfrac = 0.5 - 0.4 * (y / H)

    # mu: symmetric about x=half_L in the full beam.
    # Half-beam represents left half (x=0 is the outside, x=half_L is the inside / center).
    mu_low, mu_high = 0.2, 1.2
    mu = mu_low + (mu_high - mu_low) * (x / half_L)
    lam = FUNGI_LAMBDA * np.ones_like(x)

    sim.micro_variables = {
        'vfrac': jnp.array(vfrac),
        'theta': jnp.array(theta),
        'ratio': jnp.array(ratio),
        'mu': jnp.array(mu),
        'lambda': jnp.array(lam),
    }
    sim.output_name = "manual_graded_bending"


def plot_manual_grading(sim, output_folder):
    """Plot the prescribed grading (left half only): vfrac | fibers | mu.

    Right edge is drawn as a dashed line to indicate the symmetry plane
    (the structure continues mirrored beyond it, just not visualized).
    """
    os.makedirs(output_folder, exist_ok=True)
    H = sim.beam_height
    half_L = sim.beam_length / 2

    domain = sim.domain
    domain.topology.create_connectivity(domain.topology.dim, 0)
    c_to_v = domain.topology.connectivity(domain.topology.dim, 0)
    n_cells = domain.topology.index_map(domain.topology.dim).size_local
    nodes = domain.geometry.x[:, :2]
    elems = np.array([c_to_v.links(c)[:3] for c in range(n_cells)])
    tri = Triangulation(nodes[:, 0], nodes[:, 1], elems)

    vfrac = np.asarray(sim.micro_variables['vfrac'])
    mu = np.asarray(sim.micro_variables['mu'])
    theta = np.asarray(sim.micro_variables['theta'])
    ratio = np.asarray(sim.micro_variables['ratio'])

    # Tall-thin panels (L/2 by H per panel). Using constrained_layout for
    # tight margins; cax via make_axes_locatable so all three axes keep the
    # same physical width regardless of colorbar presence.
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.0),
                             constrained_layout=True)

    def _add_cbar(ax, mappable):
        divider = make_axes_locatable(ax)
        cax_slot = divider.append_axes("right", size="2%", pad=0.05)
        cax_slot.axis('off')
        # Vertically shrink cbar to 70% of panel height.
        cax = cax_slot.inset_axes([0.0, 0.15, 1.0, 0.7])
        fig.colorbar(mappable, cax=cax)
        # Extra spacer so the cbar doesn't butt up against the next panel.
        spacer = divider.append_axes("right", size="6%", pad=0.0)
        spacer.axis('off')

    def _frame(ax):
        # Solid border on bottom, left, top; dashed on the right = symmetry plane.
        ax.plot([0, half_L], [0, 0], color='k', lw=0.8)       # bottom
        ax.plot([0, 0], [0, H], color='k', lw=0.8)            # left
        ax.plot([0, half_L], [H, H], color='k', lw=0.8)       # top
        ax.plot([half_L, half_L], [0, H], color='k', lw=0.8, ls='--')  # symmetry
        ax.set_xlim(-0.01 * half_L, 1.01 * half_L)
        ax.set_ylim(-0.03 * H, 1.03 * H)
        ax.set_aspect('equal')
        ax.axis('off')

    # Panel 1: vfrac — cividis_r (ring convention)
    cmap_vf = mpl.colormaps['cividis_r']
    norm_vf = mpl.colors.Normalize(vmin=0.1, vmax=0.5)
    m1 = axes[0].tripcolor(tri, facecolors=vfrac, cmap=cmap_vf, norm=norm_vf,
                           edgecolors='none')
    _add_cbar(axes[0], m1)
    axes[0].set_title(r'Volume fraction $V_f$', fontsize=9, pad=2)

    # Panel 2: ellipses (theta, ratio), colored by ratio using 'Blues'.
    qp = sim.qp_coords
    cmap_r = mpl.colormaps['Blues']
    norm_r = mpl.colors.Normalize(vmin=1.0, vmax=2.5)
    base = min(H, half_L) * 0.04
    n_skip = max(1, len(qp) // 250)
    patches = [
        Ellipse(xy=(xi, yi), width=rv * base, height=base,
                angle=np.rad2deg(ang))
        for (xi, yi), rv, ang in zip(qp[::n_skip], ratio[::n_skip], theta[::n_skip])
    ]
    pc = PatchCollection(patches, cmap=cmap_r, norm=norm_r,
                         edgecolor='k', linewidth=0.15, alpha=0.9)
    pc.set_array(ratio[::n_skip])
    axes[1].add_collection(pc)
    _add_cbar(axes[1], pc)
    axes[1].set_title('Fiber orientation and shape', fontsize=9, pad=2)

    # Panel 3: mu — Greens (ring convention)
    cmap_mu = mpl.colormaps['Greens']
    norm_mu = mpl.colors.Normalize(vmin=float(mu.min()), vmax=float(mu.max()))
    m3 = axes[2].tripcolor(tri, facecolors=mu, cmap=cmap_mu, norm=norm_mu,
                           edgecolors='none')
    _add_cbar(axes[2], m3)
    axes[2].set_title(r'Shear modulus $\mu$', fontsize=9, pad=2)

    for ax in axes:
        _frame(ax)

    plt.savefig(f"{output_folder}/grading.pdf", bbox_inches='tight')
    plt.savefig(f"{output_folder}/grading.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved grading visualisation to {output_folder}/grading")


# -- Core routines ------------------------------------------------------------

def _create_dg0_tensor_functions(sim):
    from dolfinx import fem
    V_DG0 = fem.functionspace(sim.domain, ("DG", 0))
    pk1_funcs = [fem.Function(V_DG0, name=n) for n in ("PK1_11", "PK1_22", "PK1_12", "PK1_21")]
    F_funcs = [fem.Function(V_DG0, name=n) for n in ("F_11", "F_22", "F_12", "F_21")]
    return pk1_funcs, F_funcs


def _update_dg0_tensor_functions(pk1_funcs, F_funcs, pk1_vals, F_vals):
    for i, f in enumerate(pk1_funcs):
        f.x.array[:] = pk1_vals[:, i]
    for i, f in enumerate(F_funcs):
        f.x.array[:] = F_vals[:, i]


def get_F_at_quadrature_points(sim):
    F_grad = sim.qmap.gradients["F"]
    return sim.qmap.get_gradient_vals(F_grad, sim.qmap.cells)


def get_PK1_at_quadrature_points(sim):
    return sim.qmap.material.data_manager.s1.fluxes.copy()


def run_and_store(sim, output_folder, output_name, timer_name,
                  max_load, step_size_init,
                  step_size_factor=0.5, step_size_min=0.1):
    """Run with adaptive load stepping. Returns coords/F/PK1 histories + timing stats.

    Timing split (all wall time in seconds):
      - setup_wall: material + solver setup (includes RVE mesh creation for FE^2)
      - loop_wall: load-step loop (solves + retries + I/O), excludes setup
      - total_wall: setup_wall + loop_wall, true end-to-end
      - step_first_wall: wall time of the first successful solve
      - step_rest_wall / n_steps_rest: remaining successful solves
      - constitutive_wall / n_calls: dolfinx internal constitutive-integration timer
    """
    setup_t0 = time.perf_counter()
    sim._setup_material_and_problem()
    sim._setup_solver()
    setup_wall = time.perf_counter() - setup_t0

    ref_coords = sim.V.tabulate_dof_coordinates()[:, :2]
    coords_history = [ref_coords.copy()]
    F_history = [get_F_at_quadrature_points(sim).copy()]
    PK1_history = [get_PK1_at_quadrature_points(sim)]

    cur_step_size = step_size_init
    cur_load = 0.0
    converged_steps = 0
    converged = True
    step_first_wall = 0.0
    step_rest_wall = 0.0
    n_steps_rest = 0

    # Capture cumulative timing before this run to isolate per-model stats.
    def _t(name):
        try:
            t = timing(name)
        except RuntimeError:
            return 0, 0.0
        if t is None:
            return 0, 0.0
        count, wall = t[0], t[1]
        wall_s = wall.total_seconds() if hasattr(wall, 'total_seconds') else float(wall)
        return count, wall_s

    pre_count, pre_wall = _t(timer_name)

    pk1_funcs, F_funcs = _create_dg0_tensor_functions(sim)
    all_funcs = [sim.u] + sim._create_micro_variable_functions() + pk1_funcs + F_funcs
    with Timer(f"Validate: {output_name} loop"), \
         io.VTXWriter(sim.comm, f"{output_folder}/{output_name}.bp", all_funcs) as xf:
        xf.write(0.0)

        while abs(cur_load) < max_load - 1e-10:
            if converged:
                sim.u_pre.x.array[:] = sim.u.x.array[:]
                converged_steps += 1
                cur_load += cur_step_size
            else:
                sim.u.x.array[:] = sim.u_pre.x.array[:]
                sim._destroy_problem()
                sim._setup_solver()
                print(f"Load {abs(cur_load + cur_step_size):.2f} did not converge, reducing step size.")
                cur_step_size *= step_size_factor
                cur_load += cur_step_size

            if abs(cur_step_size) < abs(step_size_min):
                print("Step size too small, stopping simulation.")
                break

            sim.load_magnitude.value = cur_load

            solve_t0 = time.perf_counter()
            try:
                sim.problem.solve()
                converged = sim.problem.solver.getConvergedReason() > 0
                nr_iters = sim.problem.solver.getIterationNumber()
                print(f"Step {converged_steps}: load={abs(cur_load):.4f}, converged={converged}, iters={nr_iters}")
            except Exception:
                print("Error while solving.")
                converged = False
            solve_wall = time.perf_counter() - solve_t0

            if converged:
                if step_first_wall == 0.0:
                    step_first_wall = solve_wall
                else:
                    step_rest_wall += solve_wall
                    n_steps_rest += 1

                u_array = sim.u.x.array.reshape(-1, 2)
                coords_history.append((ref_coords + u_array).copy())
                F_vals = get_F_at_quadrature_points(sim).copy()
                pk1_vals = get_PK1_at_quadrature_points(sim)
                F_history.append(F_vals)
                PK1_history.append(pk1_vals)
                _update_dg0_tensor_functions(pk1_funcs, F_funcs, pk1_vals, F_vals)
                xf.write(abs(cur_load))
            else:
                cur_load -= cur_step_size

    post_count, post_wall = _t(timer_name)
    _, loop_wall = _t(f"Validate: {output_name} loop")
    stats = {
        'n_calls': post_count - pre_count,
        'constitutive_wall': post_wall - pre_wall,
        'setup_wall': setup_wall,
        'loop_wall': loop_wall,
        'total_wall': setup_wall + loop_wall,
        'step_first_wall': step_first_wall,
        'step_rest_wall': step_rest_wall,
        'n_steps_rest': n_steps_rest,
    }
    return np.array(coords_history), np.array(F_history), np.array(PK1_history), stats


# -- Driver -------------------------------------------------------------------

def run_comparison(
    model_keys=None,
    max_load=0.05,
    step_size_init=-0.01,
    step_size_min=-0.0001,
    macro_meshsize=0.05,
    beam_length=2.0,
    beam_height=0.3,
    output_folder="../results/bending_model_comparison",
):
    os.makedirs(output_folder, exist_ok=True)
    if model_keys is None:
        model_keys = list(MODELS.keys())

    print("=" * 60)
    print(f"Bending: multi-model comparison, models={model_keys}")
    print("=" * 60)

    results = {}
    timings = {}
    mesh_cells = None
    first_sim = None

    for key in model_keys:
        print(f"\n--- Running {key} ---")
        cfg = MODELS[key]

        sim = ThreePointBendingSimulation(
            beam_length=beam_length,
            beam_height=beam_height,
            macro_meshsize=macro_meshsize,
            use_surrogate=cfg.get('use_surrogate', True),
            rve_config=cfg.get('rve_config'),
            output_folder=output_folder + "/",
            prnn_model_loc=cfg.get('prnn_model_loc', ''),
        )
        apply_manual_grading(sim)

        if first_sim is None:
            plot_manual_grading(sim, output_folder)
            first_sim = sim

        if cfg.get('rve_config'):
            sim.rve_config['gauss_coords'] = sim.qp_coords

        coords, F, PK1, stats = run_and_store(
            sim, output_folder, key, cfg['timer_name'],
            max_load=max_load, step_size_init=step_size_init,
            step_size_min=step_size_min,
        )
        results[key] = {'coords': coords, 'F': F, 'PK1': PK1}
        timings[key] = stats
        print(f"{key}: stored {coords.shape[0]} configurations")
        avg_rest = (stats['step_rest_wall'] / stats['n_steps_rest']
                    if stats['n_steps_rest'] > 0 else 0.0)
        print(f"  total={stats['total_wall']:.2f}s "
              f"(setup={stats['setup_wall']:.2f}s + loop={stats['loop_wall']:.2f}s), "
              f"constitutive={stats['constitutive_wall']:.2f}s over {stats['n_calls']} calls")
        print(f"  1st step={stats['step_first_wall']:.2f}s, "
              f"rest={stats['step_rest_wall']:.2f}s over {stats['n_steps_rest']} steps "
              f"(avg {avg_rest:.2f}s/step)")

        if mesh_cells is None:
            mesh_cells = sim.V.mesh.geometry.dofmap

    # -- save (everything needed to recreate plots without re-running) --
    first_sim.domain.topology.create_connectivity(first_sim.domain.topology.dim, 0)
    c_to_v = first_sim.domain.topology.connectivity(first_sim.domain.topology.dim, 0)
    n_cells = first_sim.domain.topology.index_map(first_sim.domain.topology.dim).size_local
    mesh_nodes = first_sim.domain.geometry.x[:, :2].copy()
    mesh_elems = np.array([c_to_v.links(c)[:3] for c in range(n_cells)])

    save_dict = {
        'mesh_cells': np.asarray(mesh_cells),
        'mesh_nodes': mesh_nodes,
        'mesh_elems': mesh_elems,
        'qp_coords': first_sim.qp_coords,
        'beam_length': first_sim.beam_length,
        'beam_height': first_sim.beam_height,
        'micro_vfrac': np.asarray(first_sim.micro_variables['vfrac']),
        'micro_mu': np.asarray(first_sim.micro_variables['mu']),
        'micro_theta': np.asarray(first_sim.micro_variables['theta']),
        'micro_ratio': np.asarray(first_sim.micro_variables['ratio']),
        'model_keys': np.array(list(results.keys())),
    }
    for key, data in results.items():
        save_dict[f'{key}_coords'] = data['coords']
        save_dict[f'{key}_F'] = data['F']
        save_dict[f'{key}_PK1'] = data['PK1']
    np.savez(f"{output_folder}/comparison_data.npz", **save_dict)
    print(f"\nSaved to {output_folder}/comparison_data.npz")

    _write_timings_csv(timings, output_folder)
    plot_trajectories(results, output_folder, mesh_cells)
    plot_final_deformation(results, output_folder, mesh_cells)
    plot_final_deformation_separate(results, output_folder, mesh_cells)
    plot_stress_maps(results, first_sim, output_folder)
    plot_stress_strain_paths(results, output_folder, indices=_default_path_indices(first_sim.qp_coords))

    return results, timings


def _write_timings_csv(timings, output_folder):
    path = f"{output_folder}/timings.csv"
    with open(path, 'w') as f:
        f.write("model,total_wall_s,setup_wall_s,loop_wall_s,"
                "step_first_wall_s,step_rest_wall_s,n_steps_rest,"
                "constitutive_wall_s,n_constitutive_calls\n")
        for k, s in timings.items():
            f.write(f"{k},{s['total_wall']:.4f},{s['setup_wall']:.4f},"
                    f"{s['loop_wall']:.4f},{s['step_first_wall']:.4f},"
                    f"{s['step_rest_wall']:.4f},{s['n_steps_rest']},"
                    f"{s['constitutive_wall']:.4f},{s['n_calls']}\n")
    print(f"Saved timings to {path}")


# -- Plotting -----------------------------------------------------------------

def plot_trajectories(results, output_folder, mesh_cells=None):
    """All models overlaid: reference mesh + deformed node trajectories."""
    fig, ax = plt.subplots(figsize=(7, 3))
    first_key = next(iter(results))
    ref_coords = results[first_key]['coords'][0]

    if mesh_cells is not None:
        for cell in mesh_cells:
            closed = list(cell) + [cell[0]]
            ax.plot(ref_coords[closed, 0], ref_coords[closed, 1],
                    color='gray', linewidth=0.3, alpha=0.3, zorder=1)

    n_dofs = ref_coords.shape[0]
    skip = max(1, n_dofs // 80)

    for i, (key, data) in enumerate(results.items()):
        coords = data['coords']
        style = ['-', '--', '-.', ':'][i % 4]
        c = _model_color(key, i)
        for j in range(0, n_dofs, skip):
            ax.plot(coords[:, j, 0], coords[:, j, 1], style,
                    c=c, alpha=0.7, linewidth=0.6)
        ax.scatter(coords[-1, ::skip, 0], coords[-1, ::skip, 1],
                   c=c, s=6, label=MODEL_LABELS.get(key, key), zorder=5)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='lower left', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/deformation_overlay.pdf", bbox_inches='tight')
    plt.savefig(f"{output_folder}/deformation_overlay.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved deformation overlay to {output_folder}/deformation_overlay")


def plot_final_deformation(results, output_folder, mesh_cells):
    """Overlay the deformed mesh (final step) of each model as line drawings."""
    if mesh_cells is None:
        return
    fig, ax = plt.subplots(figsize=(7, 3))

    first_key = next(iter(results))
    ref = results[first_key]['coords'][0]
    # Reference mesh in light gray.
    for cell in mesh_cells:
        closed = list(cell) + [cell[0]]
        ax.plot(ref[closed, 0], ref[closed, 1],
                color='lightgray', linewidth=0.4, alpha=0.8, zorder=1)

    for i, (key, data) in enumerate(results.items()):
        coords_final = data['coords'][-1]
        c = _model_color(key, i)
        # Plot one invisible handle for the legend, then all edges.
        ax.plot([], [], color=c, linewidth=0.9,
                label=MODEL_LABELS.get(key, key))
        for cell in mesh_cells:
            closed = list(cell) + [cell[0]]
            ax.plot(coords_final[closed, 0], coords_final[closed, 1],
                    color=c, linewidth=0.6, alpha=0.85, zorder=2 + i)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='lower left', fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/final_deformation.pdf", bbox_inches='tight')
    plt.savefig(f"{output_folder}/final_deformation.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved final deformation overlay to {output_folder}/final_deformation")


def plot_final_deformation_separate(results, output_folder, mesh_cells):
    """One subplot per surrogate, fixed 2x3 layout.

    Layout:
        [ fe2,         prnn (lin),   nn_32      ]
        [ prnn_nonlin, prnn_512,     nn_512   ]
    Empty cells (missing keys) are left blank.
    """
    if mesh_cells is None:
        return
    if 'fe2' not in results:
        print("No FE2 results; skipping separate final deformation plots.")
        return

    fe2_final = results['fe2']['coords'][-1]
    ref = results[next(iter(results))]['coords'][0]

    layout = [
        ['fe2',         'prnn',     'nn_32'],
        ['prnn_nonlin', 'prnn_512', 'nn'],
    ]
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.01 * n_rows), squeeze=False)

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            key = layout[r][c]
            if key not in results:
                ax.axis('off')
                continue

            for cell in mesh_cells:
                closed = list(cell) + [cell[0]]
                ax.plot(ref[closed, 0], ref[closed, 1], color='lightgray', linewidth=0.3, alpha=0.6, zorder=1)
            if key != 'fe2':
                for cell in mesh_cells:
                    closed = list(cell) + [cell[0]]
                    ax.plot(fe2_final[closed, 0], fe2_final[closed, 1], color='gray', linewidth=0.6, alpha=0.85, zorder=2)

            coords_final = results[key]['coords'][-1]
            col = _model_color(key)
            for cell in mesh_cells:
                closed = list(cell) + [cell[0]]
                ax.plot(coords_final[closed, 0], coords_final[closed, 1], color=col, linewidth=0.6, alpha=0.9, zorder=3)

            ax.set_title(MODEL_LABELS.get(key, key), fontsize=10, pad=-0.07)
            ax.set_aspect('equal')
            ax.set_anchor('N')
            ax.axis('off')

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.02, hspace=-0.3)
    plt.savefig(f"{output_folder}/final_deformation_separate.pdf", bbox_inches='tight', pad_inches=0.0)
    plt.savefig(f"{output_folder}/final_deformation_separate.png", bbox_inches='tight', dpi=300, pad_inches=0.02)
    plt.close()
    print(f"Saved separate final deformations to {output_folder}/final_deformation_separate")


def _build_triangulation(sim):
    domain = sim.domain
    domain.topology.create_connectivity(domain.topology.dim, 0)
    c_to_v = domain.topology.connectivity(domain.topology.dim, 0)
    n_cells = domain.topology.index_map(domain.topology.dim).size_local
    nodes = domain.geometry.x[:, :2]
    elems = np.array([c_to_v.links(c)[:3] for c in range(n_cells)])
    return Triangulation(nodes[:, 0], nodes[:, 1], elems), n_cells


def plot_stress_maps(results, tri_or_sim, output_folder):
    """FE^2 stress as ground truth; one error map per surrogate.

    ``tri_or_sim`` can be either a sim (with .domain) or a
    (Triangulation, n_cells) tuple — convenient for replotting from npz.
    """
    if 'fe2' not in results:
        print("No FE2 results; skipping stress maps.")
        return

    if isinstance(tri_or_sim, tuple):
        tri, n_cells = tri_or_sim
    else:
        tri, n_cells = _build_triangulation(tri_or_sim)
    labels = [r'$P_{11}$', r'$P_{22}$', r'$P_{12}$', r'$P_{21}$']

    pk1_fe2 = results['fe2']['PK1'][-1]  # (n_cells, 4)
    if pk1_fe2.shape[0] != n_cells:
        # Multiple IPs per cell - average to per-cell.
        ips_per_cell = pk1_fe2.shape[0] // n_cells
        pk1_fe2 = pk1_fe2.reshape(n_cells, ips_per_cell, 4).mean(axis=1)

    # --- collect surrogate predictions ---
    surrogates = {}
    for key, data in results.items():
        if key == 'fe2':
            continue
        pk1_s = data['PK1'][-1]
        if pk1_s.shape[0] != n_cells:
            ips_per_cell = pk1_s.shape[0] // n_cells
            pk1_s = pk1_s.reshape(n_cells, ips_per_cell, 4).mean(axis=1)
        surrogates[key] = pk1_s

    n_rows = 1 + len(surrogates)
    stress_cmap = mpl.colormaps['Spectral_r']
    err_cmap = mpl.colormaps['coolwarm']

    def _draw(ax, field, norm, cmap):
        ax.tripcolor(tri, facecolors=field, cmap=cmap, norm=norm, edgecolors='none')
        ax.set_aspect('equal')
        ax.axis('off')

    def _row_label(ax, text):
        ax.text(-0.05, 0.5, text, transform=ax.transAxes, rotation=90,
                ha='right', va='center', fontsize=10)

    # --- figure sizing from mesh aspect ratio ---
    xmin, xmax = tri.x.min(), tri.x.max()
    ymin, ymax = tri.y.min(), tri.y.max()
    aspect = (ymax - ymin) / max(xmax - xmin, 1e-12)
    cell_w = 3.0
    cell_h = max(cell_w * aspect, 0.6)

    # --- Combined error plot: per-row shared cmap (one colorbar per row) ---
    fig, axes = plt.subplots(n_rows, 4,
                             figsize=(4 * cell_w + 1.0, n_rows * cell_h),
                             squeeze=False)
    fe2_vmax = [np.max(np.abs(pk1_fe2[:, c])) for c in range(4)]

    for comp in range(4):
        ax = axes[0, comp]
        norm = mpl.colors.Normalize(vmin=-fe2_vmax[comp], vmax=fe2_vmax[comp])
        _draw(ax, pk1_fe2[:, comp], norm, stress_cmap)
        ax.set_title(labels[comp], fontsize=10)
    _row_label(axes[0, 0], r'FE$^2$')

    per_row_norms = [None]
    for r, (key, pk1_s) in enumerate(surrogates.items(), start=1):
        err = pk1_s - pk1_fe2
        v = max(np.max(np.abs(err)), 1e-12)
        norm = mpl.colors.Normalize(vmin=-v, vmax=v)
        per_row_norms.append(norm)
        for comp in range(4):
            _draw(axes[r, comp], err[:, comp], norm, err_cmap)
        _row_label(axes[r, 0], f'{key} err')

    fig.subplots_adjust(left=0.05, right=0.93, top=0.93, bottom=0.03, wspace=0.05, hspace=0.08)

    # FE2 row colorbar (using last component's norm is misleading; give one per row)
    pos0 = axes[0, -1].get_position()
    cax0 = fig.add_axes([pos0.x1 + 0.005, pos0.y0 + 0.02, 0.01, pos0.height - 0.04])
    fe2_global = max(fe2_vmax)
    fe2_norm = mpl.colors.Normalize(vmin=-fe2_global, vmax=fe2_global)
    fig.colorbar(mpl.cm.ScalarMappable(norm=fe2_norm, cmap=stress_cmap), cax=cax0)

    for r in range(1, n_rows):
        if r != 2:
            continue
        pos = axes[r, -1].get_position()
        cax = fig.add_axes([pos0.x1 + 0.005, pos0.y0 + 0.02, 0.01, pos0.height - 0.04])
        fig.colorbar(mpl.cm.ScalarMappable(norm=per_row_norms[r], cmap=err_cmap), cax=cax)

    # fig.suptitle(r'FE$^2$ PK1 vs surrogate errors (final step)', fontsize=11, y=1.05)
    plt.savefig(f"{output_folder}/stress_errors_combined.pdf", bbox_inches='tight')
    plt.savefig(f"{output_folder}/stress_errors_combined.png", bbox_inches='tight', dpi=300)
    plt.close()

    # --- Combined prediction plot: per-component shared cmap across all rows ---
    fig, axes = plt.subplots(n_rows, 4, figsize=(4 * cell_w + 1.0, n_rows * cell_h + 0.6), squeeze=False)
    comp_vmax = []
    # Exclude nn_32 from the color range — its values can be extreme and
    # would saturate the cmap for every other model.
    surrogates_datarange = {k: v for k, v in surrogates.items() if k != 'nn_32'}
    for comp in range(4):
        v = np.max(np.abs(pk1_fe2[:, comp]))
        for pk1_s in surrogates_datarange.values():
            v = max(v, np.max(np.abs(pk1_s[:, comp])))
        comp_vmax.append(max(v, 1e-12))
    comp_norms = [mpl.colors.Normalize(vmin=-v, vmax=v) for v in comp_vmax]

    for comp in range(4):
        _draw(axes[0, comp], pk1_fe2[:, comp], comp_norms[comp], stress_cmap)
        axes[0, comp].set_title(labels[comp], fontsize=10)


    _row_label(axes[0, 0], MODEL_LABELS_SHORT['fe2'])

    for r, (key, pk1_s) in enumerate(surrogates.items(), start=1):
        for comp in range(4):
            _draw(axes[r, comp], pk1_s[:, comp], comp_norms[comp], stress_cmap)
        _row_label(axes[r, 0], MODEL_LABELS_SHORT[key])


    fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.10, wspace=0.05, hspace=0.08)
    for comp in range(4):
        pos = axes[-1, comp].get_position()
        cax = fig.add_axes([pos.x0 + 0.025, max(pos.y0 - 0.05, 0.02), pos.width - 0.05, 0.012])
        fig.colorbar(mpl.cm.ScalarMappable(norm=comp_norms[comp], cmap=stress_cmap), cax=cax, orientation='horizontal')

    # fig.suptitle(r'FE$^2$ vs surrogate PK1 predictions (final step)', fontsize=11)
    plt.savefig(f"{output_folder}/stress_predictions_combined.pdf", bbox_inches='tight')
    plt.savefig(f"{output_folder}/stress_predictions_combined.png", bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved combined stress maps to {output_folder}/stress_errors_combined and stress_predictions_combined")


# -- Replotting from saved data ----------------------------------------------

def _default_path_indices(qp_coords, n=6):
    """Pick a spread of quadrature-point indices spanning the beam (for plots)."""
    qp = np.asarray(qp_coords)
    x = qp[:, 0]; y = qp[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    targets = [
        (xmin + 0.1 * (xmax - xmin), ymax),
        (xmin + 0.1 * (xmax - xmin), ymin),
        (0.5 * (xmin + xmax), ymax),
        (0.5 * (xmin + xmax), ymin),
        (xmin + 0.9 * (xmax - xmin), ymax),
        (xmin + 0.9 * (xmax - xmin), ymin),
    ][:n]
    idx = []
    for tx, ty in targets:
        d = (x - tx) ** 2 + (y - ty) ** 2
        idx.append(int(np.argmin(d)))
    return idx


def plot_stress_strain_paths(results, output_folder, indices,
                             components=(0, 1, 2, 3), filename='stress_strain_paths'):
    """Plot F vs PK1 curves at each selected quadrature-point index for all models.

    Layout: rows = selected points, cols = components (11, 22, 12, 21).
    """
    comp_labels = {0: (r'$F_{11}$', r'$P_{11}$'),
                   1: (r'$F_{22}$', r'$P_{22}$'),
                   2: (r'$F_{12}$', r'$P_{12}$'),
                   3: (r'$F_{21}$', r'$P_{21}$')}
    indices = list(indices)
    n_rows = len(indices)
    n_cols = len(components)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.0 * n_cols, 2.0 * n_rows),
                             squeeze=False, sharex='col')

    # Identity deformation gradient (reference state): F=I, PK1=0.
    # F_11 = F_22 = 1, F_12 = F_21 = 0.
    identity_F = {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0}

    # Plot fe2 first so it sits in the background and leads the legend.
    ordered_items = list(results.items())
    ordered_items.sort(key=lambda kv: 0 if kv[0] == 'fe2' else 1)
    # Preserve original index for color/linestyle cycling.
    orig_index = {k: i for i, (k, _) in enumerate(results.items())}

    for r, qp_idx in enumerate(indices):
        for c, comp in enumerate(components):
            ax = axes[r, c]
            ax.axhline(0.0, color='gray', linewidth=0.6, zorder=0)
            ax.axvline(identity_F[comp], color='gray', linewidth=0.6, zorder=0)
            for key, data in ordered_items:
                if key == 'nn_32':
                    continue
                i = orig_index[key]
                # Skip step 0: it's the initialization (F=I, PK1=0), not a
                # prediction — plotting it creates a misleading line segment.
                F = data['F'][1:]
                P = data['PK1'][1:]
                if qp_idx >= F.shape[1]:
                    continue
                ax.plot(F[:, qp_idx, comp], P[:, qp_idx, comp],
                        color=_model_color(key, i),
                        linestyle=['-', '--', '-.', ':'][i % 4],
                        linewidth=1.0,
                        label=MODEL_LABELS.get(key, key) if (r == 0 and c == 0) else None)
            xlab, ylab = comp_labels[comp]
            if r == n_rows - 1:
                ax.set_xlabel(xlab, fontsize=9)
            if c == 0:
                ax.set_ylabel(f'qp {qp_idx}\n{ylab}', fontsize=9)
            else:
                ax.set_ylabel(ylab, fontsize=9)
            ax.tick_params(labelsize=8)

    plt.tight_layout()
    if n_rows * n_cols > 0:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, .987), ncol=len(labels),
                   fontsize=10, frameon=False)
        fig.subplots_adjust(top=1.0 - 0.5 / (2.0 * n_rows))
    plt.savefig(f"{output_folder}/{filename}.pdf", bbox_inches='tight')
    plt.savefig(f"{output_folder}/{filename}.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved stress-strain paths to {output_folder}/{filename}")


def replot_from_saved(output_folder, extra_folder=None):
    """Rebuild all comparison plots from ``comparison_data.npz`` — no sim needed.

    If ``extra_folder`` is given, its models are merged in on top of
    ``output_folder``'s (extra wins on key collisions). Mesh / grading metadata
    is taken from ``output_folder``; plots are written there.
    """
    data = np.load(f"{output_folder}/comparison_data.npz", allow_pickle=True)
    model_keys = [str(k) for k in data['model_keys']]

    results = {}
    for key in model_keys:
        results[key] = {
            'coords': data[f'{key}_coords'],
            'F': data[f'{key}_F'],
            'PK1': data[f'{key}_PK1'],
        }

    if extra_folder is not None:
        extra = np.load(f"{extra_folder}/comparison_data.npz", allow_pickle=True)
        for key in (str(k) for k in extra['model_keys']):
            results[key] = {
                'coords': extra[f'{key}_coords'],
                'F': extra[f'{key}_F'],
                'PK1': extra[f'{key}_PK1'],
            }

    mesh_cells = data['mesh_cells']
    mesh_nodes = data['mesh_nodes']
    mesh_elems = data['mesh_elems']
    tri = Triangulation(mesh_nodes[:, 0], mesh_nodes[:, 1], mesh_elems)
    n_cells = mesh_elems.shape[0]

    # Grading panel needs a lightweight sim-like object.
    class _MeshView:
        pass
    view = _MeshView()
    view.beam_length = float(data['beam_length'])
    view.beam_height = float(data['beam_height'])
    view.qp_coords = data['qp_coords']
    view.micro_variables = {
        'vfrac': data['micro_vfrac'],
        'mu': data['micro_mu'],
        'theta': data['micro_theta'],
        'ratio': data['micro_ratio'],
    }
    # plot_manual_grading builds its own triangulation from sim.domain, so
    # skip it here; user can regenerate it from the saved micro fields if
    # they want (or keep the existing grading.pdf).

    plot_trajectories(results, output_folder, mesh_cells)
    plot_final_deformation(results, output_folder, mesh_cells)
    plot_final_deformation_separate(results, output_folder, mesh_cells)
    plot_stress_maps(results, (tri, n_cells), output_folder)
    plot_stress_strain_paths(results, output_folder, indices=_default_path_indices(data['qp_coords']))
    print(f"Replotted from {output_folder}/comparison_data.npz")


# -- Entry point --------------------------------------------------------------

if __name__ == "__main__":
    replot_only = False
    output_folder = "../results/bending_model_comparison/new"
    extra_folder = None  #"../results/bending_model_comparison/v6_nn32_comparison" # None

    if replot_only:
        replot_from_saved(output_folder, extra_folder=extra_folder)
    else:
        run_comparison(
            # model_keys=['prnn_1', 'prnn', 'prnn_512', 'prnn_nonlin', 'nn', 'nn_32'], #, 'fe2'],
            model_keys=['prnn', 'prnn_512', 'prnn_nonlin', 'nn', 'nn_32', 'fe2'],
            # model_keys=['nn_32'], #, 'fe2'],
            max_load=0.05,
            step_size_init=-0.01,
            step_size_min=-0.0001,
            macro_meshsize=0.05,
            beam_length=2.0,
            beam_height=0.3,
            output_folder=output_folder,
        )
