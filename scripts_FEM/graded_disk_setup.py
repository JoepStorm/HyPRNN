"""
Pressurized annulus with graded PRNN material.

Demonstrates the effect of circumferential fiber-orientation grading vs. uniform
material under internal outward pressure.

Geometry: circular annulus, R_inner < r < R_outer, centered at origin.
BCs: outer boundary fixed (u=0), inner boundary has outward pressure.

Key comparison: stress vs. angle around the inner hole boundary.
- Uniform theta=0: material has a fixed horizontal stiff axis. Under symmetric
  pressure, fibers are radially oriented at the left/right of the hole → stress
  concentrates at those angles.
- Circumferential theta (phi+pi/2): stiff axis always tangent to circles → stress
  is uniform around the circumference → lower peak.
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

from mpi4py import MPI
import numpy as np
import jax.numpy as jnp
import ufl
from dolfinx import fem, io, mesh
from petsc4py import PETSc
from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.solvers import NonlinearMaterialProblem
from dolfinx_materials.utils import create_quadrature_functionspace
from dolfinx.io import gmsh as gmshio
import gmsh
import gc

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Ellipse, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.tri import Triangulation
plt.style.use(['science', 'bright'])
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
rc('text', usetex=True)

from scripts_materials.prnn_material import PRNNMaterial
from scripts_materials.fe2_material import HeterogeneousFE2Material
from material_params import FUNGI_LAMBDA


def create_annulus_mesh(R_inner, R_outer, lc):
    """Quarter annulus (top-left: x<=0, y>=0) centered at origin."""
    gmsh.initialize()
    gdim = 2
    outer = gmsh.model.occ.addDisk(0, 0, 0, R_outer, R_outer)
    inner = gmsh.model.occ.addDisk(0, 0, 0, R_inner, R_inner)
    annulus = gmsh.model.occ.cut([(gdim, outer)], [(gdim, inner)])

    box = gmsh.model.occ.addRectangle(-R_outer * 1.5, 0, 0, R_outer * 1.5, R_outer * 1.5)
    gmsh.model.occ.intersect(annulus[0], [(gdim, box)])

    gmsh.model.occ.synchronize()

    vols = gmsh.model.getEntities(gdim)
    gmsh.model.addPhysicalGroup(gdim, [vols[0][1]], 1, name="Disk")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.model.mesh.generate(gdim)

    domain = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=gdim).mesh
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    gmsh.finalize()
    return domain


class GradedDiskSimulation:
    """Pressurized annulus with graded PRNN material.

    Outer boundary fixed; outward traction (pressure) at inner boundary.
    """

    def __init__(self,
                 macro_meshsize=0.02,
                 R_inner=0.2,
                 R_outer=0.5,
                 use_surrogate=True,
                 rve_config=None,
                 output_folder='../results/',
                 prnn_model_loc="../trained_models/train_vary_all_v2/prnn_nonlin_1L8_12m_sigmoid/samples512_run1"):
        self.comm = MPI.COMM_WORLD
        self.dim = 2
        self.R_inner = R_inner
        self.R_outer = R_outer
        self.center = [0.0, 0.0]
        self.output_folder = output_folder
        self.prnn_model_loc = prnn_model_loc
        self.use_surrogate = use_surrogate
        self.rve_config = rve_config

        self.step_size_init = 0.2
        self.step_size_factor = 0.5
        self.step_size_min = 0.001
        self.max_load = 3.0

        self._setup_mesh(macro_meshsize)
        self._setup_function_spaces()
        self._setup_boundary_conditions()

        WJ = create_quadrature_functionspace(self.domain, self.deg_quad, self.order)
        self.qp_coords = WJ.tabulate_dof_coordinates()[:, :2]

    def _setup_mesh(self, lc):
        self.domain = create_annulus_mesh(self.R_inner, self.R_outer, lc)
        self.dim = self.domain.topology.dim
        print(f"Quarter annulus mesh: {self.domain.topology.index_map(self.dim).size_global} elements")

    def _setup_function_spaces(self):
        self.order = 1
        self.deg_quad = 2 * (self.order - 1)
        self.V = fem.functionspace(self.domain, ("P", self.order, (2,)))
        self.num_elems = self.V.dofmap.list.shape[0]
        self.ips_per_elem = 1
        self.num_ips = self.num_elems * self.ips_per_elem
        print(f"num_elements: {self.num_elems}, num_ips: {self.num_ips}")

    def _setup_boundary_conditions(self):
        """Fix outer circle; tag inner circle for pressure loading."""
        fdim = self.dim - 1

        def outer_boundary(x):
            return np.isclose(np.sqrt(x[0]**2 + x[1]**2), self.R_outer, atol=0.01)

        def inner_boundary(x):
            return np.isclose(np.sqrt(x[0]**2 + x[1]**2), self.R_inner, atol=0.01)

        outer_facets = mesh.locate_entities_boundary(self.domain, fdim, outer_boundary)
        outer_dofs_x = fem.locate_dofs_topological(self.V.sub(0), fdim, outer_facets)
        outer_dofs_y = fem.locate_dofs_topological(self.V.sub(1), fdim, outer_facets)
        self.bcs = [
            fem.dirichletbc(0.0, outer_dofs_x, self.V.sub(0)),
            fem.dirichletbc(0.0, outer_dofs_y, self.V.sub(1)),
        ]

        # x-axis symmetry (y=0): u_y = 0
        def xaxis_boundary(x):
            return np.isclose(x[1], 0.0, atol=1e-6)
        xaxis_facets = mesh.locate_entities_boundary(self.domain, fdim, xaxis_boundary)
        xaxis_dofs_y = fem.locate_dofs_topological(self.V.sub(1), fdim, xaxis_facets)
        self.bcs.append(fem.dirichletbc(0.0, xaxis_dofs_y, self.V.sub(1)))

        # y-axis symmetry (x=0): u_x = 0
        def yaxis_boundary(x):
            return np.isclose(x[0], 0.0, atol=1e-6)
        yaxis_facets = mesh.locate_entities_boundary(self.domain, fdim, yaxis_boundary)
        yaxis_dofs_x = fem.locate_dofs_topological(self.V.sub(0), fdim, yaxis_facets)
        self.bcs.append(fem.dirichletbc(0.0, yaxis_dofs_x, self.V.sub(0)))
        print(f"Symmetry BCs: {len(xaxis_facets)} x-axis facets, {len(yaxis_facets)} y-axis facets")

        inner_facets = mesh.locate_entities_boundary(self.domain, fdim, inner_boundary)
        sorted_idx = np.argsort(inner_facets)
        self.facet_tags = mesh.meshtags(
            self.domain, fdim,
            inner_facets[sorted_idx],
            np.ones(len(inner_facets), dtype=np.int32)[sorted_idx],
        )
        self.ds_inner = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tags)
        self.load_magnitude = fem.Constant(self.domain, PETSc.ScalarType(0.0))

        print(f"Outer facets fixed: {len(outer_facets)}, inner facets tagged: {len(inner_facets)}")

    def _tensor_to_vector_2D(self, T):
        if ufl.shape(T) == (2, 2):
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
            assert self.rve_config is not None, "rve_config required for FE² material."
            E_w = 1000.0
            nu_w = 0.3
            full_micro_variables = {
                'vfrac': self.micro_variables['vfrac'],
                'theta': self.micro_variables['theta'],
                'ratio': self.micro_variables['ratio'],
                'mu_fungi': self.micro_variables['mu'],
                'lambda_fungi': self.micro_variables['lambda'],
                'mu_wood': jnp.ones(self.num_ips) * E_w / (2 * (1 + nu_w)),
                'lambda_wood': jnp.ones(self.num_ips) * E_w * nu_w / ((1 + nu_w) * (1 - 2 * nu_w)),
            }
            self.rve_config['gauss_coords'] = self.qp_coords
            self.material = HeterogeneousFE2Material(
                graded_variables=full_micro_variables,
                rve_config=self.rve_config,
            )
        else:
            self.material = PRNNMaterial(
                f"{self.prnn_model_loc}_settings",
                f"{self.prnn_model_loc}_normparams",
                f"{self.prnn_model_loc}",
                self.micro_variables,
            )

        self.qmap = QuadratureMap(self.domain, self.deg_quad, self.material)
        self.qmap.register_gradient("F", self._F(self.u))

        P = self.qmap.fluxes["PK1"]
        n = ufl.FacetNormal(self.domain)
        # At inner boundary, n_domain points inward (toward center).
        # Outward pressure traction = -P_load * n_domain → residual contribution: +P_load * n·v
        self.Res = (ufl.dot(P, self._dF(self.u, self.v)) * self.qmap.dx
                    + self.load_magnitude * ufl.dot(n, self.v) * self.ds_inner(1))
        self.Jac = self.qmap.derivative(self.Res, self.u, self.du)

    def _setup_solver(self):
        petsc_options = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_atol": 1e-5,
            "snes_max_it": 15,
            "ksp_type": "gmres",
            "ksp_rtol": 1e-8,
            "pc_type": "ilu",
            "pc_factor_levels": 1,
        }
        self.problem = NonlinearMaterialProblem(
            self.qmap, self.Res, self.u, bcs=self.bcs,
            J=self.Jac, petsc_options_prefix="DISK",
            petsc_options=petsc_options,
        )

    def run(self, write_output=True, write_name=None):
        """Incrementally apply pressure to max_load and write VTX output."""
        self._setup_material_and_problem()
        self._setup_solver()

        cur_step_size = self.step_size_init
        cur_load = 0.0
        converged_steps = 0

        if write_name is None:
            write_name = f"disk_{self.output_name}"

        print(f"\nStarting annulus pressure simulation ({self.output_name})...")

        if write_output:
            micro_funcs = self._create_micro_variable_functions()
            self._create_stress_strain_functions()
            self._update_stress_strain_functions()
            stress_strain_funcs = (list(self._S_funcs.values())
                                   + list(self._E_funcs.values()))
            with io.VTXWriter(MPI.COMM_WORLD, f"{self.output_folder}/{write_name}.bp",
                              [self.u] + micro_funcs + stress_strain_funcs) as xf:
                xf.write(0.0)
                converged_steps = self._solve_loop(cur_step_size, cur_load, converged_steps, xf)
        else:
            converged_steps = self._solve_loop(cur_step_size, cur_load, converged_steps)

        return converged_steps

    def _solve_loop(self, cur_step_size, cur_load, converged_steps, xf=None):
        converged = True
        while abs(cur_load) < self.max_load - 1e-10:
            if converged:
                self.u_pre.x.array[:] = self.u.x.array[:]
                converged_steps += 1
                cur_load += cur_step_size
            else:
                self.u.x.array[:] = self.u_pre.x.array[:]
                self._destroy_problem()
                self._setup_solver()
                print(f"Load {cur_load + cur_step_size:.5f} did not converge, reducing step size.")
                cur_step_size *= self.step_size_factor
                cur_load += cur_step_size

            if abs(cur_step_size) < abs(self.step_size_min):
                print("Step size too small, stopping simulation.")
                return False

            self.load_magnitude.value = cur_load

            try:
                self.problem.solve()
                converged = self.problem.solver.getConvergedReason() > 0
                nr_iters = self.problem.solver.getIterationNumber()
                # self._print_strain_diagnostics()
                print(f"Step {converged_steps}: load={cur_load:.5f}, converged={converged}, iters={nr_iters}")
            except:
                print("Error while solving loop.")
                converged = False

            if converged:
                if xf is not None:
                    if hasattr(self, '_S_funcs'):
                        self._update_stress_strain_functions()
                    xf.write(converged_steps)
            else:
                cur_load -= cur_step_size

        return converged_steps

    def apply_uniform_material(self, vfrac=0.3, theta=0.0, ratio=1.75, mu=1.30):  #FUNGI_MU):
        """Uniform material: fixed horizontal fiber direction throughout."""
        self.micro_variables = {
            'vfrac': jnp.ones(self.num_ips) * vfrac,
            'theta': jnp.ones(self.num_ips) * theta,
            'ratio': jnp.ones(self.num_ips) * ratio,
            'mu': jnp.ones(self.num_ips) * mu,
            'lambda': jnp.ones(self.num_ips) * FUNGI_LAMBDA,
        }
        self.output_name = "uniform"

    def apply_circumferential_grading(self,
                                       mu_inner=0.1, mu_outer=2.5,
                                       vfrac_inner=0.1, vfrac_outer=0.5,
                                       ratio_inner=1.0, ratio_outer=2.5):
                                        # mu_inner = FUNGI_MU, mu_outer = FUNGI_MU,
                                        # vfrac_inner = 0.1, vfrac_outer = 0.5,
                                        # ratio_inner = 2.5, ratio_outer = 1.0):
        """Graded: radial fiber orientation + vfrac and ratio increasing away from hole.

        With fixed outer boundary + internal pressure, the stiffer outer region
        carries more load, relieving the inner boundary → lower stress concentration.
        vfrac and ratio grade from low (soft/isotropic near hole) to high (stiff/anisotropic
        near outer wall). Theta is radial so fibers directly resist outward expansion.
        """
        x = self.qp_coords[:, 0]
        y = self.qp_coords[:, 1]
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        theta = phi  # radial orientation

        t = np.clip((r - self.R_inner) / (self.R_outer - self.R_inner), 0.0, 1.0)
        mu = mu_inner * (1 - t) + mu_outer * t
        vfrac = vfrac_inner * (1 - t) + vfrac_outer * t
        ratio = ratio_inner * (1 - t) + ratio_outer * t

        self.micro_variables = {
            'vfrac': jnp.array(vfrac),
            'theta': jnp.array(theta),
            'ratio': jnp.array(ratio),
            'mu': jnp.array(mu),
            'lambda': jnp.ones(self.num_ips) * FUNGI_LAMBDA,
        }
        self.output_name = "graded"

    def apply_radial_grading(self, vfrac, r_combi, mu):
        """Apply 1D radial grading using the combined r_combi parameter.

        r_combi encodes both aspect ratio and fiber orientation (same value at each QP):
          r_combi > 1: radial fibers (theta = phi), ratio = r_combi
          r_combi < 1: circumferential fibers (theta = phi + pi/2), ratio = 1/r_combi
          r_combi = 1: isotropic

        Args:
            vfrac: per-QP volume fraction array
            r_combi: per-QP combined ratio/orientation parameter (0.4 – 2.5)
            mu: per-QP shear modulus array
        """
        phi = np.arctan2(self.qp_coords[:, 1], self.qp_coords[:, 0])
        r_combi = np.asarray(r_combi)
        self._r_combi = r_combi.copy()

        ratio = np.where(r_combi >= 1.0, r_combi, 1.0 / r_combi)
        theta = np.where(r_combi >= 1.0, phi, phi + np.pi / 2)

        self.micro_variables = {
            'vfrac': jnp.array(np.asarray(vfrac)),
            'theta': jnp.array(theta),
            'ratio': jnp.array(ratio),
            'mu': jnp.array(np.asarray(mu)),
            'lambda': jnp.ones(self.num_ips) * FUNGI_LAMBDA,
        }
        self.output_name = "radial_graded"

    def _print_strain_diagnostics(self):
        """Print max F and Green-Lagrange E at quadrature points to check training range."""
        F_grad = self.qmap.gradients["F"]
        F_flat = self.qmap.get_gradient_vals(F_grad, self.qmap.cells)  # [num_qp, 4]
        # F components: [F11, F22, F12, F21]
        F_dev = F_flat.copy()
        F_dev[:, 0] -= 1.0  # F11 - 1
        F_dev[:, 1] -= 1.0  # F22 - 1
        max_dev = np.max(np.abs(F_dev), axis=0)

        # Compute Green-Lagrange E at the worst QP
        worst_qp = np.argmax(np.linalg.norm(F_dev, axis=1))
        F_mat = np.array([[F_flat[worst_qp, 0], F_flat[worst_qp, 2]],
                          [F_flat[worst_qp, 3], F_flat[worst_qp, 1]]])
        E = 0.5 * (F_mat.T @ F_mat - np.eye(2))
        print(f"  F max deviation from I: [{max_dev[0]:.4f}, {max_dev[1]:.4f}, {max_dev[2]:.4f}, {max_dev[3]:.4f}]"
              f"  (training range ~0.5)")
        print(f"  Worst QP E: [[{E[0,0]:.4f}, {E[0,1]:.4f}], [{E[1,0]:.4f}, {E[1,1]:.4f}]]")

    def get_von_mises_at_qp(self):
        """Compute von Mises stress at quadrature points via Cauchy stress σ = (1/J) P Fᵀ."""
        pk1 = self.qmap.fluxes["PK1"].x.array.reshape(-1, 4)
        P11, P22, P12, P21 = pk1[:, 0], pk1[:, 1], pk1[:, 2], pk1[:, 3]

        F_grad = self.qmap.gradients["F"]
        F = self.qmap.get_gradient_vals(F_grad, self.qmap.cells)
        F11, F22, F12, F21 = F[:, 0], F[:, 1], F[:, 2], F[:, 3]

        J = F11 * F22 - F12 * F21

        # σ = (1/J) P Fᵀ
        s11 = (P11 * F11 + P12 * F21) / J
        s22 = (P21 * F12 + P22 * F22) / J
        s12 = 0.5 * ((P11 * F12 + P12 * F22) + (P21 * F11 + P22 * F21)) / J

        return np.sqrt(s11**2 + s22**2 - s11 * s22 + 3 * s12**2)

    def get_nodal_displacements(self):
        """Return nodal displacements as [num_nodes, 2] array."""
        return self.u.x.array.reshape(-1, 2).copy()

    def get_max_inner_displacement(self):
        """Max displacement magnitude at inner boundary nodes."""
        coords = self.domain.geometry.x[:, :2]
        r = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
        inner_mask = np.isclose(r, self.R_inner, atol=0.01)
        if not inner_mask.any():
            return 0.0
        disp = self.u.x.array.reshape(-1, 2)
        return float(np.max(np.linalg.norm(disp[inner_mask], axis=1)))

    def _create_micro_variable_functions(self):
        V_DG0 = fem.functionspace(self.domain, ("DG", 0))
        funcs = []
        for name, values in self.micro_variables.items():
            f = fem.Function(V_DG0, name=name)
            f.x.array[:] = np.asarray(values)
            funcs.append(f)
        return funcs

    def _create_stress_strain_functions(self):
        """Create DG0 functions for PK1 stress and Green-Lagrange strain components."""
        V_DG0 = fem.functionspace(self.domain, ("DG", 0))
        self._S_funcs = {
            name: fem.Function(V_DG0, name=name)
            for name in ["S11", "S22", "S12", "S21"]
        }
        self._E_funcs = {
            name: fem.Function(V_DG0, name=name)
            for name in ["E11", "E22", "E12"]
        }

    def _update_stress_strain_functions(self):
        """Fill stress/strain DG0 functions from current quadrature state."""
        # PK1 stress: stored as [P11, P22, P12, P21]
        pk1 = self.qmap.fluxes["PK1"].x.array.reshape(-1, 4)
        self._S_funcs["S11"].x.array[:] = pk1[:, 0]
        self._S_funcs["S22"].x.array[:] = pk1[:, 1]
        self._S_funcs["S12"].x.array[:] = pk1[:, 2]
        self._S_funcs["S21"].x.array[:] = pk1[:, 3]

        # Deformation gradient: stored as [F11, F22, F12, F21]
        F_grad = self.qmap.gradients["F"]
        F = self.qmap.get_gradient_vals(F_grad, self.qmap.cells)
        F11, F22, F12, F21 = F[:, 0], F[:, 1], F[:, 2], F[:, 3]

        # Green-Lagrange strain E = 0.5*(F^T F - I)
        self._E_funcs["E11"].x.array[:] = 0.5 * (F11**2 + F21**2 - 1.0)
        self._E_funcs["E22"].x.array[:] = 0.5 * (F12**2 + F22**2 - 1.0)
        self._E_funcs["E12"].x.array[:] = 0.5 * (F11 * F12 + F21 * F22)

    def _destroy_problem(self):
        if hasattr(self, 'problem'):
            for attr in ('_snes', '_A', '_b', '_x', '_P_mat'):
                obj = getattr(self.problem, attr, None)
                if obj is not None:
                    obj.destroy()
            del self.problem

    def _destroy_qmap(self):
        if not hasattr(self, 'qmap'):
            return
        for func in self.qmap.fluxes.values():
            func.x.petsc_vec.destroy()
        for func in self.qmap.internal_state_variables.values():
            func.x.petsc_vec.destroy()
        if hasattr(self.qmap, 'jacobian_flatten'):
            self.qmap.jacobian_flatten.x.petsc_vec.destroy()
        if hasattr(self.qmap, 'rotation_func'):
            self.qmap.rotation_func.x.petsc_vec.destroy()

    def reset(self):
        self.u.x.array[:] = 0.0
        self.u_pre.x.array[:] = 0.0
        self.load_magnitude.value = 0.0
        self._destroy_problem()
        self._destroy_qmap()
        if hasattr(self, 'qmap'):
            if hasattr(self.qmap, 'reset'):
                self.qmap.reset()
            del self.qmap
        gc.collect()
        print("Problem reset to initial state")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_rc_norm(r_combi):
    """TwoSlopeNorm for r_combi diverging from 1.0 (radial↔circumferential)."""
    return mpl.colors.TwoSlopeNorm(
        vmin=min(float(r_combi.min()), 0.4), vcenter=1.0, vmax=max(float(r_combi.max()), 2.5)
    )


def _make_annulus_clip(ax, R_inner, R_outer):
    """Add an invisible annulus PathPatch to ax and return it for use as clip_path."""
    n = 100
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    outer_v = np.column_stack([R_outer * np.cos(t),      R_outer * np.sin(t)])
    inner_v = np.column_stack([R_inner * np.cos(t[::-1]), R_inner * np.sin(t[::-1])])
    verts = np.vstack([outer_v, outer_v[:1], inner_v, inner_v[:1]])
    codes = (
        [MplPath.MOVETO] + [MplPath.LINETO] * (n - 1) + [MplPath.CLOSEPOLY]
        + [MplPath.MOVETO] + [MplPath.LINETO] * (n - 1) + [MplPath.CLOSEPOLY]
    )
    clip = PathPatch(MplPath(verts, codes), transform=ax.transData)
    ax.add_patch(clip)
    clip.set_visible(False)
    return clip


def _make_ellipse_collection_direct(qp, ratio, theta, R_inner, R_outer, norm, base_frac):
    """Ellipses with explicit per-element ratio and fiber angle theta (rad)."""
    base = (R_outer - R_inner) * base_frac
    patches = [
        Ellipse(xy=(xi, yi), width=rv * base, height=base, angle=np.rad2deg(ang))
        for (xi, yi), rv, ang in zip(qp, ratio, theta)
    ]
    pc = PatchCollection(patches, cmap=mpl.colormaps['twilight_shifted'], norm=norm, linewidths=0.0, alpha=0.9)
    pc.set_array(np.asarray(ratio))
    return pc


def _make_ellipse_collection(qp, r_combi, R_inner, R_outer, norm, base_frac):
    """PatchCollection of oriented ellipses: aspect ratio = anisotropy, angle = fiber direction."""
    phi = np.arctan2(qp[:, 1], qp[:, 0])
    ratio_val = np.maximum(r_combi, 1.0 / np.maximum(r_combi, 1e-9))
    angle_el = np.where(r_combi >= 1.0, np.rad2deg(phi), np.rad2deg(phi) + 90.0)
    base = (R_outer - R_inner) * base_frac
    patches = [
        Ellipse(xy=(xi, yi), width=rv * base, height=base, angle=ang)
        for (xi, yi), rv, ang in zip(qp, ratio_val, angle_el)
    ]
    pc = PatchCollection(patches, cmap=mpl.colormaps['twilight_shifted'], norm=norm, linewidths=0.0, alpha=0.9)
    pc.set_array(r_combi)
    return pc


def _get_mesh_topology(domain):
    """Return (node_coords [N,2], elem_nodes [M,3]) for matplotlib plotting."""
    node_coords = domain.geometry.x[:, :2]
    topo = domain.topology
    topo.create_connectivity(topo.dim, 0)
    c_to_v = topo.connectivity(topo.dim, 0)
    num_cells = topo.index_map(topo.dim).size_local
    elem_nodes = np.array([c_to_v.links(c)[:3] for c in range(num_cells)])
    return node_coords, elem_nodes


def _mirror_quarter(node_coords, elem_nodes):
    """Mirror top-left quarter (x<=0, y>=0) to all four quadrants.

    Returns (full_coords, full_elems) with consistent triangle winding.
    """
    n = len(node_coords)
    mirrors = [
        (np.array([-1, 1]), True),   # flip x → top-right (Q1)
        (np.array([1, -1]), True),   # flip y → bottom-left (Q3)
        (np.array([-1, -1]), False), # flip both → bottom-right (Q4)
    ]
    all_coords = [node_coords]
    all_elems = [elem_nodes]
    for i, (m, swap_winding) in enumerate(mirrors):
        all_coords.append(node_coords * m)
        new_elems = elem_nodes + (i + 1) * n
        if swap_winding:
            new_elems = new_elems[:, [0, 2, 1]]
        all_elems.append(new_elems)
    return np.vstack(all_coords), np.vstack(all_elems)


def plot_grading_overview(sim, output_folder, radial_1d=False):
    """Produce two grading overview plots: quarter-only and mirrored full circle."""
    os.makedirs(output_folder, exist_ok=True)
    node_coords, elem_nodes = _get_mesh_topology(sim.domain)
    vfrac = np.asarray(sim.micro_variables['vfrac'])
    mu = np.asarray(sim.micro_variables['mu'])
    use_rcombi = radial_1d and hasattr(sim, '_r_combi')
    lim = sim.R_outer * 1.08

    full_coords, full_elems = _mirror_quarter(node_coords, elem_nodes)

    for mode in ('quarter', 'full'):
        is_full = (mode == 'full')

        if is_full:
            coords, elems = full_coords, full_elems
            cur_vfrac = np.tile(vfrac, 4)
            cur_mu = np.tile(mu, 4)
            figsize = (9, 4.2)
            xlims, ylims = (-lim, lim), (-lim, lim)
        else:
            coords, elems = node_coords, elem_nodes
            cur_vfrac, cur_mu = vfrac, mu
            figsize = (7, 3.2)
            xlims = (-lim, 0.03 * lim)
            ylims = (-0.03 * lim, lim)

        tri = Triangulation(coords[:, 0], coords[:, 1], elems)
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        def _style(ax):
            for r_b, ls in [(sim.R_inner, '--'), (sim.R_outer, '-')]:
                ax.add_patch(plt.Circle((0, 0), r_b, fill=False, ec='black', lw=0.8, ls=ls))
            ax.set_xlim(*xlims)
            ax.set_ylim(*ylims)
            ax.set_aspect('equal')
            ax.axis('off')

        def _tripcolor(ax, field, cmap_name, norm):
            cmap = mpl.colormaps[cmap_name]
            ax.tripcolor(tri, facecolors=field, cmap=cmap, norm=norm)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                         ax=ax, location='bottom', shrink=0.65, pad=0.02, aspect=28)

        if use_rcombi:
            r_combi = sim._r_combi
            norm_rc = _make_rc_norm(r_combi)

            _tripcolor(axes[0], cur_vfrac, 'cividis_r', mpl.colors.Normalize(vmin=0.1, vmax=0.5))
            axes[0].set_title(r'Volume fraction $V_f$', fontsize=10)

            # Ellipse panel
            qp = sim.qp_coords
            if is_full:
                qp_plot = np.vstack([qp, qp * [-1, 1], qp * [1, -1], qp * [-1, -1]])
                rc_plot = np.tile(r_combi, 4)
            else:
                qp_plot, rc_plot = qp, r_combi
            n_skip = max(1, len(qp) // 600)
            pc = _make_ellipse_collection(qp_plot[::n_skip], rc_plot[::n_skip],
                                          sim.R_inner, sim.R_outer, norm_rc, base_frac=1 / 22)
            if is_full:
                clip = _make_annulus_clip(axes[1], sim.R_inner, sim.R_outer)
                pc.set_clip_path(clip)
            axes[1].add_collection(pc)
            fig.colorbar(pc, ax=axes[1], location='bottom', shrink=0.65, pad=0.02, aspect=28)
            axes[1].set_title(r'Combined ratio $r_{\mathrm{combi}}$', fontsize=10)

            _tripcolor(axes[2], cur_mu, 'Greens',
                       mpl.colors.Normalize(vmin=0.1, vmax=2.5))
            axes[2].set_title(r'Shear modulus $\mu$', fontsize=10)

        else:
            theta = ((np.asarray(sim.micro_variables['theta']) + np.pi) % (2 * np.pi)) - np.pi
            ratio = np.asarray(sim.micro_variables['ratio'])

            if is_full:
                # Mirror theta: flip x → pi-theta, flip y → -theta, flip both → theta-pi
                cur_theta = np.concatenate([theta, np.pi - theta, -theta, theta - np.pi])
                cur_theta = ((cur_theta + np.pi) % (2 * np.pi)) - np.pi
                cur_ratio = np.tile(ratio, 4)
            else:
                cur_theta, cur_ratio = theta, ratio

            for ax, (field, cmap_name, norm, title) in zip(axes, [
                (cur_vfrac, 'YlOrRd', mpl.colors.Normalize(vmin=0.1, vmax=0.5), r'Volume fraction $V_f$'),
                (cur_ratio, 'Blues', mpl.colors.Normalize(vmin=ratio.min(), vmax=ratio.max()), r'Aspect ratio $r$'),
                (cur_theta, 'twilight', mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi), r'Fiber orientation $\theta$'),
            ]):
                _tripcolor(ax, field, cmap_name, norm)
                ax.set_title(title, fontsize=10)

            # White arrows on theta panel
            qp = sim.qp_coords
            if is_full:
                qp_arr = np.vstack([qp, qp * [-1, 1], qp * [1, -1], qp * [-1, -1]])
                t_arr = cur_theta[:len(qp_arr)]  # already mirrored above at cell level
                # Recompute per-QP mirrored theta for arrows
                t_arr = np.concatenate([theta, np.pi - theta, -theta, theta - np.pi])
            else:
                qp_arr, t_arr = qp, theta
            n_skip = max(1, len(qp_arr) // 600)
            q, t = qp_arr[::n_skip], t_arr[::n_skip]
            arrow_len = sim.R_outer * 0.07
            segs = [
                [(xi - np.cos(ti) * arrow_len, yi - np.sin(ti) * arrow_len),
                 (xi + np.cos(ti) * arrow_len, yi + np.sin(ti) * arrow_len)]
                for (xi, yi), ti in zip(q, t)
            ]
            lc_col = LineCollection(segs, colors='white', linewidths=0.6, alpha=0.7)
            if is_full:
                lc_col.set_clip_path(_make_annulus_clip(axes[2], sim.R_inner, sim.R_outer))
            axes[2].add_collection(lc_col)

        for ax in axes:
            _style(ax)

        plt.tight_layout()
        fname = 'grading_overview_full' if is_full else 'grading_overview'
        plt.savefig(f'{output_folder}/{fname}.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'{output_folder}/{fname}.png', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved: {output_folder}/{fname}.pdf")


def plot_grading_comparison(sim_uniform, sim_graded, output_folder,
                             vm_uniform=None, vm_graded=None,
                             du_uniform=None, du_graded=None,
                             disp_scale=1.0, fname='grading_comparison',
                             cases=None):
    """N×M quarter overview: one row per case.

    Either pass `cases=[(label, sim, vm, du), ...]` for N-way comparison, or
    keep the legacy 2-way form (sim_uniform, sim_graded, vm/du_uniform/graded).
    Columns are vfrac, r_combi, mu, and (optionally) von Mises stress.
    """
    os.makedirs(output_folder, exist_ok=True)

    if cases is None:
        cases = [
            ('Uniform', sim_uniform, vm_uniform, du_uniform),
            ('Graded',  sim_graded,  vm_graded,  du_graded),
        ]
    sims = [(lbl, s) for lbl, s, _, _ in cases]
    vm_data = {lbl: vm for lbl, _, vm, _ in cases}
    du_data = {lbl: du for lbl, _, _, du in cases}
    with_stress = all(vm is not None for _, _, vm, _ in cases)

    lim = cases[0][1].R_outer * 1.08
    xlims = (-lim, 0.03 * lim)
    ylims = (-0.03 * lim, lim)

    # Shared norms across both rows (fixed to parameter bounds)
    norm_vf = mpl.colors.Normalize(vmin=0.1, vmax=0.5)
    norm_mu = mpl.colors.Normalize(vmin=0.1, vmax=2.5)
    norm_rc = mpl.colors.TwoSlopeNorm(vmin=0.4, vcenter=1.0, vmax=2.5)

    # cmap_vf = mpl.colormaps['YlOrRd']
    cmap_vf = mpl.colormaps['cividis_r']
    cmap_mu = mpl.colormaps['Greens']

    if with_stress:
        vmax = float(max(np.asarray(vm).max() for _, _, vm, _ in cases))
        # Symmetric norm so [0, vmax] uses upper half of Spectral_r
        norm_vm_plot = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
        norm_vm_cbar = mpl.colors.Normalize(vmin=0, vmax=vmax)
        cmap_vm = mpl.colormaps['Spectral_r']
        cmap_vm_half = mpl.colors.LinearSegmentedColormap.from_list(
            'Spectral_r_half', cmap_vm(np.linspace(0.5, 1.0, 256)),
        )

    n_cols = 4 if with_stress else 3
    n_rows = len(cases)
    fig_w = 9 if with_stress else 7
    fig_h = 2.45 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True, squeeze=False)

    pc_last = None  # keep last r_combi PatchCollection for shared colorbar
    for row, (label, sim) in enumerate(sims):
        node_coords, elem_nodes = _get_mesh_topology(sim.domain)
        tri = Triangulation(node_coords[:, 0], node_coords[:, 1], elem_nodes)

        vfrac = np.asarray(sim.micro_variables['vfrac'])
        mu = np.asarray(sim.micro_variables['mu'])
        r_combi = sim._r_combi

        ax_vf = axes[row, 0]
        ax_rc = axes[row, 1]
        ax_mu = axes[row, 2]

        ax_vf.tripcolor(tri, facecolors=vfrac, cmap=cmap_vf, norm=norm_vf)
        ax_mu.tripcolor(tri, facecolors=mu, cmap=cmap_mu, norm=norm_mu)

        # r_combi ellipses on the single quarter's QPs
        qp = sim.qp_coords
        n_skip = max(1, len(qp) // 600)
        if getattr(sim, '_random_mode', False):
            theta = np.asarray(sim.micro_variables['theta'])
            ratio = np.asarray(sim.micro_variables['ratio'])
            pc = _make_ellipse_collection_direct(
                qp[::n_skip], ratio[::n_skip], theta[::n_skip],
                sim.R_inner, sim.R_outer, norm_rc, base_frac=1 / 22,
            )
        else:
            pc = _make_ellipse_collection(qp[::n_skip], r_combi[::n_skip], sim.R_inner, sim.R_outer, norm_rc, base_frac=1 / 22)
        ax_rc.add_collection(pc)
        pc_last = pc

        panel_axes = [ax_vf, ax_rc, ax_mu]

        if with_stress:
            ax_vm = axes[row, 3]
            vm = vm_data[label]
            du = du_data[label]
            if du is not None:
                deformed = node_coords + du * disp_scale
                tri_ref = Triangulation(node_coords[:, 0], node_coords[:, 1], elem_nodes)
                ax_vm.triplot(tri_ref, color='gray', lw=0.2, alpha=0.2, zorder=-1)
                tri_vm = Triangulation(deformed[:, 0], deformed[:, 1], elem_nodes)
            else:
                tri_vm = tri
            ax_vm.tripcolor(tri_vm, facecolors=vm, cmap=cmap_vm, norm=norm_vm_plot)
            ax_vm.text(0.98, 0.02, f'{np.asarray(vm).max():.2f}',
                       transform=ax_vm.transAxes, ha='right', va='bottom',
                       fontsize=12, color='black')
            panel_axes.append(ax_vm)

        # Style and row label
        for ax in panel_axes:
            for r_b, ls in [(sim.R_inner, '--'), (sim.R_outer, '-')]:
                ax.add_patch(plt.Circle((0, 0), r_b, fill=False, ec='black', lw=0.8, ls=ls))
            ax.set_xlim(*xlims)
            ax.set_ylim(*ylims)
            ax.set_aspect('equal')
            ax.axis('off')

        ax_vf.text(xlims[0] - 0.02 * lim, 0.5 * lim, label, ha='center', va='center', fontsize=12, rotation=90)

    axes[0, 0].set_title(r'Volume fraction $V_f$', fontsize=10)
    axes[0, 1].set_title(r'Combined ratio $r_{\mathrm{combi}}$', fontsize=10)
    axes[0, 2].set_title(r'Shear modulus $\mu$', fontsize=10)
    if with_stress:
        axes[0, 3].set_title(r'Von Mises $\sigma_{VM}$ [MPa]', fontsize=10)

    # One shared colorbar per column, below both rows
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm_vf, cmap=cmap_vf), ax=axes[:, 0], location='bottom', shrink=0.75, pad=0.02, aspect=28)
    fig.colorbar(pc_last, ax=axes[:, 1], location='bottom', shrink=0.75, pad=0.02, aspect=28)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm_mu, cmap=cmap_mu), ax=axes[:, 2], location='bottom', shrink=0.75, pad=0.02, aspect=28)
    if with_stress:
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm_vm_cbar, cmap=cmap_vm_half),
                     ax=axes[:, 3], location='bottom', shrink=0.75, pad=0.02, aspect=28)

    plt.savefig(f'{output_folder}/{fname}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_folder}/{fname}.png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {output_folder}/{fname}.pdf")


def plot_radial_slice(sim, output_folder, angle_deg=10.0, angle_center_deg=0.0,
                      fname='radial_slice.pdf'):
    """Compact stacked-strip figure of 1D radial grading parameters.

    Full mesh clipped to a rectangular viewport — clean straight edges.
    Three panels (r_combi mode): vfrac colormap | r_combi ellipses | mu colormap.
    """
    os.makedirs(output_folder, exist_ok=True)
    node_coords, elem_nodes = _get_mesh_topology(sim.domain)
    full_tri = Triangulation(node_coords[:, 0], node_coords[:, 1], elem_nodes)

    half_height = sim.R_outer * np.sin(np.deg2rad(angle_deg / 2.0))
    x_lo, x_hi = sim.R_inner - 0.005, sim.R_outer + 0.005
    y_lo, y_hi = -half_height - 0.005, half_height + 0.005

    vfrac = np.asarray(sim.micro_variables['vfrac'])
    mu    = np.asarray(sim.micro_variables['mu'])
    use_rcombi = hasattr(sim, '_r_combi')

    if use_rcombi:
        r_combi  = sim._r_combi
        norm_rc  = _make_rc_norm(r_combi)
        n_panels = 3
        ylabels  = [r'$V_f$', r'$r_{\mathrm{combi}}$', r'$\mu$']
    else:
        ratio    = np.asarray(sim.micro_variables['ratio'])
        theta    = ((np.asarray(sim.micro_variables['theta']) + np.pi) % (2 * np.pi)) - np.pi
        n_panels = 4
        ylabels  = [r'$V_f$', r'$r$', r'$\theta$', r'$\mu$']

    fig, axes = plt.subplots(n_panels, 1, figsize=(4.5, n_panels * 0.9),
                             sharex=True, sharey=True)

    def _strip(ax, field, cmap_name, norm):
        ax.tripcolor(full_tri, facecolors=field, cmap=mpl.colormaps[cmap_name], norm=norm)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=mpl.colormaps[cmap_name]),
                     ax=ax, fraction=0.03, pad=0.01, aspect=8)

    if use_rcombi:
        _strip(axes[0], vfrac, 'YlOrRd', mpl.colors.Normalize(vmin=0.1, vmax=0.5))
        # Ellipses: QPs inside the rectangle viewport
        qp = sim.qp_coords
        qp_mask = (qp[:, 0] >= x_lo) & (qp[:, 0] <= x_hi) & \
                  (qp[:, 1] >= y_lo) & (qp[:, 1] <= y_hi)
        pc = _make_ellipse_collection(qp[qp_mask], r_combi[qp_mask],
                                      sim.R_inner, sim.R_outer, norm_rc, base_frac=1 / 22)
        pc.set_clip_path(axes[1].patch)
        axes[1].add_collection(pc)
        fig.colorbar(pc, ax=axes[1], fraction=0.03, pad=0.01, aspect=8)
        _strip(axes[2], mu, 'Greens', mpl.colors.Normalize(vmin=0.1, vmax=2.5))
    else:
        for ax, field, cmap_name, norm in zip(
            axes,
            [vfrac, ratio, theta, mu],
            ['YlOrRd', 'Blues', 'twilight', 'Greens'],
            [mpl.colors.Normalize(vmin=0.1, vmax=0.5),
             mpl.colors.Normalize(vmin=ratio.min(), vmax=ratio.max()),
             mpl.colors.Normalize(vmin=-np.pi,       vmax=np.pi),
             mpl.colors.Normalize(vmin=0.1,            vmax=2.5)],
        ):
            _strip(ax, field, cmap_name, norm)

    for ax, label in zip(axes, ylabels):
        ax.set_ylabel(label, rotation=0, labelpad=18, va='center')
        ax.set_aspect('auto')
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_yticks([])
        for spine in ('top', 'right', 'left'):
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel(r'$d$')
    plt.tight_layout(h_pad=0.3)
    path = os.path.join(output_folder, fname)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_stress_comparison(sim, *args, output_folder=None, disp_scale=10.0,
                           labels=('Uniform', 'Graded'), cases=None):
    """Side-by-side deformed annuli colored by von Mises stress.

    Either pass `cases=[(vm, du, label), ...]` for N-way comparison, or keep
    the legacy 2-way positional form: (vm_left, du_left, vm_right, du_right).
    """
    if cases is None:
        vm_left, du_left, vm_right, du_right = args
        cases = [(vm_left, du_left, labels[0]), (vm_right, du_right, labels[1])]
    assert output_folder is not None, "output_folder required"

    os.makedirs(output_folder, exist_ok=True)
    node_coords, elem_nodes = _get_mesh_topology(sim.domain)

    vmax = max(float(np.asarray(vm).max()) for vm, _, _ in cases)
    # Symmetric norm over [-vmax, vmax] — data in [0, vmax] maps to the upper
    # half of the colormap, starting from the midpoint colour.
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
    norm_cbar = mpl.colors.Normalize(vmin=0, vmax=vmax)
    cmap = mpl.colormaps['Spectral_r']   # reversed

    n = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5),
                             gridspec_kw={'bottom': 0.18})
    if n == 1:
        axes = [axes]
    titles = [
        lbl + r': $\sigma_{VM}^{\max}=' + f'{np.asarray(vm).max():.2f}$ [MPa]'
        for vm, _, lbl in cases
    ]
    for ax, (vm, du, _), title in zip(axes, cases, titles):
        deformed = node_coords + du * disp_scale

        tri_ref = Triangulation(node_coords[:, 0], node_coords[:, 1], elem_nodes)
        ax.triplot(tri_ref, color='gray', lw=0.2, alpha=0.2, zorder=-1)

        tri_def = Triangulation(deformed[:, 0], deformed[:, 1], elem_nodes)
        ax.tripcolor(tri_def, facecolors=vm, cmap=cmap, norm=norm)

        for r, ls in [(sim.R_inner, '--'), (sim.R_outer, '-')]:
            ax.add_patch(plt.Circle([0, 0], r, fill=False, ec='white',
                                    lw=0.8, linestyle=ls, alpha=0.5))
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=10)

    # Colorbar: only show the [0, vmax] portion of the norm
    cbar_ax = fig.add_axes([0.5 - 0.1, 0.27, 0.2, 0.012])
    cmap_half = mpl.colors.LinearSegmentedColormap.from_list(
        'Spectral_r_half', mpl.colormaps['Spectral_r'](np.linspace(0.5, 1.0, 256))
    )
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_cbar, cmap=cmap_half),
                        cax=cbar_ax, orientation='horizontal',
                        ticks=np.linspace(0, vmax, 2))
    cbar.set_label(r'$\tilde{\sigma}_{VM} [MPa]$', labelpad=-30)


    if disp_scale != 1:
        fig.text(0.5, 0.15, f'Deformation scaled ×{disp_scale}', ha='center', fontsize=8, color='gray')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/stress_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_folder}/stress_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved stress comparison: {output_folder}/stress_comparison.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_folder = "../results/graded_disk/larger_disp/test2_quarter/"
    os.makedirs(output_folder, exist_ok=True)

    # prnn_model_loc = ("../trained_models/train_vary_all/prnn_nonlin_1L8_6m_sigmoid/samples512_run0")
    # prnn_model_loc = ("../trained_models/train_vfrac_ratio/prnn_nonlin_1L8_6m_sigmoid/samples512_run0")
    prnn_model_loc = ("../trained_models/train_vary_all_v2/prnn_nonlin_1L8_12m_sigmoid/samples512_run9")  # run 9 is median
    # prnn_model_loc = ("../trained_models/train_vary_all_v2/prnn_nonlin_1L8_12m_sigmoid/samples512_run1")


    sim = GradedDiskSimulation(
        macro_meshsize=0.02,
        R_inner=0.2,
        R_outer=0.5,
        output_folder=output_folder,
        prnn_model_loc=prnn_model_loc,
    )

    # --- Run 1: Uniform material (theta=0 everywhere) ---
    sim.apply_uniform_material(vfrac=0.3, ratio=2.0)
    sim.run(write_output=True, write_name="disk_uniform")
    vm_uniform = sim.get_von_mises_at_qp()
    du_uniform = sim.get_nodal_displacements()

    # --- Run 2: Circumferential theta grading ---
    sim.reset()
    sim.apply_circumferential_grading()
    # sim.apply_circumferential_grading(mu_inner=FUNGI_MU, mu_outer=FUNGI_MU)
    sim.run(write_output=True, write_name="disk_graded")
    vm_graded = sim.get_von_mises_at_qp()
    du_graded = sim.get_nodal_displacements()

    # --- Post-processing ---
    plot_grading_overview(sim, output_folder)
    plot_stress_comparison(sim, vm_uniform, du_uniform, vm_graded, du_graded, output_folder=output_folder, disp_scale=1)

    # Summary
    print(f"\n--- Stress summary ---")
    print(f"Uniform  — peak VM: {vm_uniform.max():.4f}, mean: {vm_uniform.mean():.4f}")
    print(f"Graded   — peak VM: {vm_graded.max():.4f}, mean: {vm_graded.mean():.4f}")
    print(f"Peak reduction: {100*(1 - vm_graded.max()/vm_uniform.max()):.1f}%")
