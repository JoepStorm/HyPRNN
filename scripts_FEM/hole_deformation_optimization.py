"""Hole-bulge design with a filler PRNN surrogate.

A plate with a central hole is compressed between two rigid platens (one-sided
penalty contact on the top and bottom edges). The material at each element is a
PRNN surrogate parametrized by a filler fraction (``fil_frac``) and a fiber
orientation (``theta``, applied as a rotation of F, not seen by the PRNN).

Three entry points (select via MODE at the bottom):
    - optimize : CMA-ES over a hex-region grading of fil_frac + theta.
    - plot     : reload saved optimization results and redraw the field plots.
    - baseline : random non-symmetric theta + uniform fil_frac, for comparison.
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import gc
import numpy as np
import jax.numpy as jnp
from mpi4py import MPI
import ufl
from dolfinx import fem, io, mesh
from dolfinx.io import gmsh as gmshio
from petsc4py import PETSc
import gmsh

from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.solvers import NonlinearMaterialProblem
from dolfinx_materials.utils import create_quadrature_functionspace
from scripts_materials.prnn_material import PRNNMaterial
from material_params import FUNGI_MU, FUNGI_LAMBDA
from plotting_utils import plot_interpolation_field

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse
from matplotlib import rc
plt.style.use(['science', 'bright'])
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
rc('text', usetex=True)


# ── Design constants ─────────────────────────────────────────────────────────

# Fixed neo-Hookean reference parameters (the filler PRNN only grades fil_frac).
MU_REF = FUNGI_MU
LAMBDA_REF = FUNGI_LAMBDA

# Orientation range (radians) that normalized theta controls map to. Symmetric
# about 0 so a mirror sign flip keeps values in range.
THETA_BOUNDS = (-np.pi / 2, np.pi / 2)

# ── Hex grid helpers (pointy-top axial coordinates) ──────────────────────────

def pixel_to_hex_axial(x, y, size):
    q = (2 / 3 * x) / size
    r = (-1 / 3 * x + np.sqrt(3) / 3 * y) / size
    return q, r


def axial_round(q_frac, r_frac):
    s_frac = -q_frac - r_frac
    rq, rr, rs = np.round(q_frac), np.round(r_frac), np.round(s_frac)
    dq, dr, ds = np.abs(rq - q_frac), np.abs(rr - r_frac), np.abs(rs - s_frac)
    fix_q = (dq > dr) & (dq > ds)
    fix_r = (~fix_q) & (dr > ds)
    rq = np.where(fix_q, -rr - rs, rq)
    rr = np.where(fix_r, -rq - rs, rr)
    return rq.astype(int), rr.astype(int)


def hex_center(q, r, size):
    x = size * (3 / 2 * q)
    y = size * (np.sqrt(3) / 2 * q + np.sqrt(3) * r)
    return x, y


# ── Mesh ─────────────────────────────────────────────────────────────────────

def create_hole_mesh(length, height, hole_radius, lc):
    """Rectangular plate of given size with a centered circular hole."""
    gmsh.initialize()
    gdim = 2
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, length, height, tag=1)
    hole = gmsh.model.occ.addDisk(length / 2, height / 2, 0, hole_radius, hole_radius)
    gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, hole)])
    gmsh.model.occ.synchronize()

    volumes = gmsh.model.getEntities(gdim)
    gmsh.model.addPhysicalGroup(gdim, [volumes[0][1]], 1, name="Plate")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.model.mesh.generate(gdim)

    domain = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=gdim).mesh
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    gmsh.finalize()
    return domain


# ── Base simulation: plate with hole, penalty platens, PRNN material ─────────

class BulgeHole:
    """Plate-with-hole compressed by one-sided penalty platens on top/bottom.

    Subclasses populate ``self.micro_variables`` (keys fil_frac, mu, lambda, theta) via an ``apply_*`` method before calling ``run``.
    """

    def __init__(self, macro_meshsize=0.02, hole_radius=0.25, plate_height=1.0, output_folder='../results/', prnn_model_loc='', k_penalty=1e5, max_disp=0.2):
        self.output_folder = output_folder
        self.prnn_model_loc = prnn_model_loc
        self.k_penalty = k_penalty
        self.dim = 2

        self.plate_size = 1.0
        self.plate_height = plate_height
        self.hole_radius = hole_radius
        self.hole_center = [self.plate_size / 2, self.plate_height / 2]

        # Load stepping (negative = downward compression).
        self.step_size_init = -0.05
        self.step_size_factor = 0.5
        self.step_size_min = 0.001
        self.max_disp = max_disp
        self.output_name = "bulge"

        self.domain = create_hole_mesh(self.plate_size, self.plate_height,
                                       self.hole_radius, macro_meshsize)
        self._setup_function_spaces()
        self._setup_bcs_penalty()
        self._setup_hole_boundary_dofs()

        # Quadrature coordinates (for grading). A matching space is rebuilt in qmap.
        WJ = create_quadrature_functionspace(self.domain, self.deg_quad, self.order)
        self.qp_coords = WJ.tabulate_dof_coordinates()[:, :2]

    # -- setup ---------------------------------------------------------------

    def _setup_function_spaces(self):
        self.order = 1
        self.deg_quad = 2 * (self.order - 1)
        self.V = fem.functionspace(self.domain, ("P", self.order, (2,)))
        self.num_elems = self.V.dofmap.list.shape[0]
        self.ips_per_elem = 1
        self.num_ips = self.num_elems * self.ips_per_elem
        print(f"num_elements: {self.num_elems}, num_ips: {self.num_ips}")

    def _setup_bcs_penalty(self):
        """One-sided penalty platens on top/bottom; pin x at top/bottom centers."""
        fdim = self.dim - 1

        def bottom_edge(x):
            return np.isclose(x[1], 0.0)

        def top_edge(x):
            return np.isclose(x[1], self.plate_height)

        top_facets = mesh.locate_entities_boundary(self.domain, fdim, top_edge)
        bot_facets = mesh.locate_entities_boundary(self.domain, fdim, bottom_edge)

        TOP_TAG, BOT_TAG = 1, 2
        indices = np.concatenate([top_facets, bot_facets]).astype(np.int32)
        values = np.concatenate([
            np.full(len(top_facets), TOP_TAG, dtype=np.int32),
            np.full(len(bot_facets), BOT_TAG, dtype=np.int32),
        ])
        order = np.argsort(indices)
        self._penalty_facet_tags = mesh.meshtags(self.domain, fdim, indices[order], values[order])
        self._penalty_top_tag, self._penalty_bot_tag = TOP_TAG, BOT_TAG

        # Pin horizontal rigid-body motion at top/bottom centers.
        bot_verts = mesh.locate_entities_boundary(self.domain, 0, bottom_edge)
        top_verts = mesh.locate_entities_boundary(self.domain, 0, top_edge)
        vcoords_bot = self.domain.geometry.x[bot_verts]
        vcoords_top = self.domain.geometry.x[top_verts]
        bot_center = bot_verts[[np.argmin(np.abs(vcoords_bot[:, 0] - self.plate_size / 2))]]
        top_center = top_verts[[np.argmin(np.abs(vcoords_top[:, 0] - self.plate_size / 2))]]
        bot_center_dof_x = fem.locate_dofs_topological(self.V.sub(0), 0, bot_center)
        top_center_dof_x = fem.locate_dofs_topological(self.V.sub(0), 0, top_center)

        self.u_top = fem.Constant(self.domain, PETSc.ScalarType(0.0))
        self.bcs = [
            fem.dirichletbc(PETSc.ScalarType(0.0), bot_center_dof_x, self.V.sub(0)),
            fem.dirichletbc(PETSc.ScalarType(0.0), top_center_dof_x, self.V.sub(0)),
        ]

    def _setup_hole_boundary_dofs(self):
        """Hole-boundary DOFs sorted by angle, plus leftmost/rightmost indices."""
        def on_hole(x):
            d = np.sqrt((x[0] - self.hole_center[0]) ** 2 + (x[1] - self.hole_center[1]) ** 2)
            return np.isclose(d, self.hole_radius, atol=1e-2)

        verts = mesh.locate_entities_boundary(self.domain, 0, on_hole)
        dofs = fem.locate_dofs_topological(self.V, 0, verts)
        coords = self.V.tabulate_dof_coordinates()[dofs, :2]

        cx, cy = self.hole_center
        order = np.argsort(np.arctan2(coords[:, 1] - cy, coords[:, 0] - cx))
        self.all_hole_dofs = dofs[order]
        self.all_hole_coords = coords[order]
        self._hole_li = int(np.argmin(self.all_hole_coords[:, 0]))
        self._hole_ri = int(np.argmax(self.all_hole_coords[:, 0]))

    # -- measurements --------------------------------------------------------

    def get_hole_bulge(self):
        """Horizontal change in hole diameter (negative = contraction)."""
        ux = self.u.x.array[2 * self.all_hole_dofs]
        return float(ux[self._hole_ri] - ux[self._hole_li])

    def bulge_loss(self):
        return float(abs(self.get_hole_bulge()))

    def get_deformed_hole_coords(self):
        u = self.u.x.array
        ux = u[2 * self.all_hole_dofs]
        uy = u[2 * self.all_hole_dofs + 1]
        return self.all_hole_coords + np.column_stack([ux, uy])

    def get_deformed_outer_coords(self):
        """Ordered outer-boundary coords displaced by u (deformed perimeter)."""
        L, H = self.plate_size, self.plate_height

        def outer(x):
            return (np.isclose(x[0], 0.0) | np.isclose(x[0], L) |
                    np.isclose(x[1], 0.0) | np.isclose(x[1], H))

        verts = mesh.locate_entities_boundary(self.domain, 0, outer)
        dofs = fem.locate_dofs_topological(self.V, 0, verts)
        coords = self.V.tabulate_dof_coordinates()[dofs, :2]
        order = np.argsort(np.arctan2(coords[:, 1] - H / 2, coords[:, 0] - L / 2))
        dofs, coords = dofs[order], coords[order]
        ux = self.u.x.array[2 * dofs]
        uy = self.u.x.array[2 * dofs + 1]
        return coords + np.column_stack([ux, uy])

    def get_deformed_qp_coords(self):
        """Quadrature-point coords displaced by u (deformed configuration).

        Assumes 1 IP/element (qp i lives in cell i)
        """
        pts = np.zeros((len(self.qp_coords), 3))
        pts[:, :2] = self.qp_coords
        cells = np.arange(len(self.qp_coords), dtype=np.int32)
        u_vals = self.u.eval(pts, cells)[:, :2]
        return self.qp_coords + u_vals

    def get_qp_rotation(self):
        """Local rigid-rotation angle (rad) per qp from the polar part of F."""
        W = fem.functionspace(self.domain, ("DG", 0, (2, 2)))
        g = fem.Function(W)
        g.interpolate(fem.Expression(ufl.grad(self.u), W.element.interpolation_points))
        F = g.x.array.reshape(-1, 2, 2) + np.eye(2)
        return np.arctan2(F[:, 1, 0] - F[:, 0, 1], F[:, 0, 0] + F[:, 1, 1])

    # -- problem -------------------------------------------------------------

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

        loc = self.prnn_model_loc
        try:
            self.material = PRNNMaterial(f"{loc}_settings", f"{loc}_normparams",
                                         f"{loc}", self.micro_variables)
        except Exception:
            self.material = PRNNMaterial(f"{loc}settings", f"{loc}normparams",
                                         f"{loc}params", self.micro_variables)

        self.qmap = QuadratureMap(self.domain, self.deg_quad, self.material)
        self.qmap.register_gradient("F", self._F(self.u))
        P = self.qmap.fluxes["PK1"]
        self.Res = ufl.dot(P, self._dF(self.u, self.v)) * self.qmap.dx

        # One-sided penalty: top platen pushes down when u_y > u_top, bottom
        # platen pushes up when u_y < 0. Inactive (zero force) otherwise.
        u_y, v_y = self.u[1], self.v[1]
        k = fem.Constant(self.domain, PETSc.ScalarType(self.k_penalty))
        ds_pen = ufl.Measure("ds", domain=self.domain, subdomain_data=self._penalty_facet_tags)
        self.Res += k * ufl.max_value(u_y - self.u_top, 0.0) * v_y * ds_pen(self._penalty_top_tag)
        self.Res -= k * ufl.max_value(-u_y, 0.0) * v_y * ds_pen(self._penalty_bot_tag)

        self.Jac = self.qmap.derivative(self.Res, self.u, self.du)

    def _setup_solver(self):
        petsc_options = {
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
            petsc_options_prefix="BULGE", petsc_options=petsc_options,
        )

    def run(self, write_output=True, write_name=None):
        """Incrementally compress to max_disp. Returns step count or False."""
        self._setup_material_and_problem()
        self._setup_solver()
        if write_name is None:
            write_name = f"bulge_{self.output_name}"

        print(f"\nStarting bulge simulation ({self.output_name})...")
        if write_output:
            micro_funcs = self._create_micro_variable_functions()
            self._create_stress_function()
            all_funcs = [self.u] + micro_funcs + [self._vm_stress_func]
            with io.VTXWriter(MPI.COMM_WORLD, f"{self.output_folder}/{write_name}.bp", all_funcs) as xf:
                xf.write(0.0)
                steps = self._solve_loop(xf)
        else:
            steps = self._solve_loop()
        print(f"Simulation completed -- output: {self.output_folder}/{write_name}.bp")
        return steps

    def _solve_loop(self, xf=None):
        cur_step = self.step_size_init
        cur_load = 0.0
        steps = 0
        converged = True
        while abs(cur_load) < self.max_disp - 1e-10:
            if converged:
                self.u_pre.x.array[:] = self.u.x.array[:]
                steps += 1
                cur_load += cur_step
            else:
                self.u.x.array[:] = self.u_pre.x.array[:]
                self._destroy_problem()
                self._setup_solver()
                print(f"Step {abs(cur_load + cur_step):.4f} did not converge, reducing step.")
                cur_step *= self.step_size_factor
                cur_load += cur_step

            if abs(cur_step) < self.step_size_min:
                print("Step size too small, stopping simulation.")
                return False

            self.u_top.value = cur_load
            try:
                self.problem.solve()
                converged = self.problem.solver.getConvergedReason() > 0
                nr_iters = self.problem.solver.getIterationNumber()
                print(f"Step {steps}: disp={abs(cur_load):.4f}, "
                      f"{'OK' if converged else 'FAILED'}, iters={nr_iters}")
            except Exception:
                print("Error while solving loop.")
                converged = False

            if converged:
                if xf is not None:
                    self._update_stress_function()
                    xf.write(steps)
            else:
                cur_load -= cur_step
        return steps

    # -- VTX output fields ---------------------------------------------------

    def _create_micro_variable_functions(self):
        V_DG0 = fem.functionspace(self.domain, ("DG", 0))
        funcs = []
        for name, values in self.micro_variables.items():
            f = fem.Function(V_DG0, name=name)
            f.x.array[:] = np.asarray(values)
            funcs.append(f)
        return funcs

    def _create_stress_function(self):
        V_DG0 = fem.functionspace(self.domain, ("DG", 0))
        self._vm_stress_func = fem.Function(V_DG0, name="von_mises")
        self._vm_stress_func.x.array[:] = 0.0

    def _update_stress_function(self):
        pk1 = self.qmap.fluxes["PK1"].x.array.reshape(-1, 4)
        P11, P22, P12, P21 = pk1[:, 0], pk1[:, 1], pk1[:, 2], pk1[:, 3]
        Sxy = 0.5 * (P12 + P21)
        self._vm_stress_func.x.array[:] = np.sqrt(P11**2 + P22**2 - P11 * P22 + 3 * Sxy**2)

    # -- teardown (release PETSc/MPI objects between runs) -------------------

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
        """Reset to undeformed state. No-op before the first run()."""
        if not hasattr(self, 'u'):
            return
        self.u.x.array[:] = 0.0
        self.u_pre.x.array[:] = 0.0
        self.u_top.value = 0.0
        self._destroy_problem()
        self._destroy_qmap()
        if hasattr(self, 'qmap'):
            if hasattr(self.qmap, 'reset'):
                self.qmap.reset()
            del self.qmap
        gc.collect()

    # -- plotting ------------------------------------------------------------

    def plot_deformed_hole(self, output_folder, fname, title=''):
        os.makedirs(output_folder, exist_ok=True)
        deformed = self.get_deformed_hole_coords()
        deformed_closed = np.vstack([deformed, deformed[0]])

        angles = np.linspace(0, 2 * np.pi, 200)
        circle_x = self.hole_center[0] + self.hole_radius * np.cos(angles)
        circle_y = self.hole_center[1] + self.hole_radius * np.sin(angles)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(circle_x, circle_y, 'k--', lw=1, alpha=0.3, label='Original hole')
        line = ax.plot(deformed_closed[:, 0], deformed_closed[:, 1], '-', lw=1.5,
                       label='Deformed hole')
        ax.scatter(deformed_closed[:, 0], deformed_closed[:, 1],
                   c=line[0].get_color(), s=4)
        ax.set_aspect('equal')
        ax.legend(fontsize=9)
        ax.set_title(title)
        ax.axis('off')
        path = os.path.join(output_folder, fname)
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")


# ── Ellipse field plots (shared by hex-graded and baseline sims) ─────────────

def _ellipse_patches(q, th, ratio, base):
    return [
        Ellipse(xy=(xi, yi), width=ratio * base, height=base, angle=np.rad2deg(ti))
        for (xi, yi), ti in zip(q, th)
    ]


def _finish_ellipse_plot(fig, ax, pc, sim, output_folder, fname, hole_coords=None,
                         outer_coords=None):
    ax.add_collection(pc)
    if outer_coords is not None:
        # Undeformed reference outline (rectangle + hole) in gray.
        L, H = sim.plate_size, sim.plate_height
        ax.plot([0, L, L, 0, 0], [0, 0, H, H, 0], '-', color='gray', lw=0.8, alpha=0.5)
        angles = np.linspace(0, 2 * np.pi, 200)
        ax.plot(sim.hole_center[0] + sim.hole_radius * np.cos(angles),
                sim.hole_center[1] + sim.hole_radius * np.sin(angles),
                '-', color='gray', lw=0.8, alpha=0.5)
        closed = np.vstack([outer_coords, outer_coords[0]])
        ax.plot(closed[:, 0], closed[:, 1], 'k-', lw=1.0)
    if hole_coords is not None:
        closed = np.vstack([hole_coords, hole_coords[0]])
        ax.plot(closed[:, 0], closed[:, 1], 'k-', lw=1.0)
    else:
        angles = np.linspace(0, 2 * np.pi, 200)
        ax.plot(sim.hole_center[0] + sim.hole_radius * np.cos(angles),
                sim.hole_center[1] + sim.hole_radius * np.sin(angles), 'k-', lw=1.0)
    # cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.02)
    # cb.set_label(r'$\theta$ (canonical) [rad]')
    if hole_coords is not None:
        ax.autoscale_view()
    else:
        ax.set_xlim(0, sim.plate_size)
        ax.set_ylim(0, sim.plate_height)
    ax.set_aspect('equal')
    ax.axis('off')
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, fname)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_theta_ellipses(sim, output_folder, fname='theta_ellipses.pdf',
                        ratio=3.0, base_frac=1 / 60, max_ellipses=900):
    """Oriented ellipses showing the theta field.

    Ellipse angle uses the actual (mirrored) theta, but color encodes the
    canonical value (theta * mirror_sign) so mirror pairs (+45 / -45 across a
    symmetry axis) share a color. mirror_sign defaults to 1 when absent.
    """
    qp = sim.qp_coords
    theta = np.asarray(sim.micro_variables['theta'])
    mirror = np.asarray(getattr(sim, '_mirror_sign', np.ones(len(qp))))
    canon = theta * mirror
    n_skip = max(1, len(qp) // max_ellipses)
    q, th, c = qp[::n_skip], theta[::n_skip], canon[::n_skip]

    base = max(sim.plate_size, sim.plate_height) * base_frac
    tmin, tmax = THETA_BOUNDS
    pc = PatchCollection(_ellipse_patches(q, th, ratio, base),
                         cmap=mpl.colormaps['twilight'],
                         norm=mpl.colors.Normalize(vmin=tmin, vmax=tmax),
                         linewidths=0.0, alpha=0.9)
    pc.set_array(c)
    fig, ax = plt.subplots(figsize=(5, 5))
    _finish_ellipse_plot(fig, ax, pc, sim, output_folder, fname)


def plot_filler_angle_ellipses(sim, output_folder, fname='filler_angle_ellipses.pdf', ratio=3.0, base_frac=1 / 60, max_ellipses=2000, seed=0, deformed=False):
    """Combined plot: ellipse density shows (1 - fil_frac).

    Each candidate quadrature point keeps a matrix ellipse with probability
    (1 - fil_frac), so high filler regions are nearly empty and low filler
    regions are dense. Orientation is the mirrored theta; color is canonical.

    The pattern is drawn from a single quadrant and then flipped exactly across
    the vertical and horizontal axes (and to the opposite corner), guaranteeing a
    visually symmetric result regardless of the random draw and without needing
    the mesh itself to be symmetric.

    With deformed=True the ellipses are placed at the displaced qp locations and
    rotated by the local rigid rotation of F (requires a completed run()).
    """
    qp = sim.get_deformed_qp_coords() if deformed else sim.qp_coords
    theta = np.asarray(sim.micro_variables['theta'])
    if deformed:
        theta = theta + sim.get_qp_rotation()
    fil = np.clip(np.asarray(sim.micro_variables['fil_frac']), 0.0, 1.0)
    mirror = np.asarray(getattr(sim, '_mirror_sign', np.ones(len(qp))))
    canon = np.asarray(sim.micro_variables['theta']) * mirror

    n_skip = max(1, len(qp) // max_ellipses)
    q, th, c, f = qp[::n_skip], theta[::n_skip], canon[::n_skip], fil[::n_skip]

    # Keep ellipses in a single quadrant, then mirror exactly. Each reflection
    # flips the ellipse orientation sign; two reflections (opposite corner)
    # cancel. The canonical color is shared by all four copies.
    base = max(sim.plate_size, sim.plate_height) * base_frac

    # A region straddling a symmetry axis would be duplicated onto itself by the
    # flip across that axis. For such regions we instead keep their real points
    # on both sides of that axis and skip that flip (a region on x=center is thus
    # still mirrored top<->bottom, just not left<->right). Region membership uses
    # the reference (undeformed) hex assignment; baseline sims have no regions, so
    # a point is treated as "crossing" only if it sits within tol of the axis.
    rx = sim.qp_coords[::n_skip][:, 0] - sim.hole_center[0]
    ry = sim.qp_coords[::n_skip][:, 1] - sim.hole_center[1]
    if hasattr(sim, 'hex_size'):
        hq, hr = axial_round(*pixel_to_hex_axial(rx, ry, sim.hex_size))
        hkey = hq.astype(np.int64) * 100000 + hr.astype(np.int64)
        cross_x = np.zeros(len(rx), dtype=bool)
        cross_y = np.zeros(len(ry), dtype=bool)
        for k in np.unique(hkey):
            m = hkey == k
            if rx[m].min() < 0 < rx[m].max():
                cross_x |= m
            if ry[m].min() < 0 < ry[m].max():
                cross_y |= m
    else:
        cross_x = np.abs(rx) < 0.5 * base
        cross_y = np.abs(ry) < 0.5 * base

    center = q.mean(axis=0) if deformed else np.asarray(sim.hole_center)
    rel = q - center
    rng = np.random.default_rng(seed)
    keep = ((rx >= 0) | cross_x) & ((ry >= 0) | cross_y) & (rng.random(len(q)) < (1.0 - f))
    rel_b, th_b, c_b = rel[keep], th[keep], c[keep]
    cross_x_b, cross_y_b = cross_x[keep], cross_y[keep]

    flips = [(1.0, 1.0, 1.0), (-1.0, 1.0, -1.0), (1.0, -1.0, -1.0), (-1.0, -1.0, 1.0)]
    q_parts, th_parts, c_parts = [], [], []
    for fx, fy, fa in flips:
        m = np.ones(len(rel_b), dtype=bool)
        if fx == -1.0:
            m &= ~cross_x_b
        if fy == -1.0:
            m &= ~cross_y_b
        q_parts.append(center + rel_b[m] * np.array([fx, fy]))
        th_parts.append(th_b[m] * fa)
        c_parts.append(c_b[m])
    q = np.vstack(q_parts)
    th = np.concatenate(th_parts)
    c = np.concatenate(c_parts)

    tmin, tmax = THETA_BOUNDS
    pc = PatchCollection(_ellipse_patches(q, th, ratio, base),
                         cmap=mpl.colormaps['twilight'],
                         norm=mpl.colors.Normalize(vmin=tmin, vmax=tmax),
                         linewidths=0.0, alpha=0.9)
    pc.set_array(c)
    fig, ax = plt.subplots(figsize=(5, 5))
    hole_coords = sim.get_deformed_hole_coords() if deformed else None
    outer_coords = sim.get_deformed_outer_coords() if deformed else None
    _finish_ellipse_plot(fig, ax, pc, sim, output_folder, fname,
                         hole_coords=hole_coords, outer_coords=outer_coords)


# ── Hex-region grading ───────────────────────────────────────────────────────

class HexBulgeHole(BulgeHole):
    """Grading defined on hexagonal regions, optionally mirror-symmetric.

    Each region carries a filler fraction and an orientation; with quarter
    symmetry (symmetric_h=True, symmetric_v=True) regions live in one quadrant
    and are mirrored, with theta negated once per reflection.
    """

    def __init__(self, hex_size=0.1, symmetric_h=True, symmetric_v=True,
                 extreme=False, **kwargs):
        super().__init__(**kwargs)
        self.hex_size = hex_size
        self.symmetric_h = symmetric_h
        self.symmetric_v = symmetric_v
        self.extreme = extreme
        self._precompute_hex_assignments()

    def _precompute_hex_assignments(self):
        """Assign each quadrature point to a hex region (+ per-point mirror sign).

        symmetric_v mirrors left-right about the vertical axis; symmetric_h
        mirrors top-bottom. Each reflection negates theta; two reflections (a
        diagonal quadrant) cancel.
        """
        x = self.qp_coords[:, 0] - self.hole_center[0]
        y = self.qp_coords[:, 1] - self.hole_center[1]
        q_idx, r_idx = axial_round(*pixel_to_hex_axial(x, y, self.hex_size))

        canonical = {}        # canonical (q,r) -> region index
        hex_to_canonical = {}
        region_idx = np.empty(len(x), dtype=int)
        mirror_sign = np.ones(len(x))

        for i in range(len(x)):
            key = (int(q_idx[i]), int(r_idx[i]))
            if key not in hex_to_canonical:
                cx, cy = hex_center(key[0], key[1], self.hex_size)
                mc_x, mc_y = cx, cy
                if self.symmetric_v and mc_x < -1e-10:
                    mc_x = -mc_x
                if self.symmetric_h and mc_y < -1e-10:
                    mc_y = -mc_y
                if mc_x != cx or mc_y != cy:
                    mq, mr = axial_round(*[np.array([v]) for v in pixel_to_hex_axial(mc_x, mc_y, self.hex_size)])
                    canon = (int(mq[0]), int(mr[0]))
                else:
                    canon = key
                if canon not in canonical:
                    canonical[canon] = len(canonical)
                hex_to_canonical[key] = canon

            region_idx[i] = canonical[hex_to_canonical[key]]

            cx, cy = hex_center(key[0], key[1], self.hex_size)
            if self.symmetric_v and cx < -1e-10:
                mirror_sign[i] *= -1.0
            if self.symmetric_h and cy < -1e-10:
                mirror_sign[i] *= -1.0

        self._region_idx = region_idx
        self._mirror_sign = mirror_sign
        self.n_regions = len(canonical)

    def plot_hex_boundaries(self, output_folder, fname='hex_boundaries.pdf'):
        """Draw the outline of every hex region covering the domain (+ plate, hole).

        Geometry only -- independent of any simulation result.
        """
        x = self.qp_coords[:, 0] - self.hole_center[0]
        y = self.qp_coords[:, 1] - self.hole_center[1]
        q_idx, r_idx = axial_round(*pixel_to_hex_axial(x, y, self.hex_size))
        cells = set(zip(q_idx.tolist(), r_idx.tolist()))

        fig, ax = plt.subplots(figsize=(5, 5))
        corner_angles = np.deg2rad(np.arange(0, 360, 60))  # flat-top hexagons
        for q, r in cells:
            cx, cy = hex_center(q, r, self.hex_size)
            hx = self.hole_center[0] + cx + self.hex_size * np.cos(corner_angles)
            hy = self.hole_center[1] + cy + self.hex_size * np.sin(corner_angles)
            ax.plot(np.append(hx, hx[0]), np.append(hy, hy[0]),
                    '-', color='gray', lw=0.6, alpha=0.8)

        L, H = self.plate_size, self.plate_height
        ax.plot([0, L, L, 0, 0], [0, 0, H, H, 0], 'k-', lw=1.0)
        ang = np.linspace(0, 2 * np.pi, 200)
        ax.plot(self.hole_center[0] + self.hole_radius * np.cos(ang),
                self.hole_center[1] + self.hole_radius * np.sin(ang), 'k-', lw=1.0)

        ax.set_xlim(0, L)
        ax.set_ylim(0, H)
        ax.set_aspect('equal')
        ax.axis('off')
        os.makedirs(output_folder, exist_ok=True)
        path = os.path.join(output_folder, fname)
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    def apply_hex_grading(self, fil_values, theta_values, plot_grading=False):
        """Set micro_variables from per-region fil_frac and normalized theta."""
        raw = np.asarray(fil_values)[self._region_idx]
        if self.extreme:
            fil_field = np.where(raw > 0.75, 1.0, 0.5)
        else:
            fil_field = np.clip(raw, 0.0, 1.0)

        theta_norm = np.clip(np.asarray(theta_values)[self._region_idx], 0.0, 1.0)
        tmin, tmax = THETA_BOUNDS
        theta_field = (tmin + theta_norm * (tmax - tmin)) * self._mirror_sign

        self.micro_variables = {
            'fil_frac': jnp.array(fil_field),
            'mu': jnp.ones(self.num_ips) * MU_REF,
            'lambda': jnp.ones(self.num_ips) * LAMBDA_REF,
            'theta': jnp.array(theta_field),
        }
        self.output_name = "hex_graded_bulge"

        if plot_grading:
            plot_interpolation_field(
                domain=self.domain, control_coords=None,
                field_dict={'fil_frac': jnp.array(fil_field),
                            'theta': jnp.array(theta_field)},
                output_folder=f'{self.output_folder}',
            )

    def run_trial(self, fil_values, theta_values, plot_grading=False):
        try:
            self.reset()
        except Exception:
            pass
        self.apply_hex_grading(fil_values, theta_values, plot_grading=plot_grading)
        out = self.run(write_output=False)
        if isinstance(out, bool):
            print("Simulation failed to converge.")
            return None
        return self.bulge_loss()


# ── Baseline grading ─────────────────────────────────────────────────────────

class BaselineBulgeHole(BulgeHole):
    """Random, non-symmetric per-element theta with a uniform fil_frac."""

    def apply_random_grading(self, fil_value=0.0, seed=0, plot_grading=False):
        rng = np.random.default_rng(seed)
        tmin, tmax = THETA_BOUNDS
        theta_field = rng.uniform(tmin, tmax, self.num_ips)
        fil_field = np.full(self.num_ips, float(fil_value))
        self._mirror_sign = np.ones(self.num_ips)  # no symmetry → canonical == theta

        self.micro_variables = {
            'fil_frac': jnp.array(fil_field),
            'mu': jnp.ones(self.num_ips) * MU_REF,
            'lambda': jnp.ones(self.num_ips) * LAMBDA_REF,
            'theta': jnp.array(theta_field),
        }
        self.output_name = "baseline_bulge"

        if plot_grading:
            plot_interpolation_field(
                domain=self.domain, control_coords=None,
                field_dict={'fil_frac': jnp.array(fil_field),
                            'theta': jnp.array(theta_field)},
                output_folder=f'{self.output_folder}',
            )
        return fil_field, theta_field


def run_baseline(output_folder, prnn_model_loc, macro_meshsize=0.02,
                 hole_radius=0.25, plate_height=1.0, fil_value=0.0, seed=0,
                 max_disp=0.2):
    """Random-orientation baseline: grading plots, deformed hole, and VTX file.

    fil_value sets the uniform filler fraction (default 0; change to compare).
    """
    os.makedirs(output_folder, exist_ok=True)
    sim = BaselineBulgeHole(
        macro_meshsize=macro_meshsize, output_folder=output_folder,
        prnn_model_loc=prnn_model_loc, hole_radius=hole_radius,
        plate_height=plate_height, max_disp=max_disp,
    )
    sim.apply_random_grading(fil_value=fil_value, seed=seed, plot_grading=True)
    plot_theta_ellipses(sim, output_folder)
    plot_filler_angle_ellipses(sim, output_folder, seed=seed)

    sim.run(write_output=True, write_name='baseline_bulge')  # ParaView .bp
    loss = float(abs(sim.get_hole_bulge()))
    sim.plot_deformed_hole(output_folder, 'deformed_baseline.pdf',
                           title=f'Baseline |bulge| = {loss:.4f}')
    plot_filler_angle_ellipses(sim, output_folder,
                               fname='filler_angle_ellipses_deformed.pdf', seed=seed, deformed=True)
    print(f"\nBaseline (fil_frac={fil_value}, seed={seed}) |bulge| = {loss:.6f}")
    return loss


# ── Optimizer ────────────────────────────────────────────────────────────────

class BulgeOptimizer:
    """Maximize |hole bulge| via per-hex fil_frac + theta grading with CMA-ES."""

    def __init__(self, output_folder, prnn_model_loc, macro_meshsize=0.02,
                 hole_radius=0.25, plate_height=1.0, hex_size=0.1,
                 symmetric_h=True, symmetric_v=True,
                 fil_bounds=(0.0, 1.0), extreme=False, max_disp=0.2):
        self.output_folder = output_folder
        self.fil_bounds = fil_bounds
        os.makedirs(output_folder, exist_ok=True)

        self.sim = HexBulgeHole(
            hex_size=hex_size, macro_meshsize=macro_meshsize,
            output_folder=output_folder, prnn_model_loc=prnn_model_loc,
            hole_radius=hole_radius, plate_height=plate_height,
            symmetric_h=symmetric_h, symmetric_v=symmetric_v, extreme=extreme,
            max_disp=max_disp,
        )

        self.n_regions = self.sim.n_regions
        self.n_params = 2 * self.n_regions  # fil block + theta block

        fil_lo, fil_hi = fil_bounds
        self.lower = np.concatenate([np.full(self.n_regions, fil_lo), np.zeros(self.n_regions)])
        self.upper = np.concatenate([np.full(self.n_regions, fil_hi), np.ones(self.n_regions)])

        self.eval_count = 0
        self.history = []
        self.worst_feasible = -np.inf
        self.failure_eps = 1e-6
        self.best_loss = np.inf

    # -- decode / objective --------------------------------------------------

    def _decode(self, x):
        """Map a raw search vector to (fil_values, theta_norm_values)."""
        x = np.asarray(x, dtype=float)
        xf, xt = x[:self.n_regions], x[self.n_regions:]
        fil = np.clip(xf, self.fil_bounds[0], self.fil_bounds[1])
        return fil, np.clip(xt, 0.0, 1.0)

    def _objective(self, x):
        fil, theta_norm = self._decode(x)
        raw = self.sim.run_trial(fil, theta_norm)
        self.eval_count += 1

        if raw is None:
            # Rank failed points just below all feasible ones (0 if none seen yet).
            base = self.worst_feasible if np.isfinite(self.worst_feasible) else 0.0
            penalty = base + self.failure_eps
            self.history.append(penalty)
            print(f"  eval {self.eval_count}: FAILED -> penalty = {penalty:.6f}")
            self._plot_iteration(penalty)
            return penalty

        self.worst_feasible = max(self.worst_feasible, raw)
        self.best_loss = min(self.best_loss, raw)
        self.history.append(raw)
        print(f"  eval {self.eval_count}: |bulge| = {raw:.6f}")
        self._plot_iteration(raw)
        return raw

    def _plot_iteration(self, loss):
        contour_dir = os.path.join(self.output_folder, 'contour_iterations')
        os.makedirs(contour_dir, exist_ok=True)
        idx = self.eval_count

        deformed = self.sim.get_deformed_hole_coords()
        pts = deformed - deformed.mean(axis=0)
        pts_closed = np.vstack([pts, pts[0]])
        np.savez(os.path.join(contour_dir, f'contour_iter_{idx:04d}.npz'),
                 deformed=pts, loss=loss)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(pts_closed[:, 0], pts_closed[:, 1], '-', lw=1.5, label='Deformed')
        ax.set_aspect('equal')
        ax.legend(fontsize=9)
        ax.set_title(f'Iter {idx}, |bulge| = {loss:.6f}')
        ax.axis('off')
        plt.savefig(os.path.join(contour_dir, f'contour_iter_{idx:04d}.pdf'),
                    bbox_inches='tight', dpi=150)
        plt.close(fig)

    # -- CMA-ES --------------------------------------------------------------

    def run_cmaes(self, x0=None, sigma0=0.3, maxiter=100, seed=1, popsize=None):
        try:
            import cma
        except ImportError:
            raise ImportError("Install cma: pip install cma")

        if x0 is None:
            x0 = (self.lower + self.upper) / 2
        if popsize is None:
            popsize = 4 + int(3 * np.log(self.n_params))

        print(f"\n{'='*60}")
        print(f"CMA-ES: {self.n_regions} regions x 2 params "
              f"(fil_frac continuous, theta continuous), total {self.n_params}")
        print(f"sigma0: {sigma0}, popsize: {popsize}, maxiter: {maxiter}")
        print(f"{'='*60}\n")

        self.eval_count = 0
        self.history = []
        self.worst_feasible = -np.inf

        opts = {
            'maxiter': maxiter, 'popsize': popsize,
            'bounds': [list(self.lower), list(self.upper)],
            'seed': seed, 'verb_disp': 1, 'verb_log': 0, 'tolfun': 1e-8,
        }
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [self._objective(s) for s in solutions])
            es.disp()

        result = es.result
        print(f"\nCMA-ES finished after {self.eval_count} evaluations")
        print(f"Best |bulge|: {result.fbest:.6f}")
        return result

    # -- reporting -----------------------------------------------------------

    def plot_convergence(self, history, fname='convergence.pdf'):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(history, '-', lw=1, alpha=0.4, label='All evaluations')
        ax.plot(np.minimum.accumulate(history), '-', lw=2, label='Best so far')
        ax.set_xlabel('Function evaluation')
        ax.set_ylabel('|bulge|')
        ax.legend()
        path = os.path.join(self.output_folder, fname)
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved convergence plot: {path}")

    def save_results(self, x_best, loss_best, history, fname='optimization_results.npz'):
        fil_best, theta_norm_best = self._decode(x_best)
        tmin, tmax = THETA_BOUNDS
        np.savez(os.path.join(self.output_folder, fname),
                 x_best=x_best, fil_best=fil_best,
                 theta_norm_best=theta_norm_best,
                 theta_best=tmin + theta_norm_best * (tmax - tmin),
                 bulge_best=loss_best, history=np.array(history),
                 n_regions=self.n_regions, theta_bounds=np.array(THETA_BOUNDS))
        print(f"Saved results: {os.path.join(self.output_folder, fname)}")

    def load_best(self, fname='optimization_results.npz'):
        data = np.load(os.path.join(self.output_folder, fname))
        if 'x_best' in data:
            return self._decode(data['x_best'])
        return np.asarray(data['fil_best']), np.asarray(data['theta_norm_best'])

    # -- plotting ------------------------------------------------------------

    def plot_theta_ellipses(self, **kwargs):
        plot_theta_ellipses(self.sim, self.output_folder, **kwargs)

    def plot_filler_angle_ellipses(self, **kwargs):
        plot_filler_angle_ellipses(self.sim, self.output_folder, **kwargs)

    def _apply_and_plot_fields(self, fil, theta_norm):
        self.sim.apply_hex_grading(fil, theta_norm, plot_grading=True)
        self.sim.plot_hex_boundaries(self.output_folder)
        self.plot_theta_ellipses()
        self.plot_filler_angle_ellipses()

    def visualize_best(self, x_best, write_output=True):
        fil, theta_norm = self._decode(x_best)
        try:
            self.sim.reset()
        except Exception:
            pass
        self._apply_and_plot_fields(fil, theta_norm)
        self.sim.run(write_output=write_output)
        loss = self.sim.bulge_loss()
        self.sim.plot_deformed_hole(self.output_folder, 'deformed_best.pdf',
                                    title=f'Best |bulge| = {loss:.4f}')
        plot_filler_angle_ellipses(self.sim, self.output_folder,
                                   fname='filler_angle_ellipses_deformed.pdf', deformed=True)
        return loss

    def run_vertical(self, fname='optimization_results.npz'):
        """Re-run the optimum but force theta = 90deg everywhere.

        Keeps the optimized fil_frac field and only overwrites the orientation,
        as a check that the optimized orientation actually matters.
        """
        fil, theta_norm = self.load_best(fname)
        try:
            self.sim.reset()
        except Exception:
            pass
        self.sim.apply_hex_grading(fil, theta_norm)
        # Overwrite orientation: vertical (90 deg) everywhere, keep optimized fil.
        self.sim.micro_variables['theta'] = jnp.full(self.sim.num_ips, np.pi / 2)
        self.sim._mirror_sign = np.ones(self.sim.num_ips)  # canonical == theta
        self.plot_theta_ellipses(fname='theta_ellipses_vertical.pdf')
        self.plot_filler_angle_ellipses(fname='filler_angle_ellipses_vertical.pdf')
        self.sim.run(write_output=True, write_name='bulge_hex_graded_vertical')
        loss = self.sim.bulge_loss()
        self.sim.plot_deformed_hole(self.output_folder, 'deformed_vertical.pdf',
                                    title=f'Vertical |bulge| = {loss:.4f}')
        plot_filler_angle_ellipses(self.sim, self.output_folder,
                                   fname='filler_angle_ellipses_vertical_deformed.pdf', deformed=True)
        print(f"Vertical |bulge| = {loss:.6f}")
        return loss

    def replot_from_results(self, fname='optimization_results.npz', run_sim=False):
        """Reload saved results and redraw fields without optimizing."""
        fil, theta_norm = self.load_best(fname)
        try:
            self.sim.reset()
        except Exception:
            pass
        self._apply_and_plot_fields(fil, theta_norm)
        if run_sim:
            self.sim.run(write_output=True)
            loss = self.sim.bulge_loss()
            self.sim.plot_deformed_hole(self.output_folder, 'deformed_best.pdf',
                                        title=f'|bulge| = {loss:.4f}')
            plot_filler_angle_ellipses(self.sim, self.output_folder,
                                       fname='filler_angle_ellipses_deformed.pdf', deformed=True)
            print(f"|bulge| = {loss:.6f}")
        return fil, theta_norm


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 'optimize' runs CMA-ES then plots; 'plot' reloads saved results and only
    # redraws the field plots; 'baseline' runs the random-orientation comparison.
    MODE = 'plot'
    # MODE = 'optimize'
    # MODE = 'baseline'

    prnn_model_loc = "../trained_models/deposition/filler/vary_matpoints/matpts6_run2/"

    hex_size = 0.1
    macro_meshsize = 0.02
    hole_radius = 0.25
    plate_height = 1.0
    seed = 1
    max_disp = 0.2       # max platen displacement (compression)
    baseline_fil = 0.0   # uniform filler fraction for the baseline (change to compare)

    if MODE == 'baseline':
        output_folder = f"../results/bulge_hole/baseline_hole_{hole_radius}_fil{baseline_fil}_disp{max_disp}_seed{seed}/"
        run_baseline(output_folder=output_folder, prnn_model_loc=prnn_model_loc,macro_meshsize=macro_meshsize, hole_radius=hole_radius, plate_height=plate_height, fil_value=baseline_fil, seed=seed, max_disp=max_disp,
        )
    else:
        output_folder = f"../results/bulge_hole/hole_{hole_radius}_hex_{hex_size}_disp{max_disp}_seed{seed}/"
        optimizer = BulgeOptimizer(output_folder=output_folder, prnn_model_loc=prnn_model_loc, macro_meshsize=macro_meshsize, hole_radius=hole_radius, plate_height=plate_height, hex_size=hex_size, symmetric_h=True, symmetric_v=True, fil_bounds=(0.0, 1.0), max_disp=max_disp)
        print(f"Hex regions (1/4 domain): {optimizer.n_regions}, "
              f"total params: {optimizer.n_params}")

        if MODE == 'plot':
            fil_best, theta_norm_best = optimizer.replot_from_results(run_sim=True)
            print(f"\nLoaded best fil_frac: {np.round(fil_best, 3)}")
            print(f"Loaded best theta (norm): {np.round(theta_norm_best, 3)}")
            # Check that orientation matters: same fil_frac but theta = 90deg.
            optimizer.run_vertical()
        else:
            result = optimizer.run_cmaes(sigma0=0.3, maxiter=100, seed=seed)
            optimizer.save_results(result.xbest, result.fbest, optimizer.history)
            optimizer.plot_convergence(optimizer.history)
            optimizer.visualize_best(result.xbest)

            fil_best, theta_norm_best = optimizer._decode(result.xbest)
            print(f"\nBest bulge: {result.fbest:.6f}")
            print(f"Best fil_frac: {np.round(fil_best, 3)}")
            print(f"Best theta (norm): {np.round(theta_norm_best, 3)}")
