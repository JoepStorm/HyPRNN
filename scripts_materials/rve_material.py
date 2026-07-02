"""
RVE Material Model for FE² Multiscale Analysis

This module provides a class-based interface to use an RVE (Representative Volume Element)
as a homogenized material model. The RVE is solved using FEniCSx with periodic boundary
conditions enforced via dolfinx_mpc.
"""

# MPI/PETSc must be imported before dolfinx to avoid initialization issues
from mpi4py import MPI
from petsc4py import PETSc

from typing import Any
import numpy as np
from numpy import ndarray, dtype
import gmsh
import ufl
import dolfinx
import dolfinx_mpc
from dolfinx import fem, default_scalar_type
from dolfinx.common import Timer, list_timings
from dolfinx.io import gmsh as gmshio
from scripts_materials.neohooke import neo_hooke_pk1_ufl

# ------------------------------------------------------------------------------
# Fix Jacobian in NonlinearProblem
# ------------------------------------------------------------------------------
from collections.abc import Iterable, Sequence
import dolfinx.fem.petsc
from dolfinx import fem as _fem
from dolfinx.la.petsc import _ghost_update
from dolfinx_mpc.assemble_matrix import assemble_matrix, assemble_matrix_nest
from dolfinx_mpc.multipointconstraint import MultiPointConstraint


def fixed_assemble_jacobian_mpc(
    u: Sequence[_fem.Function] | _fem.Function,
    jacobian: _fem.Form | Sequence[Sequence[_fem.Form]],
    preconditioner: _fem.Form | Sequence[Sequence[_fem.Form]] | None,
    bcs: Iterable[_fem.DirichletBC],
    mpc: MultiPointConstraint | Sequence[MultiPointConstraint],
    _snes: PETSc.SNES,  # type: ignore
    x: PETSc.Vec,  # type: ignore
    J: PETSc.Mat,  # type: ignore
    P: PETSc.Mat,  # type: ignore
):
    """
    Fix: Add homogenization
    """
    # Copy existing soultion into the function used in the residual and
    # Jacobian
    _ghost_update(x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore
    _fem.petsc.assign(x, u)

    if isinstance(u, Sequence):
        assert isinstance(mpc, Sequence)
        for i in range(len(u)):
            mpc[i].homogenize(u[i])
            mpc[i].backsubstitution(u[i])
    else:
        assert isinstance(u, _fem.Function)
        assert isinstance(mpc, MultiPointConstraint)
        mpc.homogenize(u)
        mpc.backsubstitution(u)

    # Assemble Jacobian
    J.zeroEntries()
    if J.getType() == "nest":
        assemble_matrix_nest(J, jacobian, mpc, bcs, diagval=1.0)  # type: ignore
    else:
        assemble_matrix(jacobian, mpc, bcs, diagval=1.0, A=J)  # type: ignore
    J.assemble()
    if preconditioner is not None:
        P.zeroEntries()
        if P.getType() == "nest":
            assemble_matrix_nest(P, preconditioner, mpc, bcs, diagval=1.0)  # type: ignore
        else:
            assemble_matrix(mpc, preconditioner, bcs, diagval=1.0, A=P)  # type: ignore

        P.assemble()


dolfinx_mpc.problem.assemble_jacobian_mpc = fixed_assemble_jacobian_mpc
# ------------------------------------------------------------------------------
# End Fix Jacobian in NonlinearProblem
# ------------------------------------------------------------------------------


class RVEMaterial:
    """
    A homogenized material model based on computational homogenization of an RVE.

    The RVE is solved with periodic boundary conditions using a fluctuation-based
    formulation. The total deformation gradient is split as:
        F_total = F_macro + grad(u_tilde)
    where u_tilde is the periodic fluctuation field.

    Parameters
    ----------
    mesh_file : str, optional
        Path to the GMSH .msh file containing the RVE mesh.
    mesh_data : tuple, optional
        Pre-created mesh as (mesh, cell_tags, domain_size). Use this instead of mesh_file
        for in-memory mesh creation.
    material_properties : dict
        Dictionary containing material properties for each phase.
    domain_size : float, optional
        Size of the square RVE domain. Defaults to 1.0.
    comm : MPI.Comm, optional
        MPI communicator. Defaults to MPI.COMM_WORLD.
    do_central_diff: bool, optional
        If True, do Central Finite difference. If False, do Forward. Defaults to True.
    visualize : bool, optional
        If True, results are written to an XDMF file. Defaults to False.
    output_file : str, optional
        Path for the output XDMF file. Defaults to "results/RVE_output.xdmf".
    """

    def __init__(
        self,
        material_properties: dict,
        mesh_file: str = None,
        mesh_data: tuple = None,
        domain_size: float = 1.0,
        comm: MPI.Comm = None,
        do_central_diff: bool = True,
        visualize: bool = False,
        output_file: str = "results/RVE_output.xdmf"
    ):
        if mesh_file is None and mesh_data is None:
            raise ValueError("Either mesh_file or mesh_data must be provided")

        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.visualize = visualize
        self.output_file = output_file
        self.step_count = 0
        self.do_central_diff = do_central_diff
        self.domain_size = domain_size
        self.step_factor = 1    # set to 1 / n_sub_increments when sub-incrementing. Determines whether to plot

        # Load mesh and setup domain
        self._setup_mesh(mesh_file, mesh_data)

        # Setup material properties
        self._setup_materials(material_properties)

        # Setup function spaces and variables
        self._setup_function_spaces()

        # Setup boundary conditions with periodic constraints
        self._setup_boundary_conditions()

        # Setup kinematics and constitutive model
        self._setup_kinematics()
        # self._setup_kinematics_v2()

        # Setup solver
        self._setup_solver()

        # Setup visualization if enabled
        if self.visualize:
            self._setup_visualization()

    def _setup_mesh(self, mesh_file: str = None, mesh_data: tuple = None):
        """Load mesh from GMSH file or use pre-created mesh data."""
        if mesh_data is not None:
            # Use pre-created mesh
            self.domain, self.cell_tags, self.domain_size = mesh_data
        else:
            # Load from file
            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 0)
            gmsh.open(mesh_file)

            msh_data = gmshio.model_to_mesh(gmsh.model, self.comm, 0, gdim=2)
            self.domain = msh_data.mesh
            self.cell_tags = msh_data.cell_tags

            gmsh.finalize()

        self.dim = self.domain.topology.dim
        self.domain.topology.create_connectivity(self.dim - 1, self.dim)

        # Compute RVE volume for homogenization
        self._vol_form = fem.form(fem.Constant(self.domain, 1.0) * ufl.dx)
        self.vol_RVE = self.comm.allreduce(
            fem.assemble_scalar(self._vol_form), op=MPI.SUM
        )


    def _setup_materials(self, material_properties: dict):
        """Setup material parameters and phase identification."""
        self.mat_props = material_properties

        # Extract tags
        self.fungi_tag = material_properties['fungi']['tag']
        self.wood_tag = material_properties['wood']['tag']

        # Set Lamé parameters for fungi phase
        fungi_props = material_properties['fungi']
        if 'mu' in fungi_props and 'lambda_' in fungi_props:
            self.mu_fungi = fungi_props['mu']
            self.lmbda_fungi = fungi_props['lambda_']
        else:
            E_f = fungi_props['E']
            nu_f = fungi_props['nu']
            self.mu_fungi = E_f / (2 * (1 + nu_f))
            self.lmbda_fungi = E_f * nu_f / ((1 + nu_f) * (1 - 2 * nu_f))

        # Set Lamé parameters for wood phase
        wood_props = material_properties['wood']
        if 'mu' in wood_props and 'lambda_' in wood_props:
            self.mu_wood = wood_props['mu']
            self.lmbda_wood = wood_props['lambda_']
        else:
            E_w = wood_props['E']
            nu_w = wood_props['nu']
            self.mu_wood = E_w / (2 * (1 + nu_w))
            self.lmbda_wood = E_w * nu_w / ((1 + nu_w) * (1 - 2 * nu_w))

        # Find cells belonging to each phase
        self.wood_cells = self.cell_tags.find(self.wood_tag)
        self.fungi_cells = self.cell_tags.find(self.fungi_tag)

        # Compute volume fractions for each phase
        self._compute_volume_fractions()

    def _compute_volume_fractions(self):
        """Compute volume fractions of each material phase."""
        one = fem.Constant(self.domain, default_scalar_type(1.0))

        # Volume of wood phase
        wood_integral = fem.form(one * ufl.dx(subdomain_data=self.cell_tags, subdomain_id=self.wood_tag))
        vol_wood = self.comm.allreduce(fem.assemble_scalar(wood_integral), op=MPI.SUM)

        # Volume of fungi phase
        fungi_integral = fem.form(one * ufl.dx(subdomain_data=self.cell_tags, subdomain_id=self.fungi_tag))
        vol_fungi = self.comm.allreduce(fem.assemble_scalar(fungi_integral), op=MPI.SUM)

        self.vf_wood = vol_wood / self.vol_RVE
        self.vf_fungi = vol_fungi / self.vol_RVE

    def _setup_function_spaces(self):
        """Create function spaces for the fluctuation field and material tags."""
        # Vector function space for fluctuation field
        self.V = fem.functionspace(self.domain, ("Lagrange", 1, (self.dim,)))

        # DG0 space for material phase identification
        Q = fem.functionspace(self.domain, ("DG", 0))
        self.material_tag = fem.Function(Q, name="Material_Tag")
        self.material_tag.x.array[:] = self.fungi_tag
        self.material_tag.x.array[self.wood_cells] = self.wood_tag

        # Fluctuation field (the unknown)
        self.u_tilde = fem.Function(self.V)
        self.u_tilde.name = "Fluctuation"

    def _check_periodic_mesh(self, L):
        def left_edge(x):
            return np.isclose(x[0], 0.0) & ~np.isclose(x[1], 0.0) & ~np.isclose(x[1], L)

        def right_edge(x):
            return np.isclose(x[0], L) & ~np.isclose(x[1], 0.0) & ~np.isclose(x[1], L)

        def bottom_edge(x):
            return np.isclose(x[1], 0.0) & ~np.isclose(x[0], 0.0) & ~np.isclose(x[0], L)

        def top_edge(x):
            return np.isclose(x[1], L) & ~np.isclose(x[0], 0.0) & ~np.isclose(x[0], L)

        dofs_left = fem.locate_dofs_geometrical(self.V, left_edge)
        dofs_right = fem.locate_dofs_geometrical(self.V, right_edge)
        dofs_bottom = fem.locate_dofs_geometrical(self.V, bottom_edge)
        dofs_top = fem.locate_dofs_geometrical(self.V, top_edge)

        assert len(dofs_left) == len(dofs_right), f"Left ({len(dofs_left)}) and right ({len(dofs_right)}) boundaries must have equal DOFs"
        assert len(dofs_bottom) == len(dofs_top), f"Bottom ({len(dofs_bottom)}) and top ({len(dofs_top)}) boundaries must have equal DOFs"

        # Check that coordinates along the shared direction match between opposite boundaries
        dof_coords = self.V.tabulate_dof_coordinates()
        tol = 1e-8

        # Left-right: y-coordinates must match pairwise
        y_left = np.sort(dof_coords[dofs_left, 1])
        y_right = np.sort(dof_coords[dofs_right, 1])
        assert np.allclose(y_left, y_right, atol=tol), f"Left/right boundary nodes have mismatched y-coordinates.\n Max difference: {np.max(np.abs(y_left - y_right)):.2e}"

        # Bottom-top: x-coordinates must match pairwise
        x_bottom = np.sort(dof_coords[dofs_bottom, 0])
        x_top = np.sort(dof_coords[dofs_top, 0])
        assert np.allclose(x_bottom, x_top, atol=tol), f"Bottom/top boundary nodes have mismatched x-coordinates.\n Max difference: {np.max(np.abs(x_bottom - x_top)):.2e}"


    def _setup_boundary_conditions(self):
        """Setup Dirichlet BC at all corners, and periodic constraints on edges excluding corners"""
        L = self.domain_size

        def bot_left(x):
            return np.isclose(x[0], 0) & np.isclose(x[1], 0)

        def bot_right(x):
            return np.isclose(x[0], L) & np.isclose(x[1], 0.0)

        def top_right(x):
            return np.isclose(x[0], L) & np.isclose(x[1], L)

        def top_left(x):
            return np.isclose(x[0], 0.0) & np.isclose(x[1], L)

        # Fix all corners
        dofs_bl = fem.locate_dofs_geometrical(self.V, bot_left)
        dofs_br = fem.locate_dofs_geometrical(self.V, bot_right)
        dofs_tr = fem.locate_dofs_geometrical(self.V, top_right)
        dofs_tl = fem.locate_dofs_geometrical(self.V, top_left)
        self.bcs = [
            fem.dirichletbc(np.zeros(self.dim, dtype=default_scalar_type), dofs_bl, self.V),
            fem.dirichletbc(np.zeros(self.dim, dtype=default_scalar_type), dofs_br, self.V),
            fem.dirichletbc(np.zeros(self.dim, dtype=default_scalar_type), dofs_tr, self.V),
            fem.dirichletbc(np.zeros(self.dim, dtype=default_scalar_type), dofs_tl, self.V),
        ]

        # Periodic relation
        def periodic_relation(x):
            """Map right/top boundaries to left/bottom boundaries."""
            out_x = x[0].copy()
            out_y = x[1].copy()
            out_z = x[2].copy()
            out_x[np.isclose(x[0], L)] = 0.0
            out_y[np.isclose(x[1], L)] = 0.0
            return np.array([out_x, out_y, out_z])

        def boundary_locator(x):
            """Identify right and top boundaries, excluding corners."""
            on_right = np.isclose(x[0], L)
            on_top = np.isclose(x[1], L)
            on_left = np.isclose(x[0], 0.0)
            on_bottom = np.isclose(x[1], 0.0)

            # Exclude corners: (L,0), (0,L), (L,L)
            right_edge = on_right & ~on_top & ~on_bottom
            top_edge = on_top & ~on_left & ~on_right

            return right_edge | top_edge

        # Check if periodic mesh (equal nodes on opposite edges)
        self._check_periodic_mesh(L)

        # Setup multi-point constraints for periodicity
        self.mpc = dolfinx_mpc.MultiPointConstraint(self.V)
        self.mpc.create_periodic_constraint_geometrical(
            self.V, boundary_locator, periodic_relation, self.bcs
        )
        self.mpc.finalize()

    def _setup_kinematics(self):
        """Setup deformation gradient and stress tensor expressions.
        This is equivalent to expanding F to 3D for a plane strain problem.
        """
        # Macroscopic deformation gradient (to be updated each step)
        F_macro_init = np.eye(self.dim, dtype=default_scalar_type)
        self.F_macro = fem.Constant(self.domain, F_macro_init)

        # Total deformation gradient: F = F_macro + grad(u_tilde)
        self.F_total = ufl.variable(self.F_macro + ufl.grad(self.u_tilde))

        # First Piola-Kirchhoff stress for each phase
        P_fungi = neo_hooke_pk1_ufl(self.F_total, self.mu_fungi, self.lmbda_fungi)
        P_wood = neo_hooke_pk1_ufl(self.F_total, self.mu_wood, self.lmbda_wood)

        # Combined stress using conditional on material phase
        self.P = ufl.conditional(
            ufl.eq(self.material_tag, self.fungi_tag),
            P_fungi,
            P_wood
        )

        # Pre-compile forms for stress component integration (used in homogenization)
        self._P_forms = [
            [fem.form(self.P[i, j] * ufl.dx) for j in range(self.dim)]
            for i in range(self.dim)
        ]


    def _setup_solver(self):
        """Setup the nonlinear solver with MPC constraints."""
        v = ufl.TestFunction(self.V)

        # Weak form: balance of internal forces
        F_form = ufl.inner(self.P, ufl.grad(v)) * ufl.dx
        Jacobian = ufl.derivative(F_form, self.u_tilde, ufl.TrialFunction(self.V))

        self.problem = dolfinx_mpc.NonlinearProblem(
            F_form,
            self.u_tilde,
            bcs=self.bcs,
            mpc=self.mpc,
            J=Jacobian,

            petsc_options={
                # "snes_monitor": None,
                # "snes_linesearch_monitor": None,
                # "snes_converged_reason": None,
                # "ksp_monitor": None,
                "ksp_type": "preonly",  # No Krylov iterations needed
                "pc_type": "lu",    # Direct LU solve (exact) (sparse)
                # "pc_factor_mat_solver_type": "mumps",     # (parallel, robust)
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt", # "l2", # "bt"  # "cp" ,   # l2 appears more robust than bt in some cases
                "snes_rtol": 1e-10, #6,     # high tolerance when used inside FE2 with FD
                "snes_atol": 1e-10, #6,
                # "snes_stol": 1e-12,  # If set, make sure lower than other tolerances
                "snes_max_it": 200,
            }
        )

    def _setup_visualization(self):
        """Setup output file and visualization functions."""
        # Function space for total displacement visualization
        W = fem.functionspace(self.domain, ("Lagrange", 1, (self.dim,)))
        self.u_total_vis = fem.Function(W, name="Total_Displacement")

        # Reference coordinates for computing affine displacement
        self.x_coords = fem.Function(W)
        self.x_coords.interpolate(lambda x: x[:self.dim])

        # Stress tensor visualization
        T_stress = fem.functionspace(self.domain, ("DG", 0, (self.dim, self.dim)))
        self.P_vis = fem.Function(T_stress, name="PK1_Stress")
        self.P_expr = fem.Expression(self.P, T_stress.element.interpolation_points)

        # Open XDMF file for writing
        self.xdmf = dolfinx.io.XDMFFile(self.domain.comm, self.output_file, "w")
        self.xdmf.write_mesh(self.domain)

    def update_stress(self, F_macro_new: np.ndarray, monitor_base_solve: bool = False) -> tuple[ndarray, ndarray, bool]:
        """
        Perform one RVE solve step with a new macroscopic deformation gradient.

        Parameters
        ----------
        F_macro_new : np.ndarray
            The new macroscopic deformation gradient (2x2 array).
        monitor_base_solve : bool, optional
            If True, print SNES iteration info for the solve.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, bool]
            A tuple containing:
            - P_homogenized: The homogenized First Piola-Kirchhoff stress tensor (2x2 array).
            - sigma_homogenized: The homogenized Cauchy stress tensor (2x2 array).
            - convergence_ok: bool. Did the simulation converge?
        """
        with Timer("RVE: update_stress"):
            convergence_ok = True
            self.step_count += 1 * self.step_factor

            if monitor_base_solve:
                self.problem.solver.setMonitor(lambda snes, it, norm: print(f"  Stress solve iter {it}: {norm:.2e}"))

            # Update the macroscopic deformation gradient
            self.F_macro.value[:] = F_macro_new.astype(default_scalar_type)

            # Solve the nonlinear problem
            with Timer("RVE: stress solve"):
                _, converged, iters = self.problem.solve()

            # Disable monitor after solve
            if monitor_base_solve:
                self.problem.solver.cancelMonitor()

            if converged <= 0:
                snes_norm = self.problem.solver.getFunctionNorm()
                print(f"Warning: RVE solve did not converge. Reason: {converged}, norm: {snes_norm}, F: {F_macro_new}")
                convergence_ok = False
                # if snes_norm > 1e-3:
                #     convergence_ok = False
                #     print(f"RVE solve insufficient. Reason: {converged}, iterations: {iters}")
                #     # raise RuntimeError(f"RVE solve failed to converge. Reason: {converged}, iterations: {iters}")

            # Scatter ghost values for parallel consistency
            self.u_tilde.x.scatter_forward()

            # Compute homogenized stresses
            with Timer("RVE: homogenization"):
                P_homogenized = self._compute_homogenized_stress()

            # Compute Cauchy stress: sigma = (1/J) * P * F^T
            J = np.linalg.det(F_macro_new)
            sigma_homogenized = (1.0 / J) * P_homogenized @ F_macro_new.T

            # Write visualization output (skip during substepping or if unconverged)
            if self.visualize and convergence_ok and self.step_count == int(self.step_count):
                self._write_output(F_macro_new)

        return P_homogenized, sigma_homogenized, convergence_ok

    def update_full(
        self,
        F_macro_new: np.ndarray,
        delta: float = 1e-8,
        central_diff: bool = True,
        monitor_base_solve: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform one RVE solve step and compute tangent stiffness (using finite differences).

        Parameters
        ----------
        F_macro_new : np.ndarray
            The new macroscopic deformation gradient (2x2 array).
        delta : float, optional
            Perturbation magnitude for finite differences. Defaults to 1e-6.
        monitor_base_solve : bool, optional
            If True, print SNES iteration info for base solve.

        Depending on the setting of self.do_central_diff : bool
            If True, use central differences (more accurate, 9 solves: 1 base + 4×2 perturbed).
            If False, use forward differences (faster, 5 solves: 1 base + 4 perturbed).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing:
            - P_homogenized: The homogenized First Piola-Kirchhoff stress tensor (2x2 array).
            - sigma_homogenized: The homogenized Cauchy stress tensor (2x2 array).
            - C_full: The homogenized tangent stiffness (4x4 array).
                Rows correspond to [P_11, P_22, P_12, P_21].
                Columns correspond to [F_11, F_22, F_12, F_21].
        """
        with Timer("RVE: update_full"):
            self.step_count += 1 * self.step_factor

            if monitor_base_solve:
                self.problem.solver.setMonitor(lambda snes, it, norm: print(f"  Base solve iter {it}: {norm:.2e}"))

            # Base solve
            with Timer("RVE: stress solve"):
                self.F_macro.value[:] = F_macro_new.astype(default_scalar_type)
                _, converged, iters = self.problem.solve()

            # Disable monitor for perturbation solves
            self.problem.solver.cancelMonitor()

            if converged <= 0:
                snes_norm = self.problem.solver.getFunctionNorm()
                print(f"Warning: RVE solve did not converge. Reason: {converged}, norm: {snes_norm}, F: {F_macro_new}")
                # if snes_norm > 1e-2:
                raise RuntimeError(f"RVE solve insuffient. Reason: {converged}, iterations: {iters}")

            self.u_tilde.x.scatter_forward()

            # Compute base stress
            with Timer("RVE: homogenization"):
                P_base = self._compute_homogenized_stress()

            # Store base state for perturbations
            u_tilde_base = self.u_tilde.x.array.copy()

            # Perturbation modes for all 4 components of F: F_11, F_22, F_12, F_21
            # Note: First Piola-Kirchhoff stress P is NOT symmetric, so we need all 4 modes.
            elementary_modes = [
                np.array([[1.0, 0.0], [0.0, 0.0]]),  # dF_11
                np.array([[0.0, 0.0], [0.0, 1.0]]),  # dF_22
                np.array([[0.0, 1.0], [0.0, 0.0]]),  # dF_12
                np.array([[0.0, 0.0], [1.0, 0.0]]),  # dF_21
            ]

            C_full = np.zeros((4, 4))

            with Timer("RVE: tangent stiffness"):
                for col, mode in enumerate(elementary_modes):
                    # Forward perturbation
                    F_plus = F_macro_new + delta * mode
                    self.F_macro.value[:] = F_plus.astype(default_scalar_type)
                    self.u_tilde.x.array[:] = u_tilde_base
                    self.problem.solve()
                    self.u_tilde.x.scatter_forward()
                    P_plus = self._compute_homogenized_stress()

                    if self.do_central_diff:
                        # Backward perturbation for central difference
                        F_minus = F_macro_new - delta * mode
                        self.F_macro.value[:] = F_minus.astype(default_scalar_type)
                        self.u_tilde.x.array[:] = u_tilde_base
                        self.problem.solve()
                        self.u_tilde.x.scatter_forward()
                        P_minus = self._compute_homogenized_stress()

                        # Central difference
                        dP = (P_plus - P_minus) / (2.0 * delta)
                    else:
                        # Forward difference
                        dP = (P_plus - P_base) / delta

                    C_full[0, col] = dP[0, 0]  # dP_11/dF_col
                    C_full[1, col] = dP[1, 1]  # dP_22/dF_col
                    C_full[2, col] = dP[0, 1]  # dP_12/dF_col
                    C_full[3, col] = dP[1, 0]  # dP_21/dF_col

            # Restore base state
            self.u_tilde.x.array[:] = u_tilde_base
            self.u_tilde.x.scatter_forward()
            self.F_macro.value[:] = F_macro_new.astype(default_scalar_type)

            # Compute Cauchy stress from base P
            J = np.linalg.det(F_macro_new)
            sigma_homogenized = (1.0 / J) * P_base @ F_macro_new.T

            # Write visualization output if enabled
            if self.visualize:
                self._write_output(F_macro_new)

        return P_base, sigma_homogenized, C_full

    def _compute_homogenized_stress(self) -> np.ndarray:
        """Compute volume-averaged First Piola-Kirchhoff stress tensor."""
        with Timer("RVE: stress averaging"):
            P_avg = np.zeros((self.dim, self.dim))

            for i in range(self.dim):
                for j in range(self.dim):
                    P_int = fem.assemble_scalar(self._P_forms[i][j])
                    P_avg[i, j] = self.comm.allreduce(P_int, op=MPI.SUM) / self.vol_RVE

        return P_avg

    def compute_tangent_stiffness(
            self,
            F_macro_base: np.ndarray,
            delta: float = 1e-6
    ) -> np.ndarray:
        """
        Compute the homogenized tangent stiffness tensor using perturbation.

        Uses 4 elementary loading modes for all components of F:
        - Mode 0: dF_11 (axial X)
        - Mode 1: dF_22 (axial Y)
        - Mode 2: dF_12 (shear)
        - Mode 3: dF_21 (shear)

        Note: First Piola-Kirchhoff stress P is NOT symmetric, so all 4 modes are needed.

        Parameters
        ----------
        F_macro_base : np.ndarray
            The base macroscopic deformation gradient (2x2 array).
        delta : float, optional
            Perturbation magnitude. Defaults to 1e-6.

        Returns
        -------
        np.ndarray
            The homogenized tangent stiffness (4x4 array).
            Rows correspond to [P_11, P_22, P_12, P_21].
            Columns correspond to [F_11, F_22, F_12, F_21].
        """
        # Store current state to restore later
        u_tilde_backup = self.u_tilde.x.array.copy()
        step_backup = self.step_count

        # Elementary perturbation modes for all 4 components of F
        elementary_modes = [
            np.array([[1.0, 0.0], [0.0, 0.0]]),  # dF_11
            np.array([[0.0, 0.0], [0.0, 1.0]]),  # dF_22
            np.array([[0.0, 1.0], [0.0, 0.0]]),  # dF_12
            np.array([[0.0, 0.0], [1.0, 0.0]]),  # dF_21
        ]

        C_full = np.zeros((4, 4))

        for col, mode in enumerate(elementary_modes):
            # Forward perturbation
            F_plus = F_macro_base + delta * mode
            self.F_macro.value[:] = F_plus.astype(default_scalar_type)
            self.u_tilde.x.array[:] = u_tilde_backup
            self.problem.solve()
            self.u_tilde.x.scatter_forward()
            P_plus = self._compute_homogenized_stress()

            # Backward perturbation
            F_minus = F_macro_base - delta * mode
            self.F_macro.value[:] = F_minus.astype(default_scalar_type)
            self.u_tilde.x.array[:] = u_tilde_backup
            self.problem.solve()
            self.u_tilde.x.scatter_forward()
            P_minus = self._compute_homogenized_stress()

            # Central difference: dP/d(mode)
            dP = (P_plus - P_minus) / (2.0 * delta)

            # Extract all 4 components [P_11, P_22, P_12, P_21]
            C_full[0, col] = dP[0, 0]  # dP_11/dF_col
            C_full[1, col] = dP[1, 1]  # dP_22/dF_col
            C_full[2, col] = dP[0, 1]  # dP_12/dF_col
            C_full[3, col] = dP[1, 0]  # dP_21/dF_col

        # Restore original state
        self.u_tilde.x.array[:] = u_tilde_backup
        self.u_tilde.x.scatter_forward()
        self.F_macro.value[:] = F_macro_base.astype(default_scalar_type)
        self.step_count = step_backup

        return C_full

    def _write_output(self, F_macro_current: np.ndarray):
        """Write current state to visualization file."""
        # Compute total displacement: u_total = u_tilde + (F_macro - I) @ X
        F_minus_I = F_macro_current - np.eye(self.dim)
        x_array = self.x_coords.x.array.reshape((-1, self.dim))
        affine_disp = x_array @ F_minus_I.T
        self.u_total_vis.x.array[:] = self.u_tilde.x.array + affine_disp.flatten()

        # Interpolate stress for visualization
        self.P_vis.interpolate(self.P_expr)

        # Write to file
        self.xdmf.write_function(self.u_total_vis, self.step_count)
        self.xdmf.write_function(self.u_tilde, self.step_count)
        self.xdmf.write_function(self.P_vis, self.step_count)

    def get_fluctuation(self) -> fem.Function:
        """Return the current fluctuation field."""
        return self.u_tilde

    def reset(self):
        """Reset the fluctuation field to zero (useful for new load paths)."""
        self.u_tilde.x.array[:] = 0.0
        self.u_tilde.x.scatter_forward()

    def print_timings(self):
        """Print accumulated timing information for RVE operations."""
        list_timings(self.comm) #[TimingType.wall])

    def close(self):
        """Close output files and clean up resources."""
        if self.visualize and hasattr(self, 'xdmf'):
            self.xdmf.close()

    def __del__(self):
        """Destructor to ensure files are closed."""
        self.close()


# Example usage
if __name__ == "__main__":
    # Define material properties
    material_props = {
        'fungi': {'E': 100.0, 'nu': 0.3, 'tag': 1},
        'wood': {'E': 1000.0, 'nu': 0.3, 'tag': 2}
    }

    # Initialize the RVE material model
    rve = RVEMaterial(
        material_properties=material_props,
        mesh_file='../meshes/vfrac_ratio_big/nfib20_0.0000_2.00/rve_0.msh',
        visualize=False,
        output_file="results/rve_material.xdmf",
    )

    # Perform loading steps
    print("Starting RVE homogenization...")
    for n, eps in enumerate(np.linspace(0, 0.4, 41)):
    # for n, eps in enumerate(np.linspace(0, 0.2, 100)):
        if n == 0:
            continue

        # Define macroscopic deformation gradient (uniaxial compression in Y)
        F_macro = np.array([
            [1.0 + eps / 3, -0.2 * eps],
            [0.0, 1.0 - eps]
        ])

        print(f"\nStep {n}: eps = {eps:.4f}")

        # Update RVE and get homogenized stresses and tangent (4 solves total)
        # P_homogenized, sigma_homogenized, C_voigt = rve.update_full(F_macro, monitor_base_solve=True)
        P_homogenized, sigma_homogenized, converged = rve.update_stress(F_macro, monitor_base_solve=False)
        # print(f"F: {F_macro}")
        # print(f"Full field u: {rve.get_fluctuation().x.array[0:20]}")

        print(f"Homogenized P:\n{P_homogenized}")
        # print(f"Homogenized sigma:\n{sigma_homogenized}")
        # print(f"Tangent stiffness C (Voigt):\n{C_voigt}")

    # Print timing statistics
    rve.print_timings()

    # Clean up
    rve.close()
    print("\nDone.")
