"""
FE² Material Interface

This module provides a material interface that uses RVE (Representative Volume Element)
simulations as a constitutive model for multiscale finite element analysis (FE²).
"""
from dolfinx.common import Timer
from dolfinx_materials.generic import Material, DataManager
from scripts_materials.RVE_material import RVEMaterial
from scripts_materials.fe2mat_heterogeneous_v2 import HeterogeneousFE2Material
from scripts_materials.fe2mat_parallel import ParallelFE2Material
import numpy as np


class RVEStateManager:
    """
    Manages the state of RVE solutions at all Gauss points.

    Stores the fluctuation displacement field (u_tilde) from each RVE solve,
    allowing proper initialization for subsequent load steps.

    Currently only supports a fixed RVE mesh, and therefore a fixed number of dofs.
    """

    def __init__(self, ngauss: int, ndofs_per_rve: int):
        """
        Initialize state storage for all Gauss points.

        Parameters
        ----------
        ngauss : int
            Number of Gauss points (each has an associated RVE).
        ndofs_per_rve : int
            Number of DOFs in each RVE's displacement solution.
        """
        self.ngauss = ngauss
        self.ndofs = ndofs_per_rve

        self.u_converged = np.zeros((ngauss, ndofs_per_rve))
        self.u_trial = np.zeros((ngauss, ndofs_per_rve))

    def commit(self):
        """Accept the current trial state as the new converged state (load step converged)."""
        self.u_converged[:] = self.u_trial

    def revert(self):
        """Discard trial state and revert to last converged state (load step failed)."""
        self.u_trial[:] = self.u_converged

    def get_initial_guess(self, gauss_idx: int) -> np.ndarray:
        """Get the current trial displacement as initial guess for RVE solve."""
        return self.u_trial[gauss_idx].copy()

    def store_solution(self, gauss_idx: int, u: np.ndarray):
        """Store the RVE solution after a successful RVE solve."""
        self.u_trial[gauss_idx] = u


class FE2Material(Material):
    """
    FE² Material model using RVE simulations at each Gauss point.

    This material wraps an RVE solver and uses it to compute stress and tangent
    stiffness at each Gauss point. The RVE is solved with a prescribed macroscopic
    deformation gradient, and the homogenized response is returned.

    The formulation is 2D plane strain, using flat notation:
    - Deformation gradient F: stored as [F_11, F_22, F_12, F_21]
    - First Piola-Kirchhoff stress P: stored as [P_11, P_22, P_12, P_21]
    - Tangent C: 4x4 matrix relating dP to dF (C_ij = dP_i/dF_j)

    Parameters
    ----------
    rve : RVEMaterial
    """

    def __init__(self, rve):
        """
        Initialize the FE² material with an RVE solver.

        Parameters
        ----------
        rve : RVEMaterial
            The RVE material model instance used for microscale solves.
        """
        super().__init__()
        self.rve = rve
        self.material_properties = {}
        self._state_manager = None

    @property
    def gradients(self):
        """Define the gradient field (deformation gradient in 2D)."""
        return {"F": 4}

    @property
    def fluxes(self):
        """Define the flux field (First Piola-Kirchhoff stress in 2D)."""
        return {"PK1": 4}

    @property
    def internal_state_variables(self):
        """No explicit internal state variables; RVE state stored separately."""
        return {}

    def set_data_manager(self, ngauss: int):
        """
        Initialize data storage for the given number of Gauss points.

        Parameters
        ----------
        ngauss : int
            Number of Gauss points in the macroscale problem.
        """
        self.data_manager = DataManager(self, ngauss)

        # Initialize RVE state manager with DOF count from the RVE
        ndofs = len(self.rve.u_tilde.x.array)
        self._state_manager = RVEStateManager(ngauss, ndofs)

    def _flat_to_matrix(self, F_flat: np.ndarray) -> np.ndarray:
        return np.array([
            [F_flat[0], F_flat[2]],
            [F_flat[3], F_flat[1]]
        ])

    def _matrix_to_flat(self, P_mat: np.ndarray) -> np.ndarray:

        return np.array([P_mat[0, 0], P_mat[1, 1], P_mat[0, 1], P_mat[1, 0]])

    def constitutive_update_single(self, F_flat: np.ndarray, gauss_idx: int, dt: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform constitutive update for a single Gauss point.

        Solves the RVE problem with the given macroscopic deformation gradient
        and returns the homogenized stress and tangent stiffness.

        Parameters
        ----------
        F_flat : np.ndarray
            Flattened deformation gradient [F_11, F_22, F_12, F_21].
        gauss_idx : int
            Index of the Gauss point.
        dt : float, optional
            Time step (not used for hyperelastic RVE).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (P_flat, C_full) where P_flat is the flattened stress
            and C_full is the 4x4 tangent stiffness.
        """
        # Convert to matrix form
        F_mat = self._flat_to_matrix(F_flat)

        # Initialize RVE displacement from stored state
        u_init = self._state_manager.get_initial_guess(gauss_idx)
        self.rve.u_tilde.x.array[:] = u_init

        # Solve RVE and get homogenized response with tangent (4x4)
        P_mat, _, C_full = self.rve.update_full(F_mat, monitor_base_solve=False)

        # Store the intermediate RVE displacement
        self._state_manager.store_solution(gauss_idx, self.rve.get_fluctuation().x.array.copy())

        # Convert stress to flattened form
        P_flat = self._matrix_to_flat(P_mat)

        return P_flat, C_full

    def integrate(self, gradients: np.ndarray, dt: float = 0.0):
        """
        Integrate the constitutive response at all Gauss points.

        Loops over all Gauss points and solves an RVE for each one.

        Parameters
        ----------
        gradients : np.ndarray
            Array of shape (ngauss, 4) containing deformation gradients
            at each Gauss point in flattened form [F_11, F_22, F_12, F_21].
        dt : float, optional
            Time step

        Returns
        -------
        Tuple of (stresses, internal_state_vars, tangents) where:
            - stresses: (ngauss, 4) array of PK1 stress
            - internal_state_vars: (ngauss, 0) empty array (no ISVs)
            - tangents: (ngauss, 4, 4) array of tangent stiffness
        """
        ngauss = gradients.shape[0]

        stresses = np.zeros((ngauss, 4))
        tangents = np.zeros((ngauss, 4, 4))

        with Timer("FE2: Constitutive integration"):
            for i in range(ngauss):
                F_i = gradients[i]
                # if i == 0:
                #     print(f"F_i = {np.array2string(F_i, separator=', ')}")
                P_i, C_i = self.constitutive_update_single(F_i, i, dt)
                stresses[i] = P_i
                tangents[i] = C_i

        # Update internal state in data manager
        self.data_manager.s1.fluxes[:] = stresses

        # No internal state variables for this material - this is handled internally due to data type differences.
        isv = np.zeros((ngauss, 0))

        return stresses, isv, tangents

    def update_state(self):
        """Accept the current trial state as converged."""
        self.data_manager.update()
        self._state_manager.commit()    # was .update()

    def revert_state(self):
        """Revert to the last converged state."""
        self.data_manager.revert()
        self._state_manager.revert()

    def get_rve_displacement(self, gauss_idx: int) -> np.ndarray:
        """Get the current RVE displacement field for a specific Gauss point."""
        return self._state_manager.get_displacement(gauss_idx)

    def get_all_rve_displacements(self) -> np.ndarray:
        """Get the RVE displacement fields for all Gauss points."""
        return self._state_manager.u_s0.copy()


def create_fe2_material(
    rve_config: dict = None,
    mode: str = 'parallel',
    n_workers: int = None,
    graded_variables: dict = None,
):
    """
    Create FE² material. Three options:
    - sequential: homogenous RVEs run sequentially
    - parallel: homogenous RVEs run in parallel
    - heterogeneous: heterogeneous RVEs run in parallel
    """
    if mode == 'sequential':
        if rve_config is None:
            raise ValueError("rve_config is required for sequential mode")
        rve = RVEMaterial(
            mesh_file=rve_config['mesh_file'],
            material_properties=rve_config['material_properties'],
            do_central_diff=rve_config.get('do_central_diff', False),
            visualize=False
        )
        return FE2Material(rve)

    elif mode == 'parallel':
        if rve_config is None:
            raise ValueError("rve_config is required for parallel mode")
        return ParallelFE2Material(rve_config, n_workers=n_workers)

    elif mode == 'heterogeneous':
        if graded_variables is None:
            raise ValueError("graded_variables is required for heterogeneous mode")

        return HeterogeneousFE2Material(
            graded_variables=graded_variables,
            n_workers=n_workers,
            rve_config=rve_config,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'sequential', 'parallel', or 'heterogeneous'")

