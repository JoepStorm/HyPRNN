"""(Hy)PRNN surrogate material for macro-scale FE.

Loads a trained (Hy)PRNN (settings + normalizers + params) and exposes it as a
dolfinx_materials `Material`, returning homogenized stress and tangent at each
macro quadrature point — a fast drop-in replacement for the RVE.
"""
from dolfinx.common import Timer

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from material_params import FUNGI_LAMBDA

# Enable float64 for inference (required for Newton-Raphson convergence)
# Training scripts should NOT import this module to keep float32 training
jax.config.update("jax_enable_x64", True)

from scripts_surrogates.HyPRNN import HyPRNN
from scripts_surrogates.StandardNN import StandardNN
from scripts_surrogates.data_utils import LDDataset, load_settings, create_Q_matrix, matrix_to_tensor, tensor_to_matrix, load_params

from dolfinx_materials.generic import Material
import scripts_materials.neohooke as neohooke


def _convert_to_float64(pytree):
    """Recursively convert all arrays in a pytree to float64.

    This is used to convert model parameters from float32 (training precision)
    to float64 (inference precision) for better Newton-Raphson convergence.
    """
    def convert_leaf(x):
        if isinstance(x, (jnp.ndarray, np.ndarray)):
            return jnp.asarray(x, dtype=jnp.float64)
        return x
    return jax.tree_util.tree_map(convert_leaf, pytree)


def complete_update_pk2(model, F_vec, material, dataset, params, Q, micro_params_single=None):
    """PRNN update function using E -> PK2 formulation, then converting back to PK1.

    Args:
        model: PRNN or NN model instance
        F_vec: [4] - deformation gradient in tensor form
        material: material dictionary for PRNN
        dataset: dataset object containing normalizers
        params: PRNN model parameters
        Q: [2, 2] - rotation matrix for material orientation
        micro_params_single: [1, k] - micro parameter vector (or None)

    Returns:
        PK1 stress in tensor form [4]
    """
    F = tensor_to_matrix(F_vec)
    Q_T = jnp.moveaxis(Q, -2, -1)

    # Compute equivalent deformation gradient in rotated frame
    F_eq = Q_T @ F @ Q
    F_eq_T = jnp.moveaxis(F_eq, -2, -1)

    # Compute right Cauchy-Green tensor and Green-Lagrange strain
    E = 0.5 * (F_eq_T @ F_eq - jnp.eye(2))

    # Reshape E for PRNN: [2, 2] -> [1, 1, 2, 2]
    E_batch = E[None, None, :, :]

    if micro_params_single is not None:
        micro_params_expanded = micro_params_single[None, :]
    else:
        micro_params_expanded = None

    # Apply PRNN to get PK2_eq (normalized)
    pred_PK2_eq_normalized = model.apply(params, E_batch, material, micro_params=micro_params_expanded)[0, 0]

    # Denormalize PK2
    pred_PK2_eq_tensor = dataset.stress_normalizer.denormalize(pred_PK2_eq_normalized)

    # Convert tensor to matrix (symmetric)
    pred_PK2_eq = tensor_to_matrix(pred_PK2_eq_tensor, symmetric=True)

    # Back-rotate PK2 from equivalent frame: PK2 = Q @ PK2_eq @ Q^T
    PK2 = Q @ pred_PK2_eq @ Q_T

    # Convert PK2 to PK1: PK1 = F @ PK2
    PK1 = F @ PK2

    return matrix_to_tensor(PK1)


class PRNNMaterial(Material):
    """PRNN-based material model for FE² multiscale analysis.

    This material uses a Physics-informed Recurrent Neural Network (PRNN)
    as a surrogate for expensive RVE computations.

    Precision handling:
        - Training is done in float32 for efficiency
        - Inference uses float64 for Newton-Raphson convergence
        - Model parameters are automatically converted to float64 on load
        - All inputs/outputs are float64 compatible with dolfinx/PETSc
    """

    def __init__(self, settings_file, normparams_file, params_file, micro_variables):
        super().__init__()
        """
        Initialize the PRNN material model with loaded parameters.
        settings_file, normparams_file, params_file: paths to the settings, normalizers and model parameters files.
        micro_variables: dictionary with micro variables required for the model. E.g. {'Vfrac': Vfrac_array, 'thetas': theta_array, 'ratio': ratio_array, 'mu': mu_array}
        """
        self.mu_vals = micro_variables['mu']
        thetas = micro_variables['theta']  # Assuming 'theta' is in radians and has shape [num_samples]

        # find number of samples from micro_variables by looking at the length of the first entry
        num_IP_points = micro_variables[list(micro_variables.keys())[0]].shape[0]

        ## load settings
        settings = load_settings(settings_file)

        # Create a proxy dataset class to load the normalizers required during inference.
        # Convert normalizers to float64 for inference precision.
        proxyData = LDDataset.__new__(LDDataset)
        proxyData.loadDataparams(normparams_file)
        proxyData.stress_normalizer = _convert_to_float64(proxyData.stress_normalizer)
        proxyData.M_normalizer = _convert_to_float64(proxyData.M_normalizer)

        model_type = settings.get('model_type', 'hyprnn')

        # Load model parameters and convert to float64 for inference precision
        params_f32 = load_params(params_file)
        self.params = _convert_to_float64(params_f32)

        if model_type == 'nn':
            self.model = StandardNN(
                n_outputs=3,
                hidden_sizes=tuple(settings['nn_hidden_sizes']),
                activation=settings.get('nn_activation', 'sigmoid'),
                use_bias=settings.get('nn_bias', True),
            )
        else:
            use_multiple_materials = isinstance(settings['mat_points'], list)
            n_matpts = tuple(settings['mat_points']) if use_multiple_materials else settings['mat_points']
            self.model = HyPRNN(
                n_matpts=n_matpts,
                n_outputs=3,
                encoder_type=settings['encoder_type'],
                stress_normalizer=proxyData.stress_normalizer,
                material_micro_features=tuple(settings['mat_micro_features']),
                shared_micro_features=tuple(settings.get('shared_micro_features', [])),
                hidden_dim=settings.get('hidden_dim', 8),
                w_max=settings.get('w_max', 1.25),
                hyper_hidden_sizes=tuple(settings['hyper_hidden_sizes']) if settings.get('hyper_hidden_sizes') else None,
                hyper_activation=settings.get('hyper_activation', 'sigmoid'),
                encoder_n_layers=settings.get('encoder_n_layers', 3),
                encoder_activation=settings.get('encoder_activation', 'softplus'),
                use_multiple_materials=use_multiple_materials,
            )

        def get_material_for_sample(mu_val):
            return neohooke.create_material(lambda_=FUNGI_LAMBDA, mu_=mu_val)

        # Pre-compute Q matrices for all samples based on micro_variables
        self.Q_matrices = jnp.zeros((num_IP_points, 2, 2), dtype=jnp.float64)
        self.Q_matrices = create_Q_matrix(self.Q_matrices, jnp.asarray(thetas, dtype=jnp.float64))

        # Extract subset of micro_variables based on mat_parameters
        mat_param_keys = settings['mat_parameters']
        if isinstance(mat_param_keys, list):
            # Stack multiple parameters: [num_IP_points, num_params]
            self.micro_params = jnp.stack([jnp.asarray(micro_variables[key], dtype=jnp.float64) for key in mat_param_keys], axis=-1)
        else:
            # Single parameter: [num_IP_points, 1]
            self.micro_params = jnp.asarray(micro_variables[mat_param_keys], dtype=jnp.float64)[:, None]

        # Normalize micro_params (normalizer already converted to float64 above)
        self.micro_params = proxyData.M_normalizer.normalize(self.micro_params)

        # Convert mu_vals to float64
        self.mu_vals = jnp.asarray(self.mu_vals, dtype=jnp.float64)

        # JIT compiled stress function with vmap over material parameters
        def complete_update_single(F_single, Q_single, micro_params_single, mu_val):
            material = get_material_for_sample(mu_val)
            return complete_update_pk2(self.model, F_single,
                                   material, proxyData, self.params,
                                   Q_single, micro_params_single)

        self.update_stress_jit = jit(vmap(complete_update_single, in_axes=(0, 0, 0, 0)))

        # Tangent computation using AD - only compute for F00, F11, F01, then copy to F10
        def compute_tangent_single(F_single, Q_single, micro_params_single, mu_val):
            return jax.jacrev(lambda F: complete_update_single(F, Q_single, micro_params_single, mu_val))(F_single)

            # # Compute full jacobian
            # C_full = jax.jacrev(lambda F: complete_update_single(F, Q_single, micro_params_single, mu_val))(F_single)
            #
            # # Copy F01 columns/rows to F10 for symmetry (indices 2 -> 3)
            # # This treats F01 and F10 as coupled, which improves convergence
            # C_sym = C_full.at[:, 3].set(C_full[:, 2])  # Copy column 2 to column 3
            # C_sym = C_sym.at[3, :].set(C_sym[2, :])  # Copy row 2 to row 3
            #
            # return C_sym

        self.compute_tangent_jit = jit(vmap(compute_tangent_single, in_axes=(0, 0, 0, 0)))


    @property
    def fluxes(self):
        return {"PK1": 4}

    @property
    def gradients(self):
        return {"F": 4}

    @property
    def internal_state_variables(self):
        return {}


    def constitutive_update_vectorized(self, F, state_all):
        # Compute stress
        with Timer("PRNN: Constitutive integration"):
            with Timer("Constitutive PRNN stress: "):
                stress_vect = self.update_stress_jit(F, self.Q_matrices, self.micro_params, self.mu_vals)

            # Compute tangent using AD
            with Timer("Constitutive PRNN tangent: "):
                tangent = self.compute_tangent_jit(F, self.Q_matrices, self.micro_params, self.mu_vals)

        state_all['cauchy_stress'] = stress_vect

        return stress_vect, tangent, state_all


    def integrate(self, gradients: np.ndarray, dt: float = 0.0):
        """
        Integrate the constitutive response at all Gauss points.
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


        with Timer("PRNN: Constitutive integration"):
            stresses, tangents, _ = self.constitutive_update_vectorized(gradients, {})

        self.data_manager.s1.fluxes[:] = stresses

        # No internal state variables for this material - this is handled internally due to data type differences.
        isv = np.zeros((ngauss, 0))

        return stresses, isv, tangents
