# Neo-hooke material model implementation assuming Plane Strain
# Based on the jive material model by F.P. van der Meer
import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import NamedTuple
import ufl


class MaterialProperties(NamedTuple):
    lambda_: float
    mu_: float


def create_material(lambda_: float = 1442.307, mu_: float = 961.538) -> MaterialProperties:
    return MaterialProperties(lambda_=lambda_, mu_=mu_)


###############   Neo Hooke PK1 stress   ###############
def neo_hooke_pk1_ufl(F, mu_, lmbda_):
    """Compute first Piola-Kirchhoff stress using UFL."""
    J = ufl.det(F)
    F_inv_T = ufl.inv(F).T
    return mu_ * (F - F_inv_T) + lmbda_ * ufl.ln(J) * F_inv_T


def neo_hooke_pk1_jax_variable_params(F, mu_, lambda_):
    J = jnp.linalg.det(F)
    F_inv_T = jnp.linalg.inv(F).T
    return mu_ * (F - F_inv_T) + lambda_ * jnp.log(J) * F_inv_T


def neo_hooke_pk2_jax_variable_params(C, mu_, lambda_):
    """
    Compute second Piola-Kirchhoff stress from Green-Lagrange strain tensor.

    Args:
        C: right Cauchy-Green deformation tensor
        mu_: Second Lamé parameter (shear modulus)
        lambda_: First Lamé parameter

    Returns:
        Second Piola-Kirchhoff stress tensor
    """
    I = jnp.eye(C.shape[0])
    J = jnp.sqrt(jnp.linalg.det(C))
    C_inv = jnp.linalg.inv(C)
    return lambda_ * jnp.log(J) * C_inv + mu_ * (I - C_inv)


jit_vmap_neo_hooke_pk2_variable = jit(vmap(neo_hooke_pk2_jax_variable_params,
                                           in_axes=(0, 0, 0),
                                           out_axes=(0)))


###############   Neo Hooke Cauchy stress   ###############
def neo_hooke_update(F, material):
    """
    JAX implementation of Neo-Hookean material model.
    Made specific for 2D problems

    Args:
        F: deformation gradient tensor
        material: Material properties containing:
            lambda_: First Lamé parameter
            mu_: Second Lamé parameter (shear modulus)

    Returns:
        tuple: (stress, stiffness) - Cauchy stress in Voigt notation and tangent stiffness matrix
    """
    # Define problem dimensions
    I = jnp.eye(2)

    # # Current deformation gradient based on previous and increment
    # F = jnp.matmul(df, F0)

    # Jacobian and its logarithm
    jac = jnp.linalg.det(F)
    lnJ = jnp.log(jac)

    # Left Cauchy-Green tensor B = F·F^T
    B = F @ F.T

    # Kirchhoff stress tensor
    tau = material.lambda_ * lnJ * I + material.mu_ * (B - I)

    # Cauchy stress
    sigma = tau / jac

    # Convert to Voigt notation
    stress = jnp.array([sigma[0, 0], sigma[1, 1], sigma[0, 1]])

    # --- Stiffness Matrix Construction ---
    mu_tangent = material.mu_ - material.lambda_ * lnJ

    term1 = material.lambda_ + 2.0 * mu_tangent

    stiffness = jnp.array([
        [term1, material.lambda_, 0.0],
        [material.lambda_, term1, 0.0],
        [0.0, 0.0, mu_tangent]
    ])

    stiffness /= jac

    return stress, stiffness


jit_vmap_neo_hooke_update = jit(vmap(neo_hooke_update,
                                     in_axes=(0, None),
                                     out_axes=(0, 0)))


def neo_hooke_update_stress(F, material):
    """
    JAX implementation of Neo-Hookean material model.
    Made specific for 2D problems

    Args:
        F: deformation gradient tensor
        material: Material properties containing:
            lambda_: First Lamé parameter
            mu_: Second Lamé parameter (shear modulus)

    Returns:
        sigma - Cauchy stress in Voigt notation
    """
    # Define problem dimensions
    I = jnp.eye(2)

    # Jacobian and its logarithm
    jac = jnp.linalg.det(F)
    lnJ = jnp.log(jac)

    # Left Cauchy-Green tensor B = F·F^T
    B = F @ F.T

    # Kirchhoff stress tensor
    tau = material.lambda_ * lnJ * I + material.mu_ * (B - I)

    # Cauchy stress
    sigma = tau / jac

    return sigma


jit_vmap_neo_hooke_train_update = jit(vmap(neo_hooke_update_stress,
                                     in_axes=(0, None),
                                     out_axes=(0)))


def neo_hooke_update_stress_variable_mu(F, mu, lambda_):
    """
    JAX implementation of Neo-Hookean material model with variable mu per sample.
    Made specific for 2D problems

    Args:
        F: deformation gradient tensor [2x2]
        mu: shear modulus (scalar)
        lambda_: First Lamé parameter (scalar)

    Returns:
        sigma - Cauchy stress tensor [2x2]
    """
    # Define problem dimensions
    I = jnp.eye(2)

    # Jacobian and its logarithm
    jac = jnp.linalg.det(F)
    lnJ = jnp.log(jac)

    # Left Cauchy-Green tensor B = F·F^T
    B = F @ F.T

    # Kirchhoff stress tensor
    tau = lambda_ * lnJ * I + mu * (B - I)

    # Cauchy stress
    sigma = tau / jac

    return sigma


jit_vmap_neo_hooke_variable_mu = jit(vmap(neo_hooke_update_stress_variable_mu,
                                           in_axes=(0, 0, 0),
                                           out_axes=(0)))


# --- Example Usage ---
if __name__ == '__main__':
    ################# Validation of cauchy stress model #################
    # Test case comparison with expected Jive results
    jit_neo_hooke_update = jax.jit(neo_hooke_update)

    # Material properties
    mu = 961.538
    lambda_ = 1442.307

    material_props = create_material(lambda_, mu)

    # Deformation gradient
    F = jnp.array([[.9, 0.5],
                 [-0.1, 1.2]])

    # Compute stress and stiffness
    stress, stiffness = neo_hooke_update(F, material_props)

    print("Cauchy Stress (Voigt notation):", stress)
    print("Tangent Stiffness Matrix:", stiffness)

    assert jnp.allclose(stress, jnp.array([207.05095, 538.9093,  433.96854]), atol=1e-4), "Stress calculation is incorrect"
    # assert jnp.allclose(stiffness, jnp.array([[3012.8325, 1442.307,     0.    ],[1442.307,  3012.8325,    0.,    ], [   0.,        0.,      785.2627]]), atol=1e-3), "Tangent calculation is incorrect"
    # print(f"Stress and stiffness calculations are correct for tested F.")

    jitted_stress, jitted_stiffness = jit_neo_hooke_update(F, material_props)
    assert jnp.allclose(jitted_stress, jnp.array([207.05095, 538.9093, 433.96854]), atol=1e-4), "Jitted Stress calculation is incorrect"

    F = jnp.array([[0.9469, -0.06741],
                 [-0.03947, 1.012]])

    stress, stiffness = neo_hooke_update(F, material_props)

    assert jnp.allclose(stress, jnp.array([-167.99388, -42.682217, -106.24897]), atol=1e-4), "2nd Stress calculation is incorrect"

    ################# Batched time comparison #################
    print(f"Performing a timing test for JIT vmap Neo-Hooke update...")
    import numpy as np
    import time
    np.random.seed(0)

    # Automatic create dataset from material model
    def randBetaVec(alpha, beta, limits):
        vec = np.empty((2,2))
        for i in range(2):
            for j in range(2):
                idx = i * 2 + j
                vec[i,j] = np.random.beta(alpha, beta) * (limits[idx][1] - limits[idx][0]) + limits[idx][0]
        return vec

    min_df = -0.2
    max_df = 0.4
    steps_per_path = 20 #2000  # 20 #50  #20
    num_samples = 5

    base_path = np.linspace(0, 1, steps_per_path + 1)[1:]
    paths = np.broadcast_to(base_path, (num_samples, 2, 2, base_path.shape[0])).copy()
    paths[0, :,:, 0] = 0.0  # Manually overwrite first sample to contain the 0,0 point.
    stress = np.empty_like(paths)
    df_lims = np.array([[min_df, max_df], [min_df, max_df], [min_df, max_df], [min_df, max_df]])
    base_df_identity = np.broadcast_to(np.array([[1.0, 0.0], [0.0, 1.0]]).reshape(2, 2, 1), paths.shape)

    for i in range(num_samples):
        rand_path = randBetaVec(0.7, 0.7, df_lims)
        # print(f"Random path {i}: {rand_path}")
        rand_path = np.broadcast_to(rand_path.reshape(2, 2, 1), paths[0].shape)
        paths[i] = paths[i] * rand_path
    paths = paths + base_df_identity
    print(f"Paths created")

    # transform paths to be #samples, #steps, 2, 2
    paths = paths.transpose(0, 3, 1, 2)  # Now shape is (num_samples, steps_per_path, 2, 2)
    paths_combined = paths.reshape(-1, 2, 2)  # Now shape is (num_samples * steps_per_path, 2, 2)

    start_time  = time.time()
    out_stress = jit_vmap_neo_hooke_update(paths_combined, material_props)
    end_time = time.time()
    print(f"Time taken for first JIT vmap: {end_time - start_time:.4f} seconds")
    start_time  = time.time()
    for i in range(1000):
        out_stress = jit_vmap_neo_hooke_update(paths_combined, material_props)
    end_time = time.time()
    print(f"Time taken for following JIT vmaps on average: {(end_time - start_time)/1000:.6f} seconds")

    # transform back
    # out_stress = out_stress[0].reshape(num_samples, steps_per_path, 3)  # Now shape is (num_samples, steps_per_path, 3)
    # print(f"out_stress[0]: {out_stress[0]}")

    ################# PK1 stress test #################
    print(f"\n\nPerforming PK1 stress test with random deformation gradients...")
    from dolfinx import mesh, fem
    from mpi4py import MPI


    # Material properties
    mu = 961.538
    lambda_ = 1442.307

    # Create a single-cell mesh for UFL expression evaluation
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 1, 1, mesh.CellType.quadrilateral)
    V_tensor = fem.functionspace(domain, ("DG", 0, (2, 2)))

    # Generate random valid deformation gradients
    np.random.seed(42)
    num_samples = 10

    # Generate random F matrices with positive determinant
    F_samples = []
    for i in range(num_samples):
        # Generate random deformation gradient around identity
        # F = I + small perturbation to ensure positive determinant
        perturbation = np.random.uniform(-0.3, 0.3, (2, 2))
        F_rand = np.eye(2) + perturbation

        # Ensure positive determinant (required for valid deformation)
        if np.linalg.det(F_rand) > 0.1:
            F_samples.append(F_rand)

    # print(f"Generated {len(F_samples)} valid deformation gradients")

    # Compute PK1 stress and stiffness for each F using UFL
    PK1_results_ufl = np.zeros((len(F_samples), 2, 2))
    PK1_results_jax = np.zeros((len(F_samples), 2, 2))

    for idx, F_np in enumerate(F_samples):
        # print(f"F_{idx}:\n{F_np}")
        ##### UFL
        # Create function to hold F
        F_func = fem.Function(V_tensor)
        # Assign same F values to all cells
        F_flat = F_np.flatten()
        num_cells = V_tensor.dofmap.index_map.size_local
        F_func.x.array[:] = np.tile(F_flat, num_cells)

        # Compute PK1 stress using UFL
        PK1_ufl = neo_hooke_pk1_ufl(F_func, mu, lambda_)

        # Project or evaluate the result
        PK1_V = fem.functionspace(domain, ("DG", 0, (2, 2)))
        PK1_result = fem.Function(PK1_V)

        # Create expression for projection
        expr = fem.Expression(PK1_ufl, PK1_V.element.interpolation_points)
        PK1_result.interpolate(expr)

        # Extract values from first cell
        PK1_values = PK1_result.x.array[:4].reshape(2, 2)
        PK1_results_ufl[idx] = PK1_values
        # print(f"ufl PK1_{idx}:\n{PK1_values}")
        ##### Jax
        PK1_results_jax[idx] = neo_hooke_pk1_jax_variable_params(F_np, mu, lambda_)
        # print(f"jax PK1_{idx}:\n{PK1_results_jax[idx]}")
    assert jnp.allclose(PK1_results_ufl, PK1_results_jax, atol=1e-4), "Jax and UFL values differ!"
    print(f"Jax and UFL values for PK1 stress match!")

    # Compute PK2 values
    PK2_results_jax = np.zeros((len(F_samples), 2, 2))
    for idx, F_np in enumerate(F_samples):
        E = 1/2 * (F_np.T @ F_np - np.eye(2))
        C = 2 * E + np.eye(2)
        print(f"F: {F_np},  E: {E},  C: {C}")
        # test if E = SPD
        is_pos_def = jnp.all(jnp.linalg.eigvals(E) > 0)
        print(f"E is positive definite: {is_pos_def}")
        is_pos_def = jnp.all(jnp.linalg.eigvals(C) > 0)
        print(f"C is positive definite: {is_pos_def}")

        PK2_results_jax[idx] = neo_hooke_pk2_jax_variable_params(C, mu, lambda_)
    print(f"PK1 results from Jax:\n{PK1_results_jax}")
    print(f"PK2 results from Jax:\n{PK2_results_jax}")




