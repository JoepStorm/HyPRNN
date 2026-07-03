"""Plain feed-forward neural-network baseline (no physics constraints).

Maps a deformation-gradient history to a stress sequence; used as a reference
against the physics-informed (Hy)PRNN. Use `create_nn_model` to build one.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from scripts_surrogates.data_utils import matrix_to_tensor


ACTIVATION_FUNCTIONS = {
    'relu': nn.relu,
    'tanh': nn.tanh,
    'sigmoid': nn.sigmoid,
}


class StandardNN(nn.Module):
    """
    Standard neural network replacing the PRNN architecture.

    The input U (stretch tensor) is assumed symmetric, and therefore only the upper matrix is used
    The output P is not generally symmetric, therefore all components are returned.
    """
    n_outputs: int  # o: Number of output components
    hidden_sizes: tuple = (32, 32, 32)  # Hidden layer sizes
    activation: str = 'relu'
    use_bias: bool = True
    use_output_bias: bool = True

    @nn.compact
    def __call__(self, x, material=None, micro_params=None, reshape=True):
        # material & reshape are not used, but kept for easy compatibility with PRNN models.

        if self.activation not in ACTIVATION_FUNCTIONS:
            raise ValueError(f"Unknown activation: {self.activation}")
        act_fn = ACTIVATION_FUNCTIONS[self.activation]

        x_tensor = matrix_to_tensor(x, symmetric=True)

        # Concatenate m parameter if provided
        if micro_params is not None:
            # m has shape [b, m], x_voigt has shape [b, seq_len, 4]. Create [b, seq_len, 4+m]
            micro_broadcasted = jnp.broadcast_to(micro_params[:, None, :], (micro_params.shape[0], x_tensor.shape[1], micro_params.shape[1]))
            x_combined = jnp.concatenate([x_tensor, micro_broadcasted], axis=-1)
        else:
            x_combined = x_tensor

        # Forward pass through hidden layers
        for i, size in enumerate(self.hidden_sizes):
            x_combined = nn.Dense(features=size, use_bias=self.use_bias, name=f"hidden_{i}")(x_combined)
            x_combined = act_fn(x_combined)

        # Output layer
        outputs = nn.Dense(features=self.n_outputs, use_bias=self.use_output_bias, name="output")(x_combined)

        return outputs


def create_nn_model(n_outputs=3, hidden_sizes=(32, 32, 32), random_key=jax.random.PRNGKey(0), activation='relu', num_mat_features=0, use_bias=True):
    """Create and initialize a standard neural network model."""

    # Create model
    model = StandardNN(
        n_outputs=n_outputs,
        hidden_sizes=hidden_sizes,
        activation=activation,
        use_bias=use_bias,
        use_output_bias=use_bias,   # choice: use bias for output if we use it for hidden layer
    )

    # Create dummy material (not used but kept for compatibility)
    material = {'dummy': True}

    print(f'New NN: Material features: {num_mat_features} - Hidden layers {hidden_sizes} - Output size {n_outputs}')

    # Initialize model parameters
    params = model.init(
        random_key,
        jnp.zeros((1, 50, 2, 2)),  # Dummy input
        material,  # Material params (unused)
        jnp.zeros((1, num_mat_features))  # m parameter
    )

    return model, params, material
