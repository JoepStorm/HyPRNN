import jax
import jax.numpy as jnp
import flax.linen as nn
import scripts_materials.neohooke as neohooke
from scripts_surrogates.data_utils import matrix_to_tensor
from material_params import WOOD_MU, WOOD_LAMBDA


def _get_activation(name):
    """Map string name to Flax activation function."""
    return {'sigmoid': nn.sigmoid, 'softplus': nn.softplus,
            'relu': nn.relu, 'tanh': nn.tanh}[name]


def _max_eigenvalue_2x2_spd(W):
    """λ_max of a 2x2 symmetric positive definite matrix (closed-form)."""
    a, b, c = W[0, 0], W[0, 1], W[1, 1]
    return 0.5 * (a + c + jnp.sqrt((a - c) ** 2 + 4 * b ** 2))


def _cholesky_weight_matrix(params, w_max):
    """Build SPD weight matrix from 3 Cholesky params, bounded by w_max."""
    L = jnp.array([[jax.nn.softplus(params[0]), 0.0],
                   [params[1], jax.nn.softplus(params[2])]])
    W = L @ L.T
    return W * jnp.minimum(1.0, w_max / _max_eigenvalue_2x2_spd(W))


def _llt_matrix(params):
    """Build C_micro = L @ L^T from 3 params with softplus diagonal (zero gives identity offset)."""
    l11, l22, l21 = params
    offset = - jnp.log(2.0) + 1.0
    L = jnp.array([[nn.softplus(l11) + offset, 0.0],
                   [l21, nn.softplus(l22) + offset]])
    return L @ L.T


# ---------------------------------------------------------------------------
# Fixed (non-hyper) encoder/decoder — used when shared_micro_features is empty
# ---------------------------------------------------------------------------

class FixedLinearEncoder(nn.Module):
    """Non-hyper encoder: direct trainable Cholesky params (like DeformationGradientLayer)."""
    n_matpts: int
    w_max: float = 1.25

    @nn.compact
    def __call__(self, x, embedding=None):
        chol = self.param('cholesky_params', nn.initializers.glorot_normal(), (self.n_matpts, 3))
        W = jax.vmap(_cholesky_weight_matrix, in_axes=(0, None))(chol, self.w_max)  # [m, 2, 2]
        E_micro = jnp.einsum('bij,mjk->bmik', x, W)
        return 2 * E_micro + jnp.eye(2)


class FixedNonLinearEncoder(nn.Module):
    """Non-hyper encoder: standard Flax MLP mapping strain to LLT params."""
    n_matpts: int
    hidden_dim: int = 8
    n_hidden_layers: int = 3
    activation: str = 'softplus'

    @nn.compact
    def __call__(self, x, embedding=None):
        act = _get_activation(self.activation)
        m, hd = self.n_matpts, self.hidden_dim
        x_flat = jnp.stack([x[:, 0, 0], x[:, 1, 1], x[:, 0, 1]], axis=1)  # [b, 3]

        layers = [nn.Dense(hd, name='enc_dense_0')]
        for i in range(self.n_hidden_layers - 1):
            layers.append(nn.Dense(hd, name=f'enc_dense_{i+1}'))
        layers.append(nn.Dense(m * 3, name='enc_dense_out'))

        def mlp(inp):
            h = inp
            for layer in layers[:-1]:
                h = act(layer(h))
            return layers[-1](h)

        llt_params = (mlp(x_flat) - mlp(jnp.zeros_like(x_flat))).reshape(-1, m, 3)
        return jax.vmap(jax.vmap(_llt_matrix))(llt_params)


class FixedSparseDecoder(nn.Module):
    """Non-hyper decoder: direct trainable weights (like SparseLayer)."""
    n_matpts: int
    n_outputs: int

    @nn.compact
    def __call__(self, micro_stresses_norm, embedding=None):
        raw_weights = self.param('raw_weights', nn.initializers.glorot_normal(), (self.n_matpts, self.n_outputs))
        weights = nn.softplus(raw_weights)
        return jnp.einsum('bmo,mo->bo', micro_stresses_norm, weights)


# ---------------------------------------------------------------------------
# Shared hypernetwork
# ---------------------------------------------------------------------------

class SharedHypernetwork(nn.Module):
    """Two-layer MLP mapping selected micro_params to a shared material embedding.

    Sizes scale with n_matpts (m): hidden = hidden_mult×m, output = output_mult×m.
    The embedding is consumed by both the encoder and decoder.
    """
    n_matpts: int
    micro_features: tuple   # indices into raw micro_params fed to this network
    hidden_mult: int = 2    # hidden width = hidden_mult × n_matpts
    output_mult: int = 4    # embedding dim = output_mult × n_matpts

    @nn.compact
    def __call__(self, micro_params):
        x = micro_params[:, self.micro_features]
        h = nn.sigmoid(nn.Dense(self.hidden_mult * self.n_matpts)(x))
        return nn.sigmoid(nn.Dense(self.output_mult * self.n_matpts)(h))   # [b, emb_dim]


class SharedHypernetworkFixed(nn.Module):
    """MLP hypernetwork with fixed hidden sizes (not scaled by n_matpts)."""
    n_matpts: int
    micro_features: tuple
    hidden_sizes: tuple        # e.g. (8, 8)
    activation: str = 'sigmoid'

    @nn.compact
    def __call__(self, micro_params):
        act = _get_activation(self.activation)
        x = micro_params[:, self.micro_features]
        for h_size in self.hidden_sizes:
            x = act(nn.Dense(h_size)(x))
        return x


# ---------------------------------------------------------------------------
# Hyper encoders  (receive embedding, do one more dense layer)
# ---------------------------------------------------------------------------

class SharedLinearEncoder(nn.Module):
    """Encoder equivalent to HyperDeformationGradientLayer_grad.

    embedding → Dense(3m) → Cholesky params → SPD weight matrices W
    → C_micro = 2·E·W + I
    """
    n_matpts: int
    w_max: float = 1.25

    @nn.compact
    def __call__(self, x, embedding):
        # embedding → Cholesky params [b, m, 3] (final hypernetwork layer)
        chol = nn.Dense(3 * self.n_matpts)(embedding).reshape(-1, self.n_matpts, 3)
        # Cholesky params → SPD weight matrices [b, m, 2, 2]
        W = jax.vmap(jax.vmap(_cholesky_weight_matrix, in_axes=(0, None)),in_axes=(0, None))(chol, self.w_max)
        E_micro = jnp.einsum('bij,bmjk->bmik', x, W)
        return 2 * E_micro + jnp.eye(2)    # C_micro [b, m, 2, 2]


class SharedNonLinearEncoder(nn.Module):
    """Encoder: embedding → Dense(all_mlp_params) → per-sample MLP → L@L^T = C_micro."""
    n_matpts: int
    hidden_dim: int = 8
    n_hidden_layers: int = 3
    activation: str = 'softplus'

    @nn.compact
    def __call__(self, x, embedding):
        m, hd, n_hl = self.n_matpts, self.hidden_dim, self.n_hidden_layers
        act = _get_activation(self.activation)
        x_flat = jnp.stack([x[:, 0, 0], x[:, 1, 1], x[:, 0, 1]], axis=1)  # [b, 3]

        # Compute total MLP params: input layer + hidden layers + output layer
        # Input: 3 → hd, Hidden: hd → hd, Output: hd → m*3
        sizes = [3*hd, hd]  # W1, b1
        for _ in range(n_hl - 1):
            sizes += [hd*hd, hd]
        sizes += [hd*m*3, m*3]  # W_out, b_out

        n_mlp_params = sum(sizes)
        hyper = nn.Dense(n_mlp_params)(embedding)   # [b, n_mlp_params]

        # Unpack weights and biases
        cuts = [sum(sizes[:i]) for i in range(1, len(sizes))]
        parts = jnp.split(hyper, cuts, axis=1)

        # Reshape weight matrices
        W_list, b_list = [], []
        idx = 0
        # Input layer
        W_list.append(parts[idx].reshape(-1, 3, hd)); b_list.append(parts[idx+1]); idx += 2
        # Hidden layers
        for _ in range(n_hl - 1):
            W_list.append(parts[idx].reshape(-1, hd, hd)); b_list.append(parts[idx+1]); idx += 2
        # Output layer
        W_list.append(parts[idx].reshape(-1, hd, m*3)); b_list.append(parts[idx+1])

        def mlp(inp):
            h = inp
            for W, b in zip(W_list[:-1], b_list[:-1]):
                h = act(jnp.einsum('bi,bij->bj', h, W) + b)
            return jnp.einsum('bi,bij->bj', h, W_list[-1]) + b_list[-1]

        # Bias shift: zero strain → zero LLT perturbation
        llt_params = (mlp(x_flat) - mlp(jnp.zeros_like(x_flat))).reshape(-1, m, 3)

        # Output C = L @ L^T
        return jax.vmap(jax.vmap(_llt_matrix))(llt_params)     # C_micro [b, m, 2, 2]


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class SharedSparseDecoder(nn.Module):
    """Decoder equivalent to HyperSparseLayer.

    embedding → Dense(m·o) → softplus weights → sparse einsum over material points.
    """
    n_matpts: int
    n_outputs: int

    @nn.compact
    def __call__(self, micro_stresses_norm, embedding):
        # Final "hypernetwork" layer to produce params.
        weights = nn.Dense(self.n_matpts * self.n_outputs)(embedding).reshape(-1, self.n_matpts, self.n_outputs)  # [b, m, o]

        # enforce positivity
        positive_weights = nn.softplus(weights)

        # The sparse decoding
        return jnp.einsum('bmo,bmo->bo', micro_stresses_norm, positive_weights)  # [b, o]


# ---------------------------------------------------------------------------
# Main PRNN class
# ---------------------------------------------------------------------------

class SharedHyperPRNN(nn.Module):
    """PRNN with a single shared hypernetwork feeding both encoder and decoder.
    """
    n_matpts: int              # int for single material, tuple for multiple materials
    n_outputs: int
    encoder_type: str              # 'Linear' or 'NonLinear'
    stress_normalizer: object
    material_micro_features: tuple # (mu_idx, lambda_idx) into raw micro_params
    shared_micro_features: tuple   # indices of raw micro_params fed to the shared hypernet
    shared_hidden_mult: int = 2    # shared hidden width = mult × m
    shared_output_mult: int = 4    # embedding dim       = mult × m
    hidden_dim: int = 8            # inner MLP hidden dim (NonLinear encoder only)
    w_max: float = 1.25            # λ_max bound for weight matrices (Linear encoder only)
    hyper_hidden_sizes: tuple = None   # if set, use SharedHypernetworkFixed
    hyper_activation: str = 'sigmoid'  # activation for the hypernetwork
    encoder_n_layers: int = 3          # hidden layers in NonLinear encoder MLP
    encoder_activation: str = 'softplus'  # activation in NonLinear encoder MLP
    use_multiple_materials: bool = False
    stress_scaling_index: int = None     # Index into micro_params for per-sample stress scaling (e.g. mu)

    def setup(self):
        if self.use_multiple_materials:
            total_matpts = int(sum(self.n_matpts))
        else:
            total_matpts = self.n_matpts

        # use a hypernetwork if features are specified
        use_hyper = len(self.shared_micro_features) > 0

        if use_hyper:
            if self.hyper_hidden_sizes is not None:
                self.hypernet = SharedHypernetworkFixed( n_matpts=total_matpts, micro_features=self.shared_micro_features, hidden_sizes=self.hyper_hidden_sizes, activation=self.hyper_activation, )
            else:
                self.hypernet = SharedHypernetwork(n_matpts=total_matpts, micro_features=self.shared_micro_features, hidden_mult=self.shared_hidden_mult, output_mult=self.shared_output_mult,)
        else:
            self.hypernet = None

        if self.encoder_type == 'Linear':
            if use_hyper:
                self.encoder = SharedLinearEncoder(n_matpts=total_matpts, w_max=self.w_max)
            else:
                self.encoder = FixedLinearEncoder(n_matpts=total_matpts, w_max=self.w_max)
        elif self.encoder_type == 'NonLinear':
            if use_hyper:
                self.encoder = SharedNonLinearEncoder(n_matpts=total_matpts, hidden_dim=self.hidden_dim, n_hidden_layers=self.encoder_n_layers, activation=self.encoder_activation)
            else:
                self.encoder = FixedNonLinearEncoder(n_matpts=total_matpts, hidden_dim=self.hidden_dim, n_hidden_layers=self.encoder_n_layers, activation=self.encoder_activation)
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        if use_hyper:
            self.decoder = SharedSparseDecoder(n_matpts=total_matpts, n_outputs=self.n_outputs)
        else:
            self.decoder = FixedSparseDecoder(n_matpts=total_matpts, n_outputs=self.n_outputs)

    def __call__(self, x, material, micro_params, reshape=True):
        """Forward pass. x: Green-Lagrange strain [batch, seq, 2, 2]."""
        s = int(x.shape[1])
        original_shape = x.shape

        if reshape:
            x = x.reshape(-1, 2, 2)    # [b, 2, 2]

        b = int(x.shape[0])     # batches (each timestep considered separate)

        # Expand micro_params from [batch, k] to [batch × seq_len, k]
        micro_params = jnp.broadcast_to(micro_params[:, None, :], (micro_params.shape[0], s, micro_params.shape[1])).reshape(b, micro_params.shape[-1])

        # --- Shared embedding ---
        embedding = self.hypernet(micro_params) if self.hypernet is not None else None

        # --- Encoder ---
        C_micro = self.encoder(x, embedding)        # [b, total_m, 2, 2]

        # --- Material evaluation ---
        if self.use_multiple_materials:
            C_neohooke, C_rest = jnp.split(C_micro, self.n_matpts[:1], axis=1)

            C_neohooke = C_neohooke.reshape(-1, 2, 2)
            C_rest = C_rest.reshape(-1, 2, 2)

            mu_per_pt = jnp.repeat(micro_params[:, self.material_micro_features[0]], self.n_matpts[0])
            lambda_per_pt = jnp.repeat(micro_params[:, self.material_micro_features[1]], self.n_matpts[0])
            micro_stresses_neo = neohooke.jit_vmap_neo_hooke_pk2_variable(C_neohooke, mu_per_pt, lambda_per_pt)

            n_wood_pts = C_rest.shape[0]
            micro_stresses_rest = neohooke.jit_vmap_neo_hooke_pk2_variable(
                C_rest, jnp.full((n_wood_pts,), WOOD_MU), jnp.full((n_wood_pts,), WOOD_LAMBDA))

            micro_stresses_neo = micro_stresses_neo.reshape(b, self.n_matpts[0], 2, 2)
            micro_stresses_rest = micro_stresses_rest.reshape(b, self.n_matpts[1], 2, 2)
            micro_stresses = jnp.concatenate((micro_stresses_neo, micro_stresses_rest), axis=1)
        else:
            C_flat = C_micro.reshape(-1, 2, 2)
            mu_per_pt = jnp.repeat(micro_params[:, self.material_micro_features[0]], self.n_matpts)
            lambda_per_pt = jnp.repeat(micro_params[:, self.material_micro_features[1]], self.n_matpts)
            micro_stresses = neohooke.jit_vmap_neo_hooke_pk2_variable(C_flat, mu_per_pt, lambda_per_pt).reshape(b, self.n_matpts, 2, 2)

        # --- Normalize (with optional per-sample stress scaling, e.g. dividing by mu) ---
        micro_stresses_tensor = matrix_to_tensor(micro_stresses, symmetric=True)  # [b, m, 3]
        scale = micro_params[:, self.stress_scaling_index] if self.stress_scaling_index is not None else None
        micro_stresses_norm = self.stress_normalizer.normalize(micro_stresses_tensor, scale=scale)

        # --- Decoder ---
        outputs = self.decoder(micro_stresses_norm, embedding)  # [b, o]

        if reshape and len(original_shape) == 4:
            outputs = outputs.reshape(original_shape[0], original_shape[1], self.n_outputs)

        return outputs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_shared_hyper_prnn_model(
    n_micro_raw,
    shared_micro_features,
    n_matpts=8,
    encoder_type='Linear',
    shared_hidden_mult=2,
    shared_output_mult=4,
    hidden_dim=8,
    w_max=1.25,
    n_outputs=3,
    random_key=jax.random.PRNGKey(0),
    stress_normalizer=None,
    mat_m_feats=None,
    hyper_hidden_sizes=None,
    hyper_activation='sigmoid',
    encoder_n_layers=3,
    encoder_activation='softplus',
    stress_scaling_index=None,
):
    """Create and initialize a SharedHyperPRNN."""
    use_multiple_materials = isinstance(n_matpts, list)
    total_matpts = int(sum(n_matpts)) if use_multiple_materials else n_matpts

    model = SharedHyperPRNN(
        n_matpts=tuple(n_matpts) if use_multiple_materials else n_matpts,
        n_outputs=n_outputs,
        encoder_type=encoder_type,
        stress_normalizer=stress_normalizer,
        material_micro_features=tuple(mat_m_feats),
        shared_micro_features=tuple(shared_micro_features),
        shared_hidden_mult=shared_hidden_mult,
        shared_output_mult=shared_output_mult,
        hidden_dim=hidden_dim,
        w_max=w_max,
        hyper_hidden_sizes=tuple(hyper_hidden_sizes) if hyper_hidden_sizes is not None else None,
        hyper_activation=hyper_activation,
        encoder_n_layers=encoder_n_layers,
        encoder_activation=encoder_activation,
        use_multiple_materials=use_multiple_materials,
        stress_scaling_index=stress_scaling_index,
    )

    material = neohooke.create_material(lambda_=150., mu_=100.)
    if use_multiple_materials:
        material = {'neohooke': material}
    micro_params_dummy = jnp.zeros((1, n_micro_raw))
    params = model.init(random_key, jnp.zeros((1, 1, 2, 2)), material, micro_params_dummy)

    if len(shared_micro_features) > 0:
        hyper_desc = (f'fixed{hyper_hidden_sizes}' if hyper_hidden_sizes is not None
                      else f'{shared_hidden_mult}m×{shared_output_mult}m')
        print(f'SharedHyperPRNN: encoder={encoder_type}  m={n_matpts}  '
              f'hyper={hyper_desc}({hyper_activation})'
              f'enc_layers={encoder_n_layers}({encoder_activation})  hidden_dim={hidden_dim}  '
              f'multiple_materials={use_multiple_materials}')
    else:
        print(f'SharedHyperPRNN (fixed, no hyper): encoder={encoder_type}  m={n_matpts}  '
              f'enc_layers={encoder_n_layers}({encoder_activation})  hidden_dim={hidden_dim}  '
              f'multiple_materials={use_multiple_materials}')

    return model, params, material
