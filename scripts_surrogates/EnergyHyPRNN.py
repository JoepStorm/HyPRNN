"""Energy-based (Hy)PRNN surrogate architecture.

A Physically Recurrent Neural Network whose output is the exact gradient of a macroscopic
potential Psi(E) = sum_m alpha_m psi_m. Each fictitious Neo-Hookean material point receives the
Hencky strain through a learned strain-concentration tensor A_m (encoder), and its energy is
weighted by alpha_m >= 0 (decoder). A shared hypernetwork conditions A_m and alpha_m on
sample-level micro features (vfrac, aspect ratio, filler fraction, ...).
Use `create_energy_hyprnn_model` to build and initialize one.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import scripts_materials.neohooke as neohooke
from scripts_surrogates.data_utils import matrix_to_tensor
from material_params import WOOD_MU, WOOD_LAMBDA


_SQRT_EPS = 1e-12   # regularizer inside eigenvalue-gap square roots (smoothness at repeated eigenvalues)
_I2 = jnp.eye(2)
_SQRT2 = jnp.sqrt(2.0)


def _get_activation(name):
    """Map string name to Flax activation function."""
    return {'sigmoid': nn.sigmoid, 'softplus': nn.softplus,
            'relu': nn.relu, 'tanh': nn.tanh}[name]


def _sym(A):
    """Symmetric part of a (batched) matrix."""
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))


# ---------------------------------------------------------------------------
# Mandel notation for symmetric 2x2 tensors
# Mandel is used instead of voigt purely to make the analytical derivation easier.
# ---------------------------------------------------------------------------
#
# X = [[x11, x12], [x12, x22]]  <->  x = (x11, x22, sqrt(2) x12).
# The sqrt(2) makes X : Y = x . y, so a 3x3 matrix M acting on Mandel vectors
# represents a fourth-order tensor with minor symmetries, and its transpose
# represents the adjoint map. The identity map is M = I_3.

def _to_mandel(X):
    """[..., 2, 2] symmetric -> [..., 3]."""
    return jnp.stack([X[..., 0, 0], X[..., 1, 1], _SQRT2 * X[..., 0, 1]], axis=-1)


def _from_mandel(v):
    """[..., 3] -> [..., 2, 2] symmetric."""
    s = v[..., 2] / _SQRT2
    return jnp.stack([jnp.stack([v[..., 0], s], axis=-1),
                      jnp.stack([s, v[..., 1]], axis=-1)], axis=-2)


# ---------------------------------------------------------------------------
# Symmetric 2x2 matrix functions (log / exp) and their Fréchet derivatives
# ---------------------------------------------------------------------------
#
# Cayley–Hamilton form with a regularized eigenvalue gap, so that everything is
# smooth at repeated eigenvalues (which occur at E = 0). jnp.linalg.eigh
# is avoided on purpose: its gradient contains 1/(l1 - l2).
#
# For a symmetric A with eigenvalues l1 >= l2, gap delta = l1 - l2 and
# eigenprojectors P1, P2:
#     f(A)       = f(l2) I + d12 (A - l2 I),           d12 = (f(l1) - f(l2)) / delta
#     Df(A)[X]   = d12 X + (f'(l1) - d12) P1 X P1 + (f'(l2) - d12) P2 X P2   (Daleckii–Krein)
# The divided difference d12 is evaluated with log1p / expm1 so it stays accurate
# when delta is tiny.

def _eig2(A):
    a, b, d = A[0, 0], A[0, 1], A[1, 1]
    delta = jnp.sqrt((a - d) ** 2 + 4.0 * b ** 2 + _SQRT_EPS)   # > 0 always
    l1, l2 = 0.5 * (a + d + delta), 0.5 * (a + d - delta)
    P1 = (A - l2 * _I2) / delta
    return l1, l2, delta, P1, _I2 - P1


def _sym_log(C):
    """log C for SPD 2x2 C. Returns log C and the data for _frechet_log."""
    l1, l2, delta, P1, P2 = _eig2(C)
    d12 = jnp.log1p(delta / l2) / delta
    return jnp.log(l2) * _I2 + d12 * (C - l2 * _I2), (l1, l2, d12, P1, P2)


def _sym_exp(A):
    """exp A for symmetric 2x2 A. Returns exp A and the data for _frechet_exp."""
    l1, l2, delta, P1, P2 = _eig2(A)
    d12 = jnp.exp(l2) * jnp.expm1(delta) / delta
    return jnp.exp(l2) * _I2 + d12 * (A - l2 * _I2), (l1, l2, d12, P1, P2)


def _frechet_log(aux, X):
    """D log(C)[X] (self-adjoint on symmetric X)."""
    l1, l2, d12, P1, P2 = aux
    return d12 * X + (1.0 / l1 - d12) * P1 @ X @ P1 + (1.0 / l2 - d12) * P2 @ X @ P2


def _frechet_exp(aux, X):
    """D exp(A)[X] (self-adjoint on symmetric X)."""
    l1, l2, d12, P1, P2 = aux
    return d12 * X + (jnp.exp(l1) - d12) * P1 @ X @ P1 + (jnp.exp(l2) - d12) * P2 @ X @ P2


def _hencky_encode(M, E):
    """A_micro = A_m : log C  (= 2 A_m : H, H the Hencky strain) = log C_micro.

    M: [b, m, 3, 3] Mandel concentration matrices, E: [b, 2, 2]
    ->  A_micro [b, m, 2, 2] (the log of the micro right Cauchy-Green tensor)
        plus the Fréchet data of the log for the stress.

    C_micro = exp(A_micro) is SPD for every M and every admissible E, and
    equals I at E = 0, hence zero energy and zero stress in the reference
    configuration without any correction term. The point model is evaluated
    from A_micro directly (see _neo_hooke_log), because forming exp(A_micro)
    and taking its determinant loses the small eigenvalue to roundoff under
    strong compression with an amplifying encoder.
    """
    logC, aux_log = jax.vmap(_sym_log)(_I2 + 2.0 * E)                          # [b, 2, 2] = 2 H
    A = _from_mandel(jnp.einsum('bmij,bj->bmi', M, _to_mandel(logC)))         # [b, m, 2, 2]
    return A, aux_log


# ---------------------------------------------------------------------------
# Neo-Hookean point model (energy and analytic PK2 stress)
# ---------------------------------------------------------------------------

def _neo_hooke(C, mu, lambda_):
    """Compressible neo-Hookean energy and PK2 stress for a 2x2 right Cauchy-Green tensor.

    psi = mu/2 (tr C - 2) - mu ln J + lambda/2 (ln J)^2,   J = sqrt(det C)
    S   = 2 dpsi/dC = mu (I - C^-1) + lambda ln J C^-1
    """
    det = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
    C_inv = jnp.array([[C[1, 1], -C[0, 1]], [-C[1, 0], C[0, 0]]]) / det
    lnJ = 0.5 * jnp.log(det)
    psi = 0.5 * mu * (jnp.trace(C) - 2.0) - mu * lnJ + 0.5 * lambda_ * lnJ ** 2
    S = mu * (jnp.eye(2) - C_inv) + lambda_ * lnJ * C_inv
    return psi, S


def _neo_hooke_log(A, mu, lambda_):
    """Neo-Hookean energy and PK2 stress from A = log C (symmetric 2x2).

    Same model as _neo_hooke, but with ln J = tr(A)/2 and C^-1 = exp(-A), which
    are exact and roundoff-safe for widely separated eigenvalues. Also returns
    the Fréchet data of exp(A), needed to pull the stress back through the encoder.
    """
    expA, aux = _sym_exp(A)
    expmA, _ = _sym_exp(-A)
    lnJ = 0.5 * jnp.trace(A)
    psi = 0.5 * mu * (jnp.trace(expA) - 2.0) - mu * lnJ + 0.5 * lambda_ * lnJ ** 2
    S = mu * (_I2 - expmA) + lambda_ * lnJ * expmA
    return psi, S, aux


# ---------------------------------------------------------------------------
# Shared hypernetwork
# ---------------------------------------------------------------------------

def _constant_embedding(module, batch, dim):
    """Learned constant embedding, used when no micro features feed the hypernetwork."""
    emb = module.param('const_embedding', nn.initializers.normal(1.0), (dim,))
    return jnp.broadcast_to(nn.sigmoid(emb)[None, :], (batch, dim))


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
        if len(self.micro_features) == 0:     # no hypernetwork: learned constant embedding
            return _constant_embedding(self, micro_params.shape[0], self.output_mult * self.n_matpts)
        x = micro_params[:, self.micro_features]
        h = nn.sigmoid(nn.Dense(self.hidden_mult * self.n_matpts)(x))
        return nn.sigmoid(nn.Dense(self.output_mult * self.n_matpts)(h))   # [b, emb_dim]


class SharedHypernetworkFixed(nn.Module):
    """MLP hypernetwork with explicitly given hidden sizes; last hidden layer is the embedding."""
    n_matpts: int
    micro_features: tuple
    hidden_sizes: tuple
    activation: str = 'sigmoid'

    @nn.compact
    def __call__(self, micro_params):
        if len(self.micro_features) == 0:     # no hypernetwork: learned constant embedding
            return _constant_embedding(self, micro_params.shape[0], self.hidden_sizes[-1])
        act = _get_activation(self.activation)
        h = micro_params[:, self.micro_features]
        for width in self.hidden_sizes:
            h = act(nn.Dense(width)(h))
        return h


# ---------------------------------------------------------------------------
# Encoder  (embedding → Mandel strain-concentration matrices A_m, [b, m, 3, 3])
# ---------------------------------------------------------------------------

class SharedConcentrationEncoder(nn.Module):
    """General concentration encoder: embedding → Dense(9m) → I_3 + (3x3) per point.

    9 parameters per point; any real 3x3 matrix is admissible in log strain.
    Initialized as a small random perturbation of the identity map, so the
    points start near the macroscopic strain but are not identical.
    """
    n_matpts: int
    init_scale: float = 0.1

    @nn.compact
    def __call__(self, embedding):
        dense = nn.Dense(9 * self.n_matpts,
                         kernel_init=nn.initializers.variance_scaling(
                             self.init_scale, 'fan_in', 'truncated_normal'),
                         bias_init=nn.initializers.zeros)
        return dense(embedding).reshape(-1, self.n_matpts, 3, 3) + jnp.eye(3)   # [b, m, 3, 3]


# ---------------------------------------------------------------------------
# Decoder  (embedding → non-negative energy weights)
# ---------------------------------------------------------------------------

DECODER_ACTIVATIONS = {'softplus': nn.softplus, 'abs': jnp.abs}


def _decoder_activation(name):
    """Map a decoder_activation name to the function enforcing positive weights."""
    try:
        return DECODER_ACTIVATIONS[name]
    except KeyError:
        raise ValueError(f"Unknown decoder_activation: {name}; expected one of {sorted(DECODER_ACTIVATIONS)}")


class SharedEnergyDecoder(nn.Module):
    """Decoder: embedding → Dense(m) → softplus/abs → non-negative energy weights.

    The stress decoder is the adjoint of the encoder and carries no free
    parameters of its own.
    """
    n_matpts: int
    decoder_activation: str = 'softplus'

    @nn.compact
    def __call__(self, embedding):
        return _decoder_activation(self.decoder_activation)(nn.Dense(self.n_matpts)(embedding))   # [b, m]


class SharedEnergyDecoderNormalized(nn.Module):
    """Decoder: embedding → Dense(m) → softplus/abs → divide by sum, so positive energy weights and sum = 1."""
    n_matpts: int
    decoder_activation: str = 'softplus'

    @nn.compact
    def __call__(self, embedding):
        alpha = _decoder_activation(self.decoder_activation)(nn.Dense(self.n_matpts)(embedding))   # [b, m]
        return alpha / jnp.sum(alpha, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Main PRNN class
# ---------------------------------------------------------------------------

class EnergyHyPRNN(nn.Module):
    """Energy-based PRNN with a single shared hypernetwork feeding both encoder and decoder.

    The network defines a macroscopic potential
        Psi(E) = sum_m alpha_m psi_m(C_micro_m),   C_micro_m = exp(2 A_m : H),
    with H = 1/2 log(I + 2E) the Hencky strain, A_m a strain-concentration
    tensor (Mandel 3x3) from the encoder, alpha_m >= 0 from the decoder
    (optionally normalized to sum to 1 via normalized_decoder), and psi_m the
    neo-Hookean potential of the fictitious material points.

    The PK2 stress is the exact gradient of Psi, evaluated analytically via
        S(E) = dPsi/dE = Dlog(C)[ sum_m alpha_m A_m^T : Dexp(A_micro_m)[S_m] ],
    with S_m = 2 dpsi_m/dC_micro_m and Dlog, Dexp the (self-adjoint) Fréchet
    derivatives of the matrix log and exp. Hence the model is hyperelastic,
    has a symmetric consistent tangent, is admissible for every macroscopic
    strain with det F > 0, and is stress free and energy free at E = 0 for
    every parameter value, all by construction.
    """
    n_matpts: int              # int for single material, tuple for multiple materials
    n_outputs: int
    stress_normalizer: object
    material_micro_features: tuple # (mu_idx, lambda_idx) into raw micro_params
    shared_micro_features: tuple   # indices of raw micro_params fed to the shared hypernet
    shared_hidden_mult: int = 2    # shared hidden width = mult × m
    shared_output_mult: int = 4    # embedding dim       = mult × m
    hyper_hidden_sizes: tuple = None   # if set, use SharedHypernetworkFixed
    hyper_activation: str = 'sigmoid'  # activation for the hypernetwork
    use_multiple_materials: bool = False
    stress_scaling_index: int = None     # Index into micro_params for per-sample stress scaling (e.g. mu)
    normalized_decoder: bool = False     # if True, energy weights sum to 1
    decoder_activation: str = 'softplus'   # 'softplus' or 'abs' to enforce energy weights >= 0

    def setup(self):
        if self.use_multiple_materials:
            total_matpts = int(sum(self.n_matpts))
        else:
            total_matpts = self.n_matpts

        if self.n_outputs != 3:
            raise ValueError('The energy formulation outputs a symmetric 2x2 PK2 '
                             f'stress in Voigt form; n_outputs must be 3, got {self.n_outputs}')

        if self.hyper_hidden_sizes is not None:
            self.hypernet = SharedHypernetworkFixed(
                n_matpts=total_matpts, micro_features=self.shared_micro_features,
                hidden_sizes=self.hyper_hidden_sizes, activation=self.hyper_activation)
        else:
            self.hypernet = SharedHypernetwork(
                n_matpts=total_matpts, micro_features=self.shared_micro_features,
                hidden_mult=self.shared_hidden_mult, output_mult=self.shared_output_mult)

        self.total_matpts = total_matpts
        self.encoder = SharedConcentrationEncoder(n_matpts=total_matpts)

        if self.normalized_decoder:
            self.decoder = SharedEnergyDecoderNormalized(n_matpts=total_matpts, decoder_activation=self.decoder_activation)
        else:
            self.decoder = SharedEnergyDecoder(n_matpts=total_matpts, decoder_activation=self.decoder_activation)

    def _material_constants(self, micro_params):
        """Per material point (mu, lambda) arrays of shape [b, total_m]."""
        mu = micro_params[:, self.material_micro_features[0]]
        lambda_ = micro_params[:, self.material_micro_features[1]]

        if self.use_multiple_materials:
            n_neo, n_rest = self.n_matpts[0], self.n_matpts[1]
            mu_pp = jnp.concatenate(
                [jnp.repeat(mu[:, None], n_neo, axis=1),
                 jnp.full((mu.shape[0], n_rest), WOOD_MU)], axis=1)
            lambda_pp = jnp.concatenate(
                [jnp.repeat(lambda_[:, None], n_neo, axis=1),
                 jnp.full((lambda_.shape[0], n_rest), WOOD_LAMBDA)], axis=1)
        else:
            mu_pp = jnp.repeat(mu[:, None], self.total_matpts, axis=1)
            lambda_pp = jnp.repeat(lambda_[:, None], self.total_matpts, axis=1)

        return mu_pp, lambda_pp

    def _energy_and_stress(self, x, M, weights, mu_pp, lambda_pp):
        """Potential [b] and its analytic gradient, the PK2 stress [b, 2, 2]."""
        A_micro, aux_log = _hencky_encode(M, x)                                # [b, m, 2, 2]
        psi, S_micro, aux_exp = jax.vmap(jax.vmap(_neo_hooke_log))(A_micro, mu_pp, lambda_pp)  # [b, m], [b, m, 2, 2]
        energy = jnp.sum(weights * psi, axis=1)
        T = jax.vmap(jax.vmap(_frechet_exp))(aux_exp, S_micro)                 # Dexp(A_m)[S_m]
        G = _from_mandel(jnp.einsum('bm,bmji,bmj->bi', weights, M, _to_mandel(T)))   # sum_m a_m A_m^T : T_m
        stress = jax.vmap(_frechet_log)(aux_log, G)                            # Dlog(C)[G]
        return energy, stress

    def _operators(self, micro_params):
        """Mandel encoder matrices [n, m, 3, 3] and energy weights [n, m]."""
        embedding = self.hypernet(micro_params)
        return self.encoder(embedding), self.decoder(embedding)

    def factors(self, micro_params):
        """Encoder matrices A_m [n, m, 3, 3], weights alpha_m [n, m] and ||A_m||_2 [n, m].

        ||A_m||_2 is the largest amplification of the log strain by point m.
        For logging only.
        """
        M, alpha = self._operators(micro_params)
        return M, alpha, jnp.linalg.svd(M, compute_uv=False)[..., 0]

    def __call__(self, x, material, micro_params, reshape=True, return_energy=False):
        """Forward pass. x: Green-Lagrange strain [batch, seq, 2, 2]."""
        s = int(x.shape[1])
        original_shape = x.shape

        if reshape:
            x = x.reshape(-1, 2, 2)    # [b, 2, 2]

        x = _sym(x)
        b = int(x.shape[0])     # batches (each timestep considered separate)

        # Expand micro_params from [batch, k] to [batch × seq_len, k]
        micro_params = jnp.broadcast_to(
            micro_params[:, None, :],
            (micro_params.shape[0], s, micro_params.shape[1])).reshape(b, micro_params.shape[-1])

        # --- Shared embedding → encoder matrices and energy weights ---
        M, weights = self._operators(micro_params)               # [b, m, 3, 3], [b, m]

        # --- Material constants per fictitious point ---
        mu_pp, lambda_pp = self._material_constants(micro_params)

        # --- Potential and its analytic gradient ---
        energy, stress = self._energy_and_stress(x, M, weights, mu_pp, lambda_pp)

        # --- Normalize (with optional per-sample stress scaling, e.g. dividing by mu) ---
        stress_tensor = matrix_to_tensor(stress[:, None, :, :], symmetric=True)   # [b, 1, 3]
        scale = micro_params[:, self.stress_scaling_index] if self.stress_scaling_index is not None else None
        if self.stress_normalizer is not None:
            stress_tensor = self.stress_normalizer.normalize(stress_tensor, scale=scale)
        outputs = stress_tensor[:, 0, :]                          # [b, 3]

        if reshape and len(original_shape) == 4:
            outputs = outputs.reshape(original_shape[0], original_shape[1], self.n_outputs)
            energy = energy.reshape(original_shape[0], original_shape[1])

        if return_energy:
            return outputs, energy
        return outputs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_energy_hyprnn_model(
    n_micro_raw,
    shared_micro_features,
    n_matpts=8,
    shared_hidden_mult=2,
    shared_output_mult=4,
    n_outputs=3,
    random_key=jax.random.PRNGKey(0),
    stress_normalizer=None,
    mat_m_feats=None,
    hyper_hidden_sizes=None,
    hyper_activation='sigmoid',
    stress_scaling_index=None,
    normalized_decoder=False,
    decoder_activation='softplus',
):
    """Create and initialize an EnergyHyPRNN."""
    use_multiple_materials = isinstance(n_matpts, list)

    model = EnergyHyPRNN(
        n_matpts=tuple(n_matpts) if use_multiple_materials else n_matpts,
        n_outputs=n_outputs,
        stress_normalizer=stress_normalizer,
        material_micro_features=tuple(mat_m_feats),
        shared_micro_features=tuple(shared_micro_features),
        shared_hidden_mult=shared_hidden_mult,
        shared_output_mult=shared_output_mult,
        hyper_hidden_sizes=tuple(hyper_hidden_sizes) if hyper_hidden_sizes is not None else None,
        hyper_activation=hyper_activation,
        use_multiple_materials=use_multiple_materials,
        stress_scaling_index=stress_scaling_index,
        normalized_decoder=normalized_decoder,
        decoder_activation=decoder_activation,
    )

    material = neohooke.create_material(lambda_=150., mu_=100.)
    if use_multiple_materials:
        material = {'neohooke': material}
    micro_params_dummy = jnp.zeros((1, n_micro_raw))
    params = model.init(random_key, jnp.zeros((1, 1, 2, 2)), material, micro_params_dummy)

    hyper_desc = (f'fixed{hyper_hidden_sizes}' if hyper_hidden_sizes is not None
                  else f'{shared_hidden_mult}m×{shared_output_mult}m')
    common = (f'm={n_matpts}  normalized_decoder={normalized_decoder}  '
              f'decoder_activation={decoder_activation}  multiple_materials={use_multiple_materials}')
    if len(shared_micro_features) > 0:
        print(f'EnergyHyPRNN: hyper={hyper_desc}({hyper_activation})  {common}')
    else:
        print(f'EnergyHyPRNN (fixed, no hyper): {common}')

    return model, params, material



# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _random_strains(key, n):
    """Green-Lagrange strains from random deformation gradients with det F > 0."""
    F = jax.random.normal(key, (n, 2, 2)) * 0.4 + jnp.eye(2)
    flip = jnp.linalg.det(F) <= 0
    F = jnp.where(flip[:, None, None], F[:, :, ::-1], F)
    return 0.5 * (jnp.einsum('bji,bjk->bik', F, F) - jnp.eye(2))


def _voigt_to_matrix(v):
    return jnp.array([[v[0], v[2]], [v[2], v[1]]])


def _run_tests_generic():
    """Tests of the building blocks, independent of the encoder type."""
    # --- Mandel round trip and inner product ---
    X = jnp.array([[0.3, -0.2], [-0.2, 0.7]])
    Y = jnp.array([[-0.1, 0.4], [0.4, 0.2]])
    assert jnp.allclose(_from_mandel(_to_mandel(X)), X)
    assert jnp.allclose(jnp.sum(X * Y), _to_mandel(X) @ _to_mandel(Y))
    print('  Mandel round trip and inner product ok')

    # --- identity Mandel matrix is the identity map ---
    assert jnp.allclose(_from_mandel(jnp.eye(3) @ _to_mandel(X)), X)
    print('  identity concentration matrix ok')

    # --- a general concentration matrix can change the strain signature ---
    A = jnp.array([[1.0, -1.5, 0.0], [0.2, 0.9, 0.0], [0.0, 0.0, 1.1]])
    E_biax = jnp.diag(jnp.array([-0.15, -0.15]))[None]          # biaxial compression
    A_micro, _ = _hencky_encode(A[None, None], E_biax)
    ev = jnp.linalg.eigvalsh(A_micro[0, 0])
    assert ev[0] < 0 < ev[1], ev
    print('  general A_m maps biaxial compression to mixed tension/compression')

    # --- matrix log/exp round trip and Fréchet derivatives vs autodiff, incl. repeated eigenvalues ---
    for A in (jnp.array([[0.3, 0.1], [0.1, -0.2]]), 0.4 * jnp.eye(2)):
        expA, aux = _sym_exp(A)
        logA, _ = _sym_log(expA)
        assert jnp.allclose(logA, A, atol=1e-5)
        Xd = jnp.array([[0.7, -0.2], [-0.2, 0.4]])
        D_auto = jax.jvp(lambda Mx: _sym_exp(Mx)[0], (A,), (Xd,))[1]
        assert jnp.allclose(_frechet_exp(aux, Xd), D_auto, atol=1e-5)
    print('  matrix log/exp and Fréchet derivatives ok')

    # --- _neo_hooke_log agrees with _neo_hooke on a well-conditioned state ---
    A = jnp.array([[0.3, 0.1], [0.1, -0.2]])
    psi_a, S_a, _ = _neo_hooke_log(A, 100.0, 150.0)
    psi_b, S_b = _neo_hooke(_sym_exp(A)[0], 100.0, 150.0)
    assert jnp.allclose(psi_a, psi_b, rtol=1e-5) and jnp.allclose(S_a, S_b, rtol=1e-4, atol=1e-3)
    print('  neo-Hookean from log C agrees with neo-Hookean from C')


def _run_tests():
    key = jax.random.PRNGKey(0)
    n_micro_raw, mat_feats, shared_feats = 5, (0, 1), (0, 1, 2)
    b, s = 6, 4

    for shared in (shared_feats, ()):
        model, params, material = create_energy_hyprnn_model(
            n_micro_raw=n_micro_raw, shared_micro_features=shared,
            n_matpts=5, mat_m_feats=mat_feats, random_key=key)

        E = _random_strains(jax.random.PRNGKey(1), b * s).reshape(b, s, 2, 2)
        mp = jnp.abs(jax.random.normal(jax.random.PRNGKey(2), (b, n_micro_raw))) * 100 + 10

        out = model.apply(params, E, material, mp)
        assert out.shape == (b, s, 3), out.shape
        assert jnp.all(jnp.isfinite(out))

        # stress free and energy free reference configuration (exact, no correction)
        out0, en0 = model.apply(params, jnp.zeros((b, s, 2, 2)), material, mp, return_energy=True)
        assert jnp.allclose(out0, 0.0, atol=1e-6), jnp.abs(out0).max()
        assert jnp.allclose(en0, 0.0, atol=1e-6), jnp.abs(en0).max()

        print(f'  hyper={bool(shared)}: forward, stress-free ok')

    model, params, material = create_energy_hyprnn_model(
        n_micro_raw=n_micro_raw, shared_micro_features=shared_feats,
        n_matpts=5, mat_m_feats=mat_feats, random_key=key)
    mp = jnp.abs(jax.random.normal(jax.random.PRNGKey(2), (2, n_micro_raw))) * 100 + 10
    E = _random_strains(jax.random.PRNGKey(3), 2).reshape(2, 1, 2, 2)

    # --- factors: shapes and reported norms ---
    M, alpha, norms = model.apply(params, mp, method=EnergyHyPRNN.factors)
    assert M.shape == (2, 5, 3, 3) and alpha.shape == (2, 5) and norms.shape == (2, 5)
    print(f'  factors ok (max ||A_m||_2 = {float(norms.max()):.3f})')

    # --- admissibility with strongly amplifying encoders ---
    # float32 limit: eigenvalues of A_micro must stay within about +-85 (exp overflow).
    M_big = 4.0 * M
    mu_pp, lam_pp = jnp.full(M.shape[:2], 100.0), jnp.full(M.shape[:2], 150.0)
    F = jnp.diag(jnp.array([0.05, 1.0]))
    E_big = jnp.broadcast_to(0.5 * (F.T @ F - jnp.eye(2)), (M.shape[0], 2, 2))
    A_big, _ = _hencky_encode(M_big, E_big)
    psi_big, S_big, _ = jax.vmap(jax.vmap(_neo_hooke_log))(A_big, mu_pp, lam_pp)
    assert jnp.all(jnp.isfinite(psi_big)) and jnp.all(jnp.isfinite(S_big))
    print('  admissible with ||A_m||_2 up to %.1f at sigma_min = 0.05' % float(4 * norms.max()))

    # --- integrability: analytic stress equals the gradient of the reported energy ---
    def energy_of(strain):
        _, en = model.apply(params, strain, material, mp, return_energy=True)
        return jnp.sum(en)

    S_auto = _sym(jax.grad(energy_of)(E))[:, 0]
    S_model = jnp.stack([_voigt_to_matrix(v) for v in model.apply(params, E, material, mp)[:, 0]])
    assert jnp.allclose(S_auto, S_model, rtol=1e-4, atol=1e-4), jnp.abs(S_auto - S_model).max()
    print('  analytic stress equals dPsi/dE (integrability)')

    # --- symmetry of the consistent tangent ---
    def stress_matrix(strain):
        return _voigt_to_matrix(model.apply(params, strain[None, None], material, mp[:1])[0, 0])

    Ct = jax.jacrev(stress_matrix)(E[0, 0])                      # [2,2,2,2]
    assert jnp.allclose(Ct, jnp.transpose(Ct, (2, 3, 0, 1)), atol=1e-4)
    print('  consistent tangent is major-symmetric')

    # --- no work extracted over a closed strain cycle ---
    t = jnp.linspace(0.0, 2 * jnp.pi, 400)
    a, c = 0.06 * jnp.cos(t), 0.06 * jnp.sin(t)
    path = jnp.stack([jnp.array([[ai, ci], [ci, -ai]]) for ai, ci in zip(a, c)])
    S_path = model.apply(params, path[:, None], material, jnp.repeat(mp[:1], len(t), 0))[:, 0]
    S_mat = jnp.stack([_voigt_to_matrix(v) for v in S_path])
    dE = jnp.diff(path, axis=0)
    work = jnp.sum(jnp.einsum('bij,bij->b', 0.5 * (S_mat[:-1] + S_mat[1:]), dE))
    assert abs(float(work)) < 1e-5, work
    print(f'  closed-cycle work = {float(work):.2e}')

    # --- admissibility of the encoder output under severe compression ---
    E_sev = 0.5 * (F.T @ F - jnp.eye(2))[None, None]
    out = model.apply(params, E_sev, material, mp[:1])
    assert jnp.all(jnp.isfinite(out))
    print('  admissible at sigma_min = 0.05')

    # --- objectivity: superposed rotation leaves the PK2 stress unchanged ---
    theta = 0.7
    Q = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
    F0 = jnp.array([[1.2, 0.3], [-0.1, 0.9]])
    E_a = 0.5 * (F0.T @ F0 - jnp.eye(2))[None, None]
    F1 = Q @ F0
    E_b = 0.5 * (F1.T @ F1 - jnp.eye(2))[None, None]
    assert jnp.allclose(model.apply(params, E_a, material, mp[:1]),
                        model.apply(params, E_b, material, mp[:1]), atol=1e-5)
    print('  objectivity ok')

    # --- non-degeneracy: the initial tangent is anisotropic ---
    Ct0 = jax.jacrev(stress_matrix)(jnp.zeros((2, 2)))
    Ct0_rot = jnp.einsum('ia,jb,kc,ld,abcd->ijkl', Q, Q, Q, Q, Ct0)
    assert not jnp.allclose(Ct0, Ct0_rot, rtol=1e-3, atol=1e-3), 'initial tangent is isotropic'
    print('  initial tangent is anisotropic (encoder not degenerate)')

    # --- gradients w.r.t. parameters are finite, including at E = 0 (repeated eigenvalues) ---
    for strain in (E, jnp.zeros_like(E)):
        def loss(p):
            return jnp.sum(model.apply(p, strain, material, mp) ** 2)
        grads = jax.grad(loss)(params)
        assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads))
    print('  parameter gradients finite (also at E = 0)')


if __name__ == '__main__':
    _run_tests_generic()
    _run_tests()
    print('all tests passed')