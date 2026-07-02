#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes for training PRNNs using JAX"""

import jax.numpy as jnp
import numpy as np
import pandas
import jax
import json
jax.config.update('jax_platform_name', 'cpu')  # force jax to use cpu; which showed to be faster
# increase jax precision to double
# jax.config.update("jax_enable_x64", True)


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Convert numpy arrays to lists for JSON serialization
def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (jnp.ndarray, np.ndarray)):
        return obj.tolist()
    else:
        return obj

# Save as JSON
def save_settings(settings, path):
    with open(f"{path}.json", 'w') as f:
        json.dump(convert_to_serializable(settings), f, indent=2)


def load_settings(path):
    with open(f"{path}.json", 'r') as f:
        loaded_settings = json.load(f)
    return loaded_settings


def matrix_to_tensor(X, symmetric=False):
    if symmetric:
        rows, cols = [0, 1, 0], [0, 1, 1]  # notation xx, yy, xy=yx
    else:
        rows, cols = [0, 1, 0, 1], [0, 1, 1, 0]  # notation xx, yy, xy, yx
    return X[..., rows, cols]


def tensor_to_matrix(X, symmetric=False):
    if symmetric:
        matrix = jnp.zeros(X.shape[:-1] + (2, 2))
        matrix = matrix.at[..., 0, 0].set(X[..., 0])
        matrix = matrix.at[..., 1, 1].set(X[..., 1])
        matrix = matrix.at[..., 0, 1].set(X[..., 2])
        matrix = matrix.at[..., 1, 0].set(X[..., 2])
    else:
        matrix = jnp.zeros(X.shape[:-1] + (2, 2))
        matrix = matrix.at[..., 0, 0].set(X[..., 0])
        matrix = matrix.at[..., 1, 1].set(X[..., 1])
        matrix = matrix.at[..., 0, 1].set(X[..., 2])
        matrix = matrix.at[..., 1, 0].set(X[..., 3])
    return matrix


def create_Q_matrix(new_Q, theta):
    new_Q = new_Q.at[:, 0, 0].set(jnp.cos(theta))
    new_Q = new_Q.at[:, 0, 1].set(-jnp.sin(theta))
    new_Q = new_Q.at[:, 1, 0].set(jnp.sin(theta))
    new_Q = new_Q.at[:, 1, 1].set(jnp.cos(theta))
    return new_Q


def load_params(filename):
    """Load model params from NumPy file"""
    # Add .npy extension if not present
    if not filename.endswith('.npy'):
        filename = f"{filename}.npy"

    # Load the checkpoint
    checkpoint = np.load(filename, allow_pickle=True).item()

    # Convert NumPy arrays back to JAX arrays
    if checkpoint['best_params'] is not None:
        params = jax.tree_util.tree_map(lambda x: jnp.array(x), checkpoint['best_params'])
    else:
        params = jax.tree_util.tree_map(lambda x: jnp.array(x), checkpoint['params'])
    return params


class boundNormalizer:
    """Normalization for strain features.

    Scales strain data to a [-1,1] interval
    TODO: this is probably wrong, we need to take the absolute of the maximum of both!
    """

    def __init__(self, X):
        self.min = X.min(axis=0).values
        self.max = X.max(axis=0).values

    def normalize(self, x):
        return 2.0 * ((x - self.min) / (self.max-self.min)) - 1.0


class meanunitNormalizer:
    """Normalization of general data.
    scales data to zero mean and unit variance.
    """
    def __init__(self, X, normalize=True):
        if not normalize:
            self.mean = 0.0
            self.std = 1.0
            return
        else:
            self.mean = X.mean()
            self.std = X.std()

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


class meanUnitNormalizer_dim1:
    def __init__(self, X, normalize=True):
        # Handle both single boolean and list of booleans
        if isinstance(normalize, bool):
            # single boolean applies to all features
            if not normalize:
                self.mean = jnp.zeros(X.shape[1])
                self.std = jnp.ones(X.shape[1])
                return
            else:
                self.mean = X.mean(axis=0)
                self.std = X.std(axis=0)
                assert jnp.all(self.std > 1e-6), "Standard deviation is zero for some material features, normalization cannot be performed."
        else:
            # list of booleans, one per feature
            assert len(normalize) == X.shape[1], f"normalize list length ({len(normalize)}) must match number of features ({X.shape[1]})"
            self.mean = jnp.zeros(X.shape[1])
            self.std = jnp.ones(X.shape[1])

            for i, should_normalize in enumerate(normalize):
                if should_normalize:
                    self.mean = self.mean.at[i].set(X[:, i].mean())
                    self.std = self.std.at[i].set(X[:, i].std())
                    assert self.std[i] > 1e-6, f"Standard deviation is zero for feature {i}, normalization cannot be performed."

    def normalize(self, x):
        return (x - self.mean) / self.std
    def denormalize(self, x):
        return x * self.std + self.mean

class norm_LDstresses:
    """Normalize stresses by component-wise max absolute value.

    Optionally supports per-sample scaling (e.g. dividing by mu) before the
    global normalization.  When ``scale`` is given to ``__init__``, the
    ``norm_max`` is computed from the already-scaled data so that the
    normalised values are balanced across different scale magnitudes.
    """
    def __init__(self, X, normalize=True, scale=None):
        self.do_norm = normalize
        if not self.do_norm:
            self.norm_max = jnp.ones(X.shape[2])
        else:
            X_for_max = X / scale[:, None, None] if scale is not None else X
            self.norm_max = jnp.amax(jnp.abs(X_for_max), axis=(0, 1))

    def normalize(self, x, scale=None):
        """Normalize: optionally divide by per-sample scale, then by global max."""
        if scale is not None:
            x = x / scale[:, None, None]
        if self.do_norm:
            return x / self.norm_max
        return x

    def denormalize(self, x, scale=None):
        """Denormalize: multiply by global max, then optionally by per-sample scale."""
        x = x * self.norm_max
        if scale is not None:
            x = x * scale[:, None, None]
        return x

    def make_denormalizer(self, scale=None):
        """Return a denorm(x) callable for use with eval functions."""
        if scale is not None:
            return lambda x: self.denormalize(x, scale)
        return self.denormalize

class Normalize_set:
    """
    Normalize a train, validation and optionally test set based only on the training set samples.
    """
    def __init__(self, dataset, norm_input=True, norm_output=True):
        """
        :param dataset:
        """
        self.norm_input = norm_input
        self.norm_output = norm_output

        if norm_input:
            self.in_normalizer = boundNormalizer(dataset[:,0])
        if norm_output:
            self.out_normalizer = boundNormalizer(dataset[:,1])

    def normalize(self, dataset):
        new_dataset = dataset.copy()
        if self.norm_input:
            new_dataset[:,0] = self.in_normalizer.normalize(dataset[:,0])
        if self.norm_output:
            new_dataset[:,1] = self.out_normalizer.normalize(dataset[:,1])
        return new_dataset

def extract_normalizer_info(normalizer):
    """Extract normalization parameters from a normalizer object for saving."""
    import numpy as np

    if normalizer is None:
        return None

    normalizer_type = type(normalizer).__name__

    if normalizer_type == 'meanUnitNormalizer_dim1':
        return {
            'type': 'meanUnitNormalizer_dim1',
            'mean': np.array(normalizer.mean),
            'std': np.array(normalizer.std)
        }
    elif normalizer_type == 'norm_LDstresses':
        return {
            'type': 'norm_LDstresses',
            'do_norm': normalizer.do_norm,
            'norm_max': np.array(normalizer.norm_max)
        }
    else:
        print(f"Warning: Unknown normalizer type {normalizer_type}, cannot save")
        return None


def create_normalizer_from_info(info):
    """Reconstruct a normalizer object from saved parameters."""
    if info is None:
        return None

    normalizer_type = info['type']

    if normalizer_type == 'meanUnitNormalizer_dim1':
        normalizer = meanUnitNormalizer_dim1.__new__(meanUnitNormalizer_dim1)
        normalizer.mean = jnp.array(info['mean'])
        normalizer.std = jnp.array(info['std'])
        return normalizer
    elif normalizer_type == 'norm_LDstresses':
        normalizer = norm_LDstresses.__new__(norm_LDstresses)
        normalizer.do_norm = info['do_norm']
        normalizer.norm_max = jnp.array(info['norm_max'])
        return normalizer
    else:
        print(f"Warning: Unknown normalizer type {normalizer_type}, cannot reconstruct")
        return None


def polar_decomposition(F):
    """Perform polar decomposition on a deformation gradient tensor F."""

    # The identity case is handled separately to avoid NaNs in the AD jacrev/jacfwd SVD computation.
    # When F is the identity, return identity for both U and R. We still do F @ I to keep the dependency on F for AD.
    # This is dirty workaround, and ideally a better solution should be found.
    def identity_case(F):
        U = F @ jnp.eye(2)
        R = F @ jnp.eye(2)
        return U, R

    def svd_case(F):
        U_svd, S, Vh = jnp.linalg.svd(F, full_matrices=False)

        if F.ndim == 2:
            R = jnp.dot(U_svd, Vh)
            S_diag = jnp.diag(S)
            U = jnp.dot(Vh.T, jnp.dot(S_diag, Vh))
        else:
            R = jnp.einsum('...ij,...jk->...ik', U_svd, Vh)
            S_diag = jnp.zeros_like(F)
            for i in range(F.shape[-1]):
                S_diag = S_diag.at[..., i, i].set(S[..., i])
            U = jnp.einsum('...ji,...jk,...kl->...il', Vh, S_diag, Vh)

        return U, R

    # Use lax.cond for JIT-compatible branching
    # Having an identity matrix somehow causes NaNs in the SVD, so we handle that case separately
    is_identity = jnp.allclose(F, jnp.eye(2))
    return jax.lax.cond(is_identity, identity_case, svd_case, F)


def sqrt_matrix(C):
    # Take the square root of a 2x2 matrix. Works with batched inputs.

    # Extract components
    a = C[..., 0, 0]  # Shape (N,)
    b = C[..., 0, 1]  # Shape (N,)
    c = C[..., 1, 1]  # Shape (N,)

    # Calculate determinant and trace
    det = a * c - b**2
    tr = a + c

    # Calculate s and t
    s = jnp.sqrt(det)     # s = sqrt(det(C)) = sqrt(ac - b^2)
    t = jnp.sqrt(tr + 2 * s)  # t = sqrt(tr(C) + 2s) = sqrt(a + c + 2s)

    # Reshape s and t to (N, 1, 1) for broadcasting
    s = s[..., None, None]
    t = t[..., None, None]

    # Create a batch of identity matrices through broadcasting
    I = jnp.eye(2)

    # Apply the formula: U = (1/t) * (C + s*I)
    U = (C + s * I) / t

    return U


class LDDataset_PK1:
    """Dataset for handling Large deformation deformation gradient - PK1 stress paths in JAX.

    Dataasets should come prepared as numpy arrays saved in .npy files.
    """
    def __init__(self, base_filename, seq_length=None, num_samples=None, mat_file=None, mat_features=None, norm_stresses=False, norm_matparams=False, **kwargs):
        # example file base_filename = f"datasets/fungi/ellipsoid_fixed/mesh500_t50"
        # self.sig = jnp.load(f"{base_filename}_stresses.npy")  # stresses Shape: [meshes, timesteps, 3]
        # self.sig = jnp.load(f"{base_filename}_cauchy.npy")  # stresses Shape: [meshes, timesteps, 3]  # NOTE: new cauchy does contain 4 components. Old stresss 3.
        self.sig = jnp.load(f"{base_filename}_PK1.npy")  # stresses Shape: [meshes, timesteps, 2, 2]
        self.stiff = self.sig  # TMP overwrite - stiff is unnecessary # jnp.load(f"{base_filename}_stiffnesses.npy")  # Shape: [meshes, timesteps, 3]
        self.F = jnp.load(f"{base_filename}_F.npy")  # Shape: [meshes, timesteps, 2, 2]

        # if self.F or self.sig only has shape length 3, expand with one at start. Just adds a batch dimension of 1, so we can still process it.
        if len(self.F.shape) == 3:
            self.F = self.F[jnp.newaxis, ...]
            self.sig = self.sig[jnp.newaxis, ...]

        if seq_length is None:
            seq_length = self.sig.shape[1]
        self.seq_length = seq_length
        if num_samples is None:
            num_samples = self.sig.shape[0]
        self.num_samples = num_samples

        assert self.sig.shape[0] == self.num_samples, f"Prescribed number of samples {self.sig.shape[0]} does not indicated match number of samples {self.num_samples}"
        assert self.sig.shape[1] == self.seq_length, f"Dataset number of timesteps {self.sig.shape[1]} does not match indicated sequence length {self.seq_length}"

        # Create a Nx2x2 matrix Q representing the rotation for each sample
        # Initiate as identity matrix
        self.Q = jnp.zeros((num_samples, 2, 2)) + jnp.eye(2)

        self.mat_features = mat_features

        self.mat_params = False
        self.stress_scaling_feature = kwargs.get('stress_scaling_feature', None)
        self.stress_scale = jnp.ones(num_samples)
        if mat_file is not None:

            df = pandas.read_csv(mat_file, sep=' ')
            self.theta = df['angle'].values if 'angle' in df.columns else np.zeros(num_samples)

            self.Q = create_Q_matrix(self.Q, self.theta)

            # Per-sample stress scaling (e.g. divide by mu to equalize stress magnitudes across samples)
            if self.stress_scaling_feature is not None:
                self.stress_scale = jnp.array(df[self.stress_scaling_feature].values[:num_samples])
                print(f"Stress scaling by '{self.stress_scaling_feature}': range [{float(self.stress_scale.min()):.4f}, {float(self.stress_scale.max()):.4f}]")

            # Add features to include as inputs to the network
            if mat_features is not None:
                if len(mat_features) > 0:
                    print(f"Material features are specified: {mat_features}")

                    self.mat_params = True
                    # Create an array called self.M with the material features. Only append the columns corresponding to the names in mat_features.
                    self.M = jnp.zeros((num_samples, len(mat_features)))

                    for i, feature in enumerate(mat_features):
                        self.M = self.M.at[:,i].set(df[feature].values)

                    # check if norm_matparams is not a list and not a boolean
                    if not isinstance(norm_matparams, (bool, list, type(None))):
                        print("Using provided material parameter normalizer.")
                        self.M_normalizer = norm_matparams
                    else:
                        self.M_normalizer = meanUnitNormalizer_dim1(self.M, normalize=norm_matparams)
                    self.M_normalized = self.M_normalizer.normalize(self.M)
                else:
                    print(f"Material features not specified.")
            else:
                print(f"Material features not specified.")
        # use polar decomposition to get right stretch tensor U and rotation tensor R
        self.U, self.R = polar_decomposition(self.F)

        # Compute the equivalent stretch tensor rotated by the local rotation Q
        # broadcast Q to the sequence length to match the shape of U
        self.Q = jnp.repeat(self.Q[:, jnp.newaxis, :, :], self.seq_length, axis=1)  # Shape: [num_samples, seq_length, 2, 2]

        # compute the transpose of Q
        Q_T = jnp.moveaxis(self.Q, -2, -1)

        # Compute the equivalent stretch tensor: U_eq = sqrt(Q^T * U^2 * Q)
        U_eq = sqrt_matrix(Q_T @ self.U @ self.U @ self.Q)

        # The prnn always takes (U-I) as input. Pre-compute it for our data generation.
        I = jnp.eye(2)
        self.U_min_I =  U_eq - I

        # Convert sigma_F to sigma_U: sig_U = R^T * sig_F * R
        # first convert sigma to matrix instead of tensor notation [3] -> [2, 2]
        if len(self.sig.shape) == 4:
            sigma_matrix = self.sig
        else:
            if self.sig.shape[-1] == 3:  # sigma_yx assumed equal to sigma_xy
                sigma_matrix = tensor_to_matrix(self.sig, symmetric=True)
            else:
                sigma_matrix = tensor_to_matrix(self.sig, symmetric=False)

        R_transpose = jnp.moveaxis(self.R, -2, -1)  # Transpose the last two dimensions of R
        sig_U_unnormalized = jnp.einsum('...ij,...jk->...ik', R_transpose, jnp.einsum('...ij,...jk->...ik', sigma_matrix, self.R))

        self.sig_U_eq_unnormalized = Q_T @ sig_U_unnormalized @ self.Q

        # convert target to voigt notation, otherwise the MSE counts off-diagonals double.
        # self.sig_U_tensor_unnormalized = matrix_to_tensor(sig_U_unnormalized, symmetric=True)
        # non-symmetric: large deformations, not generally symmetric stress
        self.sig_U_tensor_unnormalized = matrix_to_tensor(sig_U_unnormalized, symmetric=False)

        nan_indices = jnp.argwhere(jnp.isnan(jnp.abs(self.sig_U_tensor_unnormalized)))
        if nan_indices.shape[0] > 0:
            print(f"################ Warning: {nan_indices.shape[0]} NaNs found in self.sig. Masking NaN's and continuing... ################")  # at indices: {nan_indices}
        # Setting sig to zero causes masking.
        self.sig_U_tensor_unnormalized = jnp.nan_to_num(self.sig_U_tensor_unnormalized, nan=0.0)
        self.sig = jnp.nan_to_num(self.sig, nan=0.0)

        # Stress normalization (with optional per-sample scaling, e.g. dividing by mu)
        scale = self.stress_scale if self.stress_scaling_feature is not None else None
        if not isinstance(norm_stresses, bool):
            print("Using provided stress normalizer.")
            self.stress_normalizer = norm_stresses
        else:
            print("Normalizing stress features.")
            self.stress_normalizer = norm_LDstresses(self.sig_U_tensor_unnormalized, normalize=norm_stresses, scale=scale)
        self.sig_U_tensor_normalized = self.stress_normalizer.normalize(self.sig_U_tensor_unnormalized, scale=scale)

        # Masking: non-converged points have 0.0 for all terms in sig_F.
        # When that is the case, that step has False, otherwise True

        # Shape is Nxtxo. Only consider it to be False when the sum of the components (o) is exactly zero.
        self.sig = matrix_to_tensor(self.sig, symmetric=False)
        # self.sig = self.sig.reshape(self.sig.shape[0], self.sig.shape[1], -1)
        self.mask = jnp.sum(jnp.abs(self.sig), axis=-1) > 1e-12  # Shape: (N, seq_length)
        # # expand from Nxt to Nxtxo:
        self.mask = jnp.repeat(self.mask[:, :, jnp.newaxis], self.sig.shape[-1], axis=-1)  # Shape: (N, seq_length, o)

        # Remove timesteps from training by setting a mask (inefficient since we still compute all, but easy)
        # self.mask = self.mask.at[:, :-20, :].set(False)

    def get_batch(self, idx):
        """Get a single batch at index idx"""
        if self.mat_params:
            m = self.M_normalized[idx]
            return self.F[idx], self.sig[idx], self.stiff[idx], m
        else:
            return self.F[idx], self.sig[idx], self.stiff[idx]

    def get_all_batches(self):
        """Get all batches as a dictionary of arrays"""
        data = {
            'F': self.F,            # original input
            'sig_F': self.sig,      # original output
            # 'sig_U': self.sig_U_normalized,        # sig_U: output rotated locally
            't': self.sig_U_tensor_normalized,      # sig_U: output rotated locally converted to [.., 3]
            'stiff': self.stiff,
            'x': self.U_min_I,            # PRNN input stretch, equivalent after rotating Q.
            'R': self.R,
            'sig_U_eq_unnorm': self.sig_U_eq_unnormalized,   # sig_U unnormalized
            'Q': self.Q,
            'mask': self.mask,
        }
        if self.mat_params:
            data['m'] = self.M_normalized
        if self.stress_scaling_feature is not None:
            data['stress_scale'] = self.stress_scale

        return data

    def get_subset(self, idxs):
        """Get all batches as a dictionary of arrays"""
        data = {
            'F': self.F[idxs],
            'sig_F': self.sig[idxs],
#             'sig_U': self.sig_U_normalized[idxs],        # sig_U: output rotated locally
            't': self.sig_U_tensor_normalized[idxs],  # sig_U: output rotated locally converted to [.., 3]
            'stiff': self.stiff[idxs],
            'x': self.U_min_I[idxs],
            'R': self.R[idxs],
            'sig_U_eq_unnorm': self.sig_U_eq_unnormalized[idxs],
            'Q': self.Q[idxs],
            'mask': self.mask[idxs],
        }
        if self.mat_params:
            data['m'] = self.M_normalized[idxs]
        if self.stress_scaling_feature is not None:
            data['stress_scale'] = self.stress_scale[idxs]

        return data

    def saveDataparams(self, filename):
        """
        Save all parameters required for later use.
        This includes normalization types and parameters
        """
        import json
        import numpy as np

        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (jnp.ndarray, np.ndarray)):
                return obj.tolist()
            else:
                return obj

        data = {
            'stress_normalizer': convert_to_serializable(extract_normalizer_info(self.stress_normalizer)),
            'mat_features': self.mat_features if hasattr(self, 'mat_features') else None,
            'stress_scaling_feature': self.stress_scaling_feature if hasattr(self, 'stress_scaling_feature') else None,
        }
        if self.mat_params:
            data['M_normalizer'] = convert_to_serializable(extract_normalizer_info(self.M_normalizer))

        # Save as JSON
        with open(f"{filename}.json", 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved dataset parameters to {filename}")
        return

    def loadDataparams(self, filename):
        """
        Load all parameters required for later use.
        This includes normalization types and parameters
        """
        import json

        with open(f"{filename}.json", 'r') as f:
            data = json.load(f)

        self.stress_normalizer = create_normalizer_from_info(data.get('stress_normalizer', None))
        self.stress_scaling_feature = data.get('stress_scaling_feature', None)
        self.mat_features = data.get('mat_features', None)

        if self.mat_features is not None:
            self.mat_params = True
            self.M_normalizer = create_normalizer_from_info(data.get('M_normalizer', None))
        else:
            self.mat_params = False
            self.M_normalizer = None

        # print(f"Loaded normalizers: stress_normalizer={self.stress_normalizer}, M_normalizer={self.M_normalizer}")
        # print(f"Material features: {self.mat_features}")
        return


class LDDataset:
    """Dataset for handling Large deformation  stress paths in JAX.
    From F - PK1 pairs, Data is processed to create E - PK2 training pairs, reducing the complexity for the network.

    Datasets should come prepared as numpy arrays saved in .npy files.
    """
    def __init__(self, base_filename, seq_length=None, num_samples=None, mat_file=None, mat_features=None, norm_stresses=False, norm_matparams=False, **kwargs):
        self.PK1 = jnp.load(f"{base_filename}_PK1.npy")  # stresses Shape: [meshes, timesteps, 2, 2]
        self.stiff = self.PK1  # TMP overwrite - stiff is unnecessary # jnp.load(f"{base_filename}_stiffnesses.npy")  # Shape: [meshes, timesteps, 3]
        self.F = jnp.load(f"{base_filename}_F.npy")  # Shape: [meshes, timesteps, 2, 2]

        # if self.F or self.PK1 only has shape length 3, expand with one at start. Just adds a batch dimension of 1, so we can still process it.
        if len(self.F.shape) == 3:
            self.F = self.F[jnp.newaxis, ...]
            self.PK1 = self.PK1[jnp.newaxis, ...]

        if seq_length is None:
            seq_length = self.PK1.shape[1]
        self.seq_length = seq_length
        if num_samples is None:
            num_samples = self.PK1.shape[0]
        self.num_samples = num_samples

        assert self.PK1.shape[0] == self.num_samples, f"Prescribed number of samples {self.PK1.shape[0]} does not indicated match number of samples {self.num_samples}"
        assert self.PK1.shape[1] == self.seq_length, f"Dataset number of timesteps {self.PK1.shape[1]} does not match indicated sequence length {self.seq_length}"

        # Create a Nx2x2 matrix Q representing the rotation for each sample
        # Initiate as identity matrix
        self.Q = jnp.zeros((num_samples, 2, 2)) + jnp.eye(2)

        self.mat_features = mat_features

        self.mat_params = False
        self.stress_scaling_feature = kwargs.get('stress_scaling_feature', None)
        self.stress_scale = jnp.ones(num_samples)
        if mat_file is not None:

            df = pandas.read_csv(mat_file, sep=' ')
            self.theta = df['angle'].values if 'angle' in df.columns else np.zeros(num_samples)

            self.Q = create_Q_matrix(self.Q, self.theta)

            # Per-sample stress scaling (e.g. divide by mu to equalize stress magnitudes across samples)
            if self.stress_scaling_feature is not None:
                self.stress_scale = jnp.array(df[self.stress_scaling_feature].values[:num_samples])
                print(f"Stress scaling by '{self.stress_scaling_feature}': range [{float(self.stress_scale.min()):.4f}, {float(self.stress_scale.max()):.4f}]")

            # Add features to include as inputs to the network
            if mat_features is not None:
                if len(mat_features) > 0:
                    print(f"Material features are specified: {mat_features}")

                    self.mat_params = True
                    # Create an array called self.M with the material features. Only append the columns corresponding to the names in mat_features.
                    self.M = jnp.zeros((num_samples, len(mat_features)))

                    for i, feature in enumerate(mat_features):
                        self.M = self.M.at[:,i].set(df[feature].values)

                    # check if norm_matparams is not a list and not a boolean
                    if not isinstance(norm_matparams, (bool, list, type(None))):
                        print("Using provided material parameter normalizer.")
                        self.M_normalizer = norm_matparams
                    else:
                        self.M_normalizer = meanUnitNormalizer_dim1(self.M, normalize=norm_matparams)
                    self.M_normalized = self.M_normalizer.normalize(self.M)
                else:
                    print(f"Material features not specified.")
            else:
                print(f"Material features not specified.")

        self.U, self.R = polar_decomposition(self.F)

        # Compute the equivalent stretch tensor rotated by the local rotation Q
        # broadcast Q to the sequence length to match the shape of F
        self.Q = jnp.repeat(self.Q[:, jnp.newaxis, :, :], self.seq_length, axis=1)  # Shape: [num_samples, seq_length, 2, 2]
        Q_T = jnp.moveaxis(self.Q, -2, -1)   # transpose

        # Compute the equivalent deformation gradient rotated by local rotation Q
        F_eq = Q_T @ self.F @ self.Q
        F_eq_T = jnp.moveaxis(F_eq, -2, -1)

        # Compute the right Cauchy-Green deformation tensor: C_eq = F_eq^T @ F_eq
        C_eq = F_eq_T @ F_eq

        # Compute Green-Lagrange strain: E = 0.5 * (C - I)
        I = jnp.eye(2)
        self.E = 0.5 * (C_eq - I)   # PRNN input

        assert jnp.allclose(self.E, jnp.moveaxis(self.E, -2, -1)), "Green-Lagrange strain tensor E is not symmetric."

        # Convert PK1 to PK2: PK2 = F^{-1} @ PK1
        if len(self.PK1.shape) == 4:
            PK1_matrix = self.PK1
        else:
            PK1_matrix = tensor_to_matrix(self.PK1, symmetric=True)

        self.PK2 = jnp.linalg.inv(self.F) @ PK1_matrix

        # Rotate PK2 to equivalent frame: PK2_eq = Q^T @ PK2 @ Q
        self.PK2_eq_unnormalized = Q_T @ self.PK2 @ self.Q

        self.PK2_tensor_unnormalized = matrix_to_tensor(self.PK2_eq_unnormalized, symmetric=True)

        nan_indices = jnp.argwhere(jnp.isnan(jnp.abs(self.PK2_tensor_unnormalized)))
        print(f"################ Warning: {nan_indices.shape[0]} NaNs found in stress data. Masking NaN's and continuing... ################")
        self.PK2_tensor_unnormalized = jnp.nan_to_num(self.PK2_tensor_unnormalized, nan=0.0)
        self.PK1 = jnp.nan_to_num(self.PK1, nan=0.0)

        # Stress normalization (with optional per-sample scaling, e.g. dividing by mu)
        scale = self.stress_scale if self.stress_scaling_feature is not None else None
        if not isinstance(norm_stresses, bool):
            print("Using provided stress normalizer.")
            self.stress_normalizer = norm_stresses
        else:
            print("Normalizing stress features.")
            self.stress_normalizer = norm_LDstresses(self.PK2_tensor_unnormalized, normalize=norm_stresses, scale=scale)
        self.PK2_tensor_normalized = self.stress_normalizer.normalize(self.PK2_tensor_unnormalized, scale=scale)

        # Masking: non-converged points have 0.0 for all components in stress.
        # When that is the case, that step has False, otherwise True
        self.mask = jnp.sum(jnp.abs(self.PK2_tensor_unnormalized), axis=-1) > 1e-12  # Shape: (N, seq_length)
        # expand from Nxt to Nxtxo:
        self.mask = jnp.repeat(self.mask[:, :, jnp.newaxis], self.PK2_tensor_unnormalized.shape[-1], axis=-1)  # Shape: (N, seq_length, o)

        # Remove timesteps from training by setting a mask (inefficient since we still compute all, but easy)
        # self.mask = self.mask.at[:, :-20, :].set(False)

    def get_all_batches(self):
        """Get all batches as a dictionary of arrays"""
        data = {
            'F': self.F,
            'PK1': self.PK1,
            't': self.PK2_tensor_normalized,
            'stiff': self.stiff,
            'x': self.E,
            'PK2_eq_unnorm': self.PK2_eq_unnormalized,   # sig_U unnormalized
            'Q': self.Q,
            'mask': self.mask,
        }
        if self.mat_params:
            data['m'] = self.M_normalized
        if self.stress_scaling_feature is not None:
            data['stress_scale'] = self.stress_scale

        return data

    def get_subset(self, idxs):
        """Get all batches as a dictionary of arrays"""
        data = {
            'F': self.F[idxs],
            'PK1': self.PK1[idxs],
            't': self.PK2_tensor_normalized[idxs],
            'stiff': self.stiff[idxs],
            'x': self.E[idxs],
            'PK2_eq_unnorm': self.PK2_eq_unnormalized[idxs],
            'Q': self.Q[idxs],
            'mask': self.mask[idxs],
        }
        if self.mat_params:
            data['m'] = self.M_normalized[idxs]
        if self.stress_scaling_feature is not None:
            data['stress_scale'] = self.stress_scale[idxs]

        return data

    def saveDataparams(self, filename):
        """
        Save all parameters required for later use.
        This includes normalization types and parameters
        """
        import json
        import numpy as np

        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (jnp.ndarray, np.ndarray)):
                return obj.tolist()
            else:
                return obj

        data = {
            'stress_normalizer': convert_to_serializable(extract_normalizer_info(self.stress_normalizer)),
            'mat_features': self.mat_features if hasattr(self, 'mat_features') else None,
            'stress_scaling_feature': self.stress_scaling_feature if hasattr(self, 'stress_scaling_feature') else None,
        }
        if self.mat_params:
            data['M_normalizer'] = convert_to_serializable(extract_normalizer_info(self.M_normalizer))

        # Save as JSON
        with open(f"{filename}.json", 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved dataset parameters to {filename}")
        return

    def loadDataparams(self, filename):
        """
        Load all parameters required for later use.
        This includes normalization types and parameters
        """
        import json

        with open(f"{filename}.json", 'r') as f:
            data = json.load(f)

        self.stress_normalizer = create_normalizer_from_info(data.get('stress_normalizer', None))
        self.stress_scaling_feature = data.get('stress_scaling_feature', None)
        self.mat_features = data.get('mat_features', None)

        if self.mat_features is not None:
            self.mat_params = True
            self.M_normalizer = create_normalizer_from_info(data.get('M_normalizer', None))
        else:
            self.mat_params = False
            self.M_normalizer = None

        # print(f"Loaded normalizers: stress_normalizer={self.stress_normalizer}, M_normalizer={self.M_normalizer}")
        # print(f"Material features: {self.mat_features}")
        return

if __name__ == "__main__":
    # Some test code to verify the LDDataset class works as expected, especially the polar decomposition and stress conversion.
    # Example base filename, adjust according to your dataset structure
    # base_filename = f"datasets/fungi/ellipsoid_fixed/mesh500_t50"
    # dataclass = LDDataset(base_filename, seq_length=50, num_samples=500, mat_file=None, mat_features=None, norm_stresses=True)
    base_filename = f"../data/vfrac_ratio_smallmu_biggerdomain/2runs_1/mixed_t100"
    dataclass = LDDataset(base_filename, seq_length=100, num_samples=686, mat_file=None, mat_features=None, norm_stresses=True)
    data = dataclass.get_all_batches()
    print(data['F'].shape, data['sig_F'].shape, data['stiff'].shape)
    print("full input F")
    print(data['F'][0,30])
    print(data['x'][0,30])
    print(data['R'][0,30])
    print("Recomputed F")
    print(data['R'][0,30] @ (data['x'][0,30] + jnp.eye(2)))  # should equal F

    print(f"output sig_F")
    print(data['sig_F'][0,30])
    print(f"output target")
    print(data['t'][0,30])

    # Recompute sig_F from target
    print("Recomputing sig_F from target")
    t = data['t'][0,30]

    # denormalize
    # TODO: this might be outdated - does not include the Q rotation
    sig_u_voigt_unnorm = dataclass.stress_normalizer.denormalize(t)
    # convert from voigt to matrix
    sig_u_unnorm = jnp.zeros((2,2))
    sig_u_unnorm = sig_u_unnorm.at[0,0].set(sig_u_voigt_unnorm[0])  # sigma_x
    sig_u_unnorm = sig_u_unnorm.at[1,1].set(sig_u_voigt_unnorm[1])  # sigma_y
    sig_u_unnorm = sig_u_unnorm.at[0,1].set(sig_u_voigt_unnorm[2])  # sigma_xy
    sig_u_unnorm = sig_u_unnorm.at[1,0].set(sig_u_voigt_unnorm[2])  # sigma_xy
    print("recomputed sig_U from target")
    print(sig_u_unnorm)
    print(f"original sig_U_eq")
    print(data['sig_U_eq_unnorm'][0,30])
    # now denormalize
    # now rotate back to sig_F by
    # applying R sig_U R^T
    tmp2 = data['R'][0,30] @ sig_u_unnorm @ data['R'][0,30].T
    # tmp2 = data['R'][0,30] @ data['sig_U_unnorm'][0,30] @ data['R'][0,30].T
    print("recomputed sig_F")
    print(tmp2)
    tmp2_voigt = jnp.zeros((3,))
    tmp2_voigt = tmp2_voigt.at[0].set(tmp2[0,0])  # sigma_xx
    tmp2_voigt = tmp2_voigt.at[1].set(tmp2[1,1])  # sigma_yy
    tmp2_voigt = tmp2_voigt.at[2].set(tmp2[0,1])  # sigma_xy (assumed equal to sigma_yx)
    print(tmp2_voigt)
    print(f"True original sig_F: {data['sig_F'][0,30]}")
    assert jnp.allclose(tmp2_voigt, data['sig_F'][0,30]), "Recomputed sig_F does not match original sig_F"

    # Test saving and loading of dataset parameters

    # Save dataset normalizers
    dataclass.saveDataparams('datasets/fungi/ellipsoid_fixed/normparams')

    # Load new dataset:
    newdataclass = LDDataset.__new__(LDDataset)
    newdataclass.loadDataparams('datasets/fungi/ellipsoid_fixed/normparams')

    # assert whether newdataclass has the same normalizer parameters
    assert newdataclass.stress_normalizer is not None, "Stress normalizer not loaded"
    assert jnp.allclose(newdataclass.stress_normalizer.norm_max, dataclass.stress_normalizer.norm_max), "Stress normalizer parameters do not match"
    assert newdataclass.mat_features == dataclass.mat_features, "Material features do not match"






