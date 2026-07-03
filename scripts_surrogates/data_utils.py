#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility classes for data handling when training surrogate models.
This class takes in RVE-simulated data, and converts it to clean inputs and targets for the PRNN.
Rotation matrices are pre-computed in this class. 
"""

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

        return
