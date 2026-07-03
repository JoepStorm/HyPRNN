import functools
import numpy as np
import jax
import jax.numpy as jnp
import optax
import copy
from flax.training import train_state


def create_learning_rate_fn(config):
    """
    Creates learning rate schedule with warmup followed by a cosine decay.
    requires in config: warmup_epochs, schedule_steps, steps_per_epoch, base_learning_rate
    Based on a fixed number of gradient steps. Depending on the number of batches, the # epochs thus varies.
    """
    # Warmup
    warmup_steps = config.warmup_epochs * config.steps_per_epoch
    warmup_fn = optax.linear_schedule( init_value=0.,
                                       end_value=config.base_learning_rate,
                                       transition_steps=warmup_steps)
    # Cosine decay
    cosine_steps = max(config.schedule_steps - warmup_steps, 1)
    cosine_fn = optax.cosine_decay_schedule( init_value=config.base_learning_rate,
                                             decay_steps=cosine_steps,
                                             alpha=config.min_lr_factor)    # alpha = limits minimum value. Minimum = alpha * init_value

    # Indefinite constant lr at same value of final cosine
    # This is useful when no maximum number of epochs is defined, but only an (early) stopping criteria
    min_lr = cosine_fn(cosine_steps - 1)
    constant_fn = optax.constant_schedule(value=min_lr)

    schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn, constant_fn],
      boundaries=[warmup_steps, warmup_steps + cosine_steps],)
    return schedule_fn


class Trainer:
    """Class for handling PRNN training tasks using JAX.

    Wraps a Flax model and performs training and evaluation tasks.
    Loss function and optimizer are configurable. Early stopping is
    implemented with adjustable patience.
    """

    def __init__(self, model, params, out_norm_factor, **kwargs):
        self._model = model
        self._epoch = 0
        self._best_num_nans = 0
        # self._criterion = kwargs.get('loss', self.mse_loss)

        # Get initial parameters
        self.rng = kwargs.get('random_key', jax.random.PRNGKey(0))
        self.rng, init_rng = jax.random.split(self.rng)

        # Get PRNN material properties
        self.material = kwargs.get('material', None)

        # Obtain normalization parameter
        self.out_norm_factor = out_norm_factor

        # # Print parameter count
        # total_params = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))
        # print('Total parameter count:', total_params)

        max_epochs = kwargs.get('max_epochs', 100000)
        self.train_losses = np.zeros(max_epochs)
        self.val_losses = np.zeros(max_epochs)
        self.test_losses = np.zeros(max_epochs)

        # Setup optimizer
        evaluate_only = kwargs.get('evaluate_only', False)
        if not evaluate_only:
            lr_config = kwargs.get('lr_config', None)
            self.lr_schedule = create_learning_rate_fn(lr_config)
            self._optimizer = kwargs.get('optimizer', optax.adam(self.lr_schedule))
        else:
            print(f"Default adam optimizer initialized - assumed not used!")
            self._optimizer = optax.adam(1e-2)  # some default settings.

        # Initialize train state
        self._state = train_state.TrainState.create(
            apply_fn=self._model.apply,
            params={'params': params['params']},
            tx=self._optimizer)


    def train_step(self, state, batch):
        """Single training step"""
        return self._train_step_jit(state, batch, self.material)

    @staticmethod
    def _signed_log(x):
        """Signed log transformation: sign(x) * log(1 + |x|). Compresses extreme values while preserving sign."""
        return jnp.sign(x) * jnp.log1p(jnp.abs(x))

    @staticmethod
    @jax.jit
    def _train_step_jit(state, batch, material):
        """JIT-compiled implementation of train step"""
        x = batch['x']
        t = batch['t']
        m = batch.get('m', None)
        mask = batch.get('mask')

        def loss_fn(params):
            y = state.apply_fn(params, x, material, micro_params=m)

            # loss = jnp.mean((y - t) ** 2)
            # MSE loss - with masking non-converged FEM steps and NaN PRNN predictions
            nan_mask = ~jnp.isnan(y)
            combined_mask = mask * nan_mask

            loss = jnp.sum((y - t) ** 2 * combined_mask) / jnp.sum(combined_mask)

            # # MSE loss in log-space to reduce emphasis on extreme values
            # y_log = Trainer._signed_log(y)
            # t_log = Trainer._signed_log(t)
            # loss = jnp.sum((y_log - t_log) ** 2 * combined_mask) / jnp.sum(combined_mask)

            # Debug info
            num_nans = jnp.sum(~nan_mask)
            # print(f"loss: {loss.item():.6f}, mask_sum: {mask_sum.item()}, num_nans: {num_nans.item()}/{y.size}")
            return loss, (y, num_nans)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (y, num_nans)), grads = grad_fn(state.params)

        # Update parameters only if there were no nans
        new_state = jax.lax.cond(
            num_nans > 0,
            lambda: state,  # Don't update parameters
            lambda: state.apply_gradients(grads=grads)  # Update parameters
        )

        return new_state, loss, num_nans


    @staticmethod
    @jax.jit
    def _eval_step_jit(state, batch, material):
        x = batch['x']
        t = batch['t']
        m = batch.get('m', None)
        mask = batch.get('mask')

        y = state.apply_fn(state.params, x, material, micro_params=m)

        # MSE loss - with masking non-converged FEM steps and NaN PRNN predictions
        nan_mask = ~jnp.isnan(y)
        combined_mask = mask * nan_mask
        num_nans = jnp.sum(~nan_mask)

        loss = jnp.sum((y - t) ** 2 * combined_mask) / jnp.sum(combined_mask)

        # # MSE loss in log-space to reduce emphasis on extreme values
        # y_log = Trainer._signed_log(y)
        # t_log = Trainer._signed_log(t)
        # loss = jnp.sum((y_log - t_log) ** 2 * combined_mask) / jnp.sum(combined_mask)

        return loss, num_nans

    @staticmethod
    # @jax.jit
    def eval_step_L1(state, batch, denormalizer, material=None):
        x = batch['x']
        t = batch['t']
        m = batch.get('m', None)
        mask = batch.get('mask')

        y = state.apply_fn(state.params, x, material, micro_params=m)

        nan_mask = ~jnp.isnan(y)
        combined_mask = mask * nan_mask

        # loss = jnp.sum(jnp.abs(y - t) * mask) / jnp.sum(mask)
        y_denorm = denormalizer(y)
        t_denorm = denormalizer(t)
        loss = jnp.sum(jnp.abs(y_denorm - t_denorm) * combined_mask) / jnp.sum(combined_mask)

        return loss

    @staticmethod
    # @jax.jit
    def eval_step_L2(state, batch, denormalizer, material=None):
        x = batch['x']
        t = batch['t']
        m = batch.get('m', None)
        mask = batch.get('mask')

        y = state.apply_fn(state.params, x, material, micro_params=m)

        nan_mask = ~jnp.isnan(y)
        combined_mask = mask * nan_mask

        y_denorm = denormalizer(y)
        t_denorm = denormalizer(t)
        loss = jnp.sum((y_denorm - t_denorm) ** 2 * combined_mask) / jnp.sum(combined_mask)

        return loss

    @staticmethod
    def eval_relative_norm(state, batch, denormalizer, material):
        """Relative L1 error based on the vector norm of the stress, avoiding component-wise blow-up."""
        x = batch['x']
        t = batch['t']
        m = batch.get('m', None)
        mask = batch.get('mask')

        y = state.apply_fn(state.params, x, material, micro_params=m)

        nan_mask = ~jnp.isnan(y)
        combined_mask = mask * nan_mask

        y_denorm = denormalizer(y)
        t_denorm = denormalizer(t)

        # Norm over stress components (last axis), giving (batch, time)
        error_norm = jnp.sqrt(jnp.sum((y_denorm - t_denorm) ** 2, axis=-1))
        target_norm = jnp.sqrt(jnp.sum(t_denorm ** 2, axis=-1))

        # Use mask reduced to (batch, time) — take any-component mask
        mask_reduced = jnp.min(combined_mask, axis=-1)

        # Normalize by max target norm per sample to avoid division by near-zero
        max_target_norm = jnp.max(target_norm * mask_reduced, axis=-1, keepdims=True)
        max_target_norm = jnp.maximum(max_target_norm, 1e-10)

        rel_error = error_norm / max_target_norm * mask_reduced
        rel_loss = jnp.sum(rel_error) / jnp.sum(mask_reduced)

        return rel_loss

    def train(self, training_data, validation_data, test_data=None, **kwargs):
        """Train the model with early stopping."""
        epochs = kwargs.get('max_epochs', 100)
        patience = kwargs.get('patience', 100)
        interval = kwargs.get('interval', 1)
        train_batch_size = kwargs.get('train_batch_size', 4)
        verbose = kwargs.get('verbose', True)

        train_indices = np.arange(training_data['x'].shape[0])
        num_batches = train_indices.shape[0] // train_batch_size
        assert train_indices.shape[0] % train_batch_size == 0, f"Number of training samples {train_indices.shape[0]} not divisible by batch size {train_batch_size}"

        stall_iters = 0
        self._best_val = float('inf')
        self._best_params = None

        for i in range(epochs):
            self._epoch = i
            shuffled_indices = np.random.permutation(train_indices)

            # Update
            running_loss = 0.0
            for batch_idx in range(num_batches):
                start_indices = batch_idx * train_batch_size
                end_indices = (batch_idx + 1) * train_batch_size
                batch_indices = shuffled_indices[start_indices:end_indices]

                batch = {k: v[batch_indices] for k, v in training_data.items()}

                self._state, loss, num_nans = self.train_step(self._state, batch)

                running_loss += loss * self.out_norm_factor**2  # Keep track of and print the 'unnormalized' loss for better comparison between runs.
                # As we take the MSE, we need to multiply with the factor**2. For MAE this should just be out_norm_factor.

            running_loss /= num_batches
            self.train_losses[i] = running_loss

            # Validation
            # if i < interval or i % interval == 0:
            if i % interval == 0:
                val_loss, num_nans = self._eval_step_jit(self._state, validation_data, self.material) * self.out_norm_factor**2
                self.val_losses[i] = val_loss

                if test_data is not None:
                    test_loss, _ = self._eval_step_jit(self._state, test_data, self.material) * self.out_norm_factor**2
                    self.test_losses[i] = test_loss

                if verbose:
                    test_str = f' test loss {self.test_losses[i]:.6f}' if test_data is not None else ''
                    print('Epoch', self._epoch, 'training loss', running_loss, 'validation loss', val_loss, test_str, 'lr', self.lr_schedule(self._state.step), 'Num nans', num_nans, 'stall iters:', stall_iters, '/', patience )

                if self._epoch == 0 or val_loss <= self._best_val:
                    self._best_val = val_loss
                    self._best_params = copy.deepcopy(self._state.params)
                    self._best_num_nans = num_nans
                    stall_iters = 0

                else:
                    if i <= interval:
                        stall_iters += 1
                    else:
                        stall_iters += interval

                if stall_iters >= patience:
                    if verbose:
                        print('Early stopping criterion reached.')
                    break

        if verbose:
            print('End of training.')

    def get_losses(self):
        """Return training, validation, and test losses."""
        return self.train_losses[:self._epoch], self.val_losses[:self._epoch], self.test_losses[:self._epoch]

    def save(self, filename):
        """Save model state to file using NumPy instead of TensorFlow"""
        import numpy as np
        from pathlib import Path

        # Convert JAX arrays to NumPy arrays
        params = jax.tree_util.tree_map(lambda x: np.array(x), self._state.params)
        best_params = jax.tree_util.tree_map(lambda x: np.array(x),
                                             self._best_params) if self._best_params is not None else None

        # Create parent directories if they don't exist
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        # Create dictionary with state information
        checkpoint = {
            'epoch': self._epoch,
            'best_val': float(self._best_val),
            'params': params,
            'best_params': best_params,
        }

        # Save using NumPy's save function
        np.save(filename, checkpoint, allow_pickle=True)
        print(f"Model with epoch {self._epoch}-patience and best_val: {float(self._best_val)}, and {self._best_num_nans} NaN: saved to {filename}.npy")

    def load(self, filename, set_best_params=True):
        """Load model state from NumPy file"""
        import numpy as np

        # Add .npy extension if not present
        if not filename.endswith('.npy'):
            filename = f"{filename}.npy"

        # Load the checkpoint
        checkpoint = np.load(filename, allow_pickle=True).item()

        # Update trainer state
        self._epoch = checkpoint['epoch']
        self._best_val = checkpoint['best_val']
        print(f"best_val: {self._best_val}")
        print(f"_epoch: {self._epoch}")
        params = jax.tree_util.tree_map(lambda x: jnp.array(x), checkpoint['params'])
        self._state = self._state.replace(params=params)

        if checkpoint['best_params'] is not None:
            # Convert NumPy arrays back to JAX arrays
            self._best_params = jax.tree_util.tree_map(lambda x: jnp.array(x), checkpoint['best_params'])
            if set_best_params:
                self._state = self._state.replace(params=self._best_params)
                return self._best_params
            else:
                return params
        else:
            if set_best_params:
                print("No best parameters found in checkpoint. Using current parameters.")
            self._best_params = None
            return params
