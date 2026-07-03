"""
Script to train a few models.

This serves a similar function to train_multiconfig_batch, but is easier to run a few models directly.
This is, for example, used to obtain the training time per model.
"""

import os
import csv
import time
import random
import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')

from trainer import Trainer
from data_utils import LDDataset, Config, save_settings
from StandardNN import create_nn_model
from HyPRNN import create_shared_hyper_prnn_model


RUNS_PER_MODEL = 2
# -- Common settings (shared across all configs) -----------------------------

COMMON = {
    'savefolder': '../trained_models/example/',
    'data_path': '../data/vary_all/mixed_t50_mergedv2',
    'matdata_path': '../data/vary_all/mixed_t50_mergedv2_matparam.data',

    'seq_length': 50,
    'train_batch_size': 2,
    'num_samples': 3072,
    'val_test_samples': 512,

    'norm_stresses': True,

    'lr_schedule_steps': 10000,
    'max_epochs': 5000,
    'warmup_epochs': 0,
    'base_lr': 1e-3,
    'min_lr_factor': 1,
    'patience': 50,
    'interval': 1,

    'feature_dim': 3,
    'verbose': False,
    'seed': 0,
}


# -- Per-model configurations ------------------------------------------------
# Each entry = one row group in the CSV. Tweak settings here to match the
# models referenced in validate_bending.py.

PRNN_MAT_PARAMS = {
    'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
    'norm_matparams': [False, False, True, True],
    'shared_micro_features': [2, 3],
    'mat_micro_features': [0, 1],
}

NN_MAT_PARAMS = {
    'mat_parameters': ['mu', 'vfrac', 'ratio'],
    'norm_matparams': [True, True, True],
}

MODELS_TO_TRAIN = {
    'prnn_lin_32': {
        'model_type': 'hyprnn',
        'train_samples': 32,
        'encoder_type': 'Linear',
        'mat_points': 6,
        'hyper_hidden_sizes': (8,),
        'hyper_activation': 'sigmoid',
        **PRNN_MAT_PARAMS,
    },
    'prnn_lin_512': {
        'model_type': 'hyprnn',
        'train_samples': 512,
        'encoder_type': 'Linear',
        'mat_points': 6,
        'hyper_hidden_sizes': (8,),
        'hyper_activation': 'sigmoid',
        **PRNN_MAT_PARAMS,
    },
    'prnn_nonlin_512': {
        'model_type': 'hyprnn',
        'train_samples': 512,
        'encoder_type': 'NonLinear',
        'mat_points': 12,
        'hyper_hidden_sizes': (8, 8, 8),
        'hyper_activation': 'sigmoid',
        'encoder_n_layers': 1,
        'encoder_activation': 'sigmoid',
        'hidden_dim': 8,
        **PRNN_MAT_PARAMS,
    },
    'nn_512': {
        'model_type': 'nn',
        'train_samples': 512,
        'nn_hidden_sizes': (64, 64, 64, 64),
        'nn_activation': 'sigmoid',
        'nn_bias': True,
        **NN_MAT_PARAMS,
    },
}

# -- Setup -------------------------------------------------------------------

os.makedirs(COMMON['savefolder'], exist_ok=True)

# Load the full dataset once. Since different configs may use different
# mat_parameters / norm_matparams, we (re)load the dataset per model below.

all_indices = np.arange(COMMON['num_samples'])
train_indices_full = all_indices[:int(-2 * COMMON['val_test_samples'])]
val_indices = all_indices[int(-2 * COMMON['val_test_samples']):int(-COMMON['val_test_samples'])]
test_indices = all_indices[int(-COMMON['val_test_samples']):]
print(f"Train pool/Val/Test sizes: "
      f"{train_indices_full.shape[0]}/{val_indices.shape[0]}/{test_indices.shape[0]}")


# -- Training loop -----------------------------------------------------------

csv_rows = []  # (model_name, run_idx, train_samples, walltime_s)
seed = COMMON['seed']

for model_name, model_cfg in MODELS_TO_TRAIN.items():
    print("\n" + "=" * 60)
    print(f"Training model: {model_name}")
    print("=" * 60)

    dataset = LDDataset(
        COMMON['data_path'],
        seq_length=COMMON['seq_length'],
        num_samples=COMMON['num_samples'],
        mat_file=COMMON['matdata_path'],
        mat_features=model_cfg['mat_parameters'],
        norm_stresses=COMMON['norm_stresses'],
        norm_matparams=model_cfg['norm_matparams'],
    )
    valset = dataset.get_subset(val_indices)

    train_samples = model_cfg['train_samples']

    for run_i in range(RUNS_PER_MODEL):
        seed += 1
        key = jax.random.PRNGKey(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Different random training subset per run.
        cur_indices = train_indices_full.copy()
        np.random.shuffle(cur_indices)
        cur_indices = cur_indices[:train_samples]
        trainset = dataset.get_subset(cur_indices)

        # Assemble per-run settings dict (what train() expects).
        settings = dict(COMMON)
        settings.update(model_cfg)
        settings['train_samples'] = train_samples
        settings['seed'] = seed
        # Batch size fallback for very small training sets.
        settings['train_batch_size'] = min(COMMON['train_batch_size'], train_samples)
        settings['savename'] = f"{COMMON['savefolder']}{model_name}_run{run_i}"

        # Build model.
        if model_cfg['model_type'] == 'hyprnn':
            model, params, material = create_shared_hyper_prnn_model(
                random_key=key,
                n_micro_raw=len(settings['mat_parameters']),
                shared_micro_features=settings['shared_micro_features'],
                n_matpts=settings['mat_points'],
                encoder_type=settings['encoder_type'],
                hidden_dim=settings.get('hidden_dim', 8),
                stress_normalizer=dataset.stress_normalizer,
                mat_m_feats=settings['mat_micro_features'],
                hyper_hidden_sizes=settings.get('hyper_hidden_sizes', None),
                hyper_activation=settings.get('hyper_activation', 'sigmoid'),
                encoder_n_layers=settings.get('encoder_n_layers', 3),
                encoder_activation=settings.get('encoder_activation', 'softplus'),
                stress_scaling_index=None
            )
        elif model_cfg['model_type'] == 'nn':
            model, params, material = create_nn_model(
                random_key=key,
                hidden_sizes=settings['nn_hidden_sizes'],
                num_mat_features=len(settings['mat_parameters']),
                activation=settings['nn_activation'],
                use_bias=settings['nn_bias'],
            )
        else:
            raise ValueError(f"Unknown model_type: {model_cfg['model_type']}")

        lr_config = Config(
            warmup_epochs=settings['warmup_epochs'],
            schedule_steps=settings['lr_schedule_steps'],
            steps_per_epoch=settings['train_samples'] / settings['train_batch_size'],
            base_learning_rate=settings['base_lr'],
            min_lr_factor=settings['min_lr_factor'],
        )
        train_handler = Trainer(model, params, material=material,
                                lr_config=lr_config, random_key=key, out_norm_factor=1)

        print(f"\n--- {model_name} run {run_i + 1}/{RUNS_PER_MODEL} "
              f"(train_samples={train_samples}, seed={seed}) ---")
        t0 = time.perf_counter()
        train_handler.train(trainset, valset, **settings)
        walltime = time.perf_counter() - t0
        print(f"Walltime: {walltime:.2f} s")

        train_handler.save(settings['savename'])
        dataset.saveDataparams(settings['savename'] + '_normparams')
        save_settings(settings, f"{settings['savename']}_settings")

        csv_rows.append((model_name, run_i, train_samples, walltime))

        # Append to CSV after every run so partial progress is persisted.
        csv_path = f"{COMMON['savefolder']}training_times.csv"
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['model', 'run', 'train_samples', 'walltime_s'])
            for row in csv_rows:
                w.writerow([row[0], row[1], row[2], f"{row[3]:.4f}"])
            # Averages per model.
            w.writerow([])
            w.writerow(['model', 'n_runs', 'mean_walltime_s', 'std_walltime_s'])
            by_model = {}
            for m, _, _, t in csv_rows:
                by_model.setdefault(m, []).append(t)
            for m, ts in by_model.items():
                arr = np.array(ts)
                w.writerow([m, len(arr), f"{arr.mean():.4f}", f"{arr.std(ddof=0):.4f}"])

print(f"\nSaved training times to {COMMON['savefolder']}training_times.csv")
