"""Train (and test) shared-PRNN surrogates on the deposition filler dataset,
varying only the number of material points.

Run once with MODE='train' to train all configs, then again with MODE='test'
to evaluate and produce a plot of average performance per mat_points.

Only a shared_prnn model with a Linear (hypernetwork) encoder is considered;
the sole variation is the number of material points (4..12). The material
parametrization uses the deposition filler settings (fil_frac).
"""
import os
import random
import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')

from trainer import Trainer
from data_utils import LDDataset, Config, save_settings
from LDprnn_shared_hyper import create_shared_hyper_prnn_model

import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use(['science', 'bright'])
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
rc('text', usetex=True)

# MODE = 'train'
MODE = 'test'

MAT_POINTS = [4, 6, 8, 10, 12]   # the only variation
RUNS_PER_MODEL = 3               # repeats per mat_points for averaging

mix_label = 'fil_frac'

COMMON = {
    'savefolder': '../trained_models/deposition/filler/vary_matpoints/',
    'data_path': '../data/deposition/filler/dataset_combi_v4v6/mixed_t50_merged',
    'matdata_path': '../data/deposition/filler/dataset_combi_v4v6/mixed_t50_merged_matparam.data',

    'seq_length': 50,
    'train_samples': 256,
    'train_batch_size': 2,
    'num_samples': 512,
    'val_test_samples': 128,

    'model_type': 'shared_prnn',
    'encoder_type': 'Linear',
    'decoder_type': 'HyperSparseLayer',

    'norm_stresses': True,
    'stress_scaling_feature': None,

    # deposition filler material parametrization
    'mat_parameters': ['mu', 'lambda', mix_label],
    'norm_matparams': [False, False, True],
    'shared_micro_features': [2],
    'mat_micro_features': [0, 1],
    'hyper_hidden_sizes': (8,),
    'hyper_activation': 'sigmoid',

    'lr_schedule_steps': 10000,
    'max_epochs': 3000,
    'warmup_epochs': 0,
    'base_lr': 1e-3,
    'min_lr_factor': 1,
    'patience': 50,
    'interval': 1,

    'feature_dim': 3,
    'verbose': False,
    'seed': 0,
}

os.makedirs(COMMON['savefolder'], exist_ok=True)


def build_model(n_matpts, key, dataset):
    return create_shared_hyper_prnn_model(
        random_key=key,
        n_micro_raw=len(COMMON['mat_parameters']),
        shared_micro_features=COMMON['shared_micro_features'],
        n_matpts=n_matpts,
        encoder_type=COMMON['encoder_type'],
        stress_normalizer=dataset.stress_normalizer,
        mat_m_feats=COMMON['mat_micro_features'],
        hyper_hidden_sizes=COMMON['hyper_hidden_sizes'],
        hyper_activation=COMMON['hyper_activation'],
        stress_scaling_index=None,
    )


def savename(n_matpts, run_i):
    return f"{COMMON['savefolder']}matpts{n_matpts}_run{run_i}"


# Load data once (mat_parameters are fixed across configs).
dataset = LDDataset(
    COMMON['data_path'],
    seq_length=COMMON['seq_length'],
    num_samples=COMMON['num_samples'],
    mat_file=COMMON['matdata_path'],
    mat_features=COMMON['mat_parameters'],
    norm_stresses=COMMON['norm_stresses'],
    norm_matparams=COMMON['norm_matparams'],
    stress_scaling_feature=COMMON['stress_scaling_feature'],
)

all_indices = np.arange(COMMON['num_samples'])
train_indices_full = all_indices[:int(-2 * COMMON['val_test_samples'])]
val_indices = all_indices[int(-2 * COMMON['val_test_samples']):int(-COMMON['val_test_samples'])]
test_indices = all_indices[int(-COMMON['val_test_samples']):]
print(f"Train pool/Val/Test sizes: "
      f"{train_indices_full.shape[0]}/{val_indices.shape[0]}/{test_indices.shape[0]}")

valset = dataset.get_subset(val_indices)
testset = dataset.get_subset(test_indices)


if MODE == 'train':
    seed = COMMON['seed']
    for n_matpts in MAT_POINTS:
        print("\n" + "=" * 60)
        print(f"Training mat_points = {n_matpts}")
        print("=" * 60)

        for run_i in range(RUNS_PER_MODEL):
            seed += 1
            key = jax.random.PRNGKey(seed)
            np.random.seed(seed)
            random.seed(seed)

            cur_indices = train_indices_full.copy()
            np.random.shuffle(cur_indices)
            cur_indices = cur_indices[:COMMON['train_samples']]
            trainset = dataset.get_subset(cur_indices)

            settings = dict(COMMON)
            settings['mat_points'] = n_matpts
            settings['seed'] = seed
            settings['savename'] = savename(n_matpts, run_i)

            model, params, material = build_model(n_matpts, key, dataset)
            lr_config = Config(
                warmup_epochs=settings['warmup_epochs'],
                schedule_steps=settings['lr_schedule_steps'],
                steps_per_epoch=settings['train_samples'] / settings['train_batch_size'],
                base_learning_rate=settings['base_lr'],
                min_lr_factor=settings['min_lr_factor'],
            )
            train_handler = Trainer(model, params, material=material,
                                    lr_config=lr_config, random_key=key, out_norm_factor=1)

            print(f"\n--- mat_points={n_matpts} run {run_i + 1}/{RUNS_PER_MODEL} (seed={seed}) ---")
            train_handler.train(trainset, valset, test_data=testset, **settings)

            train_handler.save(settings['savename'])
            dataset.saveDataparams(settings['savename'] + '_normparams')
            save_settings(settings, f"{settings['savename']}_settings")

    print(f"\nDone. Models saved under {COMMON['savefolder']}")


elif MODE == 'test':
    # Evaluate each trained model and collect L1/L2/relative losses.
    results = {n: {'L1': [], 'L2': [], 'rel': []} for n in MAT_POINTS}
    test_denorm = dataset.stress_normalizer.make_denormalizer(testset.get('stress_scale'))

    seed = COMMON['seed']
    for n_matpts in MAT_POINTS:
        for run_i in range(RUNS_PER_MODEL):
            seed += 1
            key = jax.random.PRNGKey(seed)

            path = savename(n_matpts, run_i)
            if not os.path.exists(path + '.npy'):
                print(f"Skipping missing model: {path}")
                continue

            model, params, material = build_model(n_matpts, key, dataset)
            lr_config = Config(
                warmup_epochs=COMMON['warmup_epochs'],
                schedule_steps=COMMON['lr_schedule_steps'],
                steps_per_epoch=COMMON['train_samples'] / COMMON['train_batch_size'],
                base_learning_rate=COMMON['base_lr'],
                min_lr_factor=COMMON['min_lr_factor'],
            )
            train_handler = Trainer(model, params, material=material,
                                    lr_config=lr_config, random_key=key, out_norm_factor=1)
            train_handler.load(path)

            l1 = train_handler.eval_step_L1(train_handler._state, testset, test_denorm, material)
            l2 = train_handler.eval_step_L2(train_handler._state, testset, test_denorm, material)
            rel = train_handler.eval_relative_norm(train_handler._state, testset, test_denorm, material)
            results[n_matpts]['L1'].append(float(l1))
            results[n_matpts]['L2'].append(float(l2))
            results[n_matpts]['rel'].append(float(rel))
            print(f"mat_points={n_matpts} run{run_i}: L1={l1:.4e}, L2={l2:.4e}, rel={rel:.4e}")

    # Aggregate and plot average performance per mat_points.
    matpts = [n for n in MAT_POINTS if results[n]['L1']]
    metrics = [('L1', 'Test L1 loss'), ('L2', 'Test L2 loss'), ('rel', 'Relative error')]

    # Save aggregated results.
    with open(f"{COMMON['savefolder']}summary.csv", 'w') as f:
        f.write('mat_points,n_runs,L1_mean,L1_std,L2_mean,L2_std,rel_mean,rel_std\n')
        for n in matpts:
            r = results[n]
            f.write(f"{n},{len(r['L1'])},"
                    f"{np.mean(r['L1']):.6e},{np.std(r['L1']):.6e},"
                    f"{np.mean(r['L2']):.6e},{np.std(r['L2']):.6e},"
                    f"{np.mean(r['rel']):.6e},{np.std(r['rel']):.6e}\n")
    print(f"Summary saved to {COMMON['savefolder']}summary.csv")

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 2.6))
    for ax, (key_m, ylabel) in zip(np.atleast_1d(axes), metrics):
        means = [np.mean(results[n][key_m]) for n in matpts]
        stds = [np.std(results[n][key_m]) for n in matpts]
        ax.errorbar(matpts, means, yerr=stds, marker='o', capsize=3, color=colours[0])
        ax.set_xlabel('Material points')
        ax.set_ylabel(ylabel)
        ax.set_xticks(matpts)
    plt.tight_layout()
    out_path = f"{COMMON['savefolder']}performance_per_matpoints.pdf"
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Performance plot saved to {out_path}")
