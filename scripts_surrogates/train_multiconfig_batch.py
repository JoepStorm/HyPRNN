import sys
import os
import random
import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')

from trainer import Trainer
from data_utils import LDDataset, Config, save_settings
from LDprnn import create_prnn_model
from LDnn import create_nn_model
from LDprnn_shared_hyper import create_shared_hyper_prnn_model

import matplotlib.pyplot as plt
from matplotlib import rc
try:
    plt.style.use(['science', 'bright'])
    rc('text', usetex=True)
except:
    print(f"not using custom colors")

mode = 'train'
# mode = 'test'  # if in train mode, just call the script without arguments.
# mode = 'plot'  # Just read saved results and plot

# output_fig_name = "prnn_learning_curves"
output_fig_name = "full"

savefolder = 'trained_models/train_vary_all_v2/'
# savefolder = 'trained_models/train_mu/'
# savefolder = 'trained_models/train_vfrac_ratio/'
# savefolder = 'trained_models/matparam_compare/'
runs_per_setting = 10
training_samples_array = [2, 4, 8, 12, 16, 24, 32, 64, 96, 128, 192, 256, 512]
# training_samples_array = [2, 4, 8, 12, 16, 24, 32, 64, 96] #, 128, 192, 256, 512]
# training_samples_array = [2, 16, 64]

# ============================================================================
# CONFIGURATION DEFINITIONS
# ============================================================================

# Base settings shared by all configurations
base_settings = {
    # 'data_path': f'../data/RVE/vfrac_ratio_smallmu_biggerdomain/10runs_0/mixed_t100',
    # 'matdata_path': f'../data/RVE/vfrac_ratio_smallmu_biggerdomain/10runs_0/mixed_t100_matparam.data',
    # 'data_path': f'../data/RVE/vfrac_ratio_smallmu_biggerdomain/10runs_literature_matprops/mixed_t100_merged',
    # 'matdata_path': f'../data/RVE/vfrac_ratio_smallmu_biggerdomain/10runs_literature_matprops/mixed_t100_merged_matparam.data',
    # 'seq_length': 100,

    # 'data_path': f'../data/RVE/vfrac_ratio_smallmu_biggerdomain/10runs_literature_matprops/mixed_t50_merged',
    # 'matdata_path': f'../data/RVE/vfrac_ratio_smallmu_biggerdomain/10runs_literature_matprops/mixed_t50_merged_matparam.data',
    # 'data_path': f'../data/RVE/vary_mu/vary_mu/mixed_t50_merged',
    # 'matdata_path': f'../data/RVE/vary_mu/vary_mu/mixed_t50_merged_matparam.data',
    # 'data_path': f'../data/RVE/vary_vfrac_ratio/vary_vfrac_ratio/mixed_t50_merged',
    # 'matdata_path': f'../data/RVE/vary_vfrac_ratio/vary_vfrac_ratio/mixed_t50_merged_matparam.data',
    # 'data_path': f'../data/RVE/vary_all/vary_all/mixed_t50_merged',
    # 'matdata_path': f'../data/RVE/vary_all/vary_all/mixed_t50_merged_matparam.data',
    'data_path': f'../data/RVE/vary_all/vary_all/mixed_t50_mergedv2',
    'matdata_path': f'../data/RVE/vary_all/vary_all/mixed_t50_mergedv2_matparam.data',
    'seq_length': 50,

    'train_batch_size': 2,
    # 'num_samples': 3430,
    # 'val_test_samples': 3430 * 0.2,
    'num_samples': 3072,
    'val_test_samples': 512,

    'norm_stresses': True,
    'stress_scaling_feature': None,  # 'mu'

    'lr_schedule_steps': 10000,
    'max_epochs': 5000,
    'warmup_epochs': 0,
    'base_lr': 1e-3, #2,
    'min_lr_factor': 1, #5e-2,
    'patience': 50,
    'interval': 1,

    'feature_dim': 3,
    'verbose': False,
    'seed': 0
}

# Define all configurations to run
# Each config should have: name, model_type, and relevant parameters
configurations = [
    # --- NN configurations ---
    {
        'name': 'nn_8_2',
        'model_type': 'nn',
        'nn_hidden_sizes': (8, 8),
        'nn_activation': 'sigmoid',
        'nn_bias': True,
        'mat_parameters': ['mu', 'vfrac', 'ratio'],
        'norm_matparams': [True, True, True],
    },
    {
        'name': 'nn_16_2',
        'model_type': 'nn',
        'nn_hidden_sizes': (16, 16),
        'nn_activation': 'sigmoid',
        'nn_bias': True,
        'mat_parameters': ['mu', 'vfrac', 'ratio'],
        'norm_matparams': [True, True, True],
    },
    {
        'name': 'nn_64_2',
        'model_type': 'nn',
        'nn_hidden_sizes': (64, 64),
        'nn_activation': 'sigmoid',
        'nn_bias': True,
        'mat_parameters': ['mu', 'vfrac', 'ratio'],
        'norm_matparams': [True, True, True],
    },
    {
        'name': 'nn_32_3',
        'model_type': 'nn',
        'nn_hidden_sizes': (32, 32, 32),
        'nn_activation': 'sigmoid',
        'nn_bias': True,
        'mat_parameters': ['mu', 'vfrac', 'ratio'],
        'norm_matparams': [True, True, True],
    },
    {
        'name': 'nn_64_3',
        'model_type': 'nn',
        'nn_hidden_sizes': (64, 64, 64),
        'nn_activation': 'sigmoid',
        'nn_bias': True,
        'mat_parameters': ['mu', 'vfrac', 'ratio'],
        'norm_matparams': [True, True, True],
    },
    {
        'name': 'nn_128_3',
        'model_type': 'nn',
        'nn_hidden_sizes': (128, 128, 128),
        'nn_activation': 'sigmoid',
        'nn_bias': True,
        'mat_parameters': ['mu', 'vfrac', 'ratio'],
        'norm_matparams': [True, True, True],
    },
    {
        'name': 'nn_64_4',
        'model_type': 'nn',
        'nn_hidden_sizes': (64, 64, 64, 64),
        'nn_activation': 'sigmoid',
        'nn_bias': True,
        'mat_parameters': ['mu', 'vfrac', 'ratio'],
        'norm_matparams': [True, True, True],
    },
    {
        'name': 'nn_tanh_64_3',
        'model_type': 'nn',
        'nn_hidden_sizes': (64, 64, 64),
        'nn_activation': 'tanh',
        'nn_bias': True,
        'mat_parameters': ['mu', 'vfrac', 'ratio'],
        'norm_matparams': [True, True, True],
    },
    {
        'name': 'nn_relu_64_3',
        'model_type': 'nn',
        'nn_hidden_sizes': (64, 64, 64),
        'nn_activation': 'relu',
        'nn_bias': True,
        'mat_parameters': ['mu', 'vfrac', 'ratio'],
        'norm_matparams': [True, True, True],
    },

    # --- Shared Hyper PRNN configurations ---
    # - Nonlinear -
    {
        'name': 'prnn_nonlin_3L8_6m',
        'model_type': 'shared_prnn',
        'encoder_type': 'NonLinear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 6,
        'hyper_hidden_sizes': (8, 8, 8),
        'hyper_activation': 'sigmoid',
        'encoder_n_layers': 3,
        'encoder_activation': 'softplus',
        'hidden_dim': 8,
    },
    # {
    #     'name': 'prnn_nonlin_3L8_6m3',
    #     'model_type': 'shared_prnn',
    #     'encoder_type': 'NonLinear',
    #     'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
    #     'norm_matparams': [False, False, True, True],
    #     'shared_micro_features': [2, 3],
    #     'mat_micro_features': [0, 1],
    #     'mat_points': [6, 3],
    #     'hyper_hidden_sizes': (8, 8, 8),
    #     'hyper_activation': 'sigmoid',
    #     'encoder_n_layers': 3,
    #     'encoder_activation': 'softplus',
    #     'hidden_dim': 8,
    # },
    {
        'name': 'prnn_nonlin_3L8_3m',
        'model_type': 'shared_prnn',
        'encoder_type': 'NonLinear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 3,
        'hyper_hidden_sizes': (8, 8, 8),
        'hyper_activation': 'sigmoid',
        'encoder_n_layers': 3,
        'encoder_activation': 'softplus',
        'hidden_dim': 8,
    },
    {
        'name': 'prnn_nonlin_3L8_2m',
        'model_type': 'shared_prnn',
        'encoder_type': 'NonLinear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 2,
        'hyper_hidden_sizes': (8, 8, 8),
        'hyper_activation': 'sigmoid',
        'encoder_n_layers': 3,
        'encoder_activation': 'softplus',
        'hidden_dim': 8,
    },
    {
        'name': 'prnn_nonlin_3L8_12m',
        'model_type': 'shared_prnn',
        'encoder_type': 'NonLinear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 12,
        'hyper_hidden_sizes': (8, 8, 8),
        'hyper_activation': 'sigmoid',
        'encoder_n_layers': 3,
        'encoder_activation': 'softplus',
        'hidden_dim': 8,
    },
    {
        'name': 'prnn_nonlin_3L8_24m',
        'model_type': 'shared_prnn',
        'encoder_type': 'NonLinear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 24,
        'hyper_hidden_sizes': (8, 8, 8),
        'hyper_activation': 'sigmoid',
        'encoder_n_layers': 3,
        'encoder_activation': 'softplus',
        'hidden_dim': 8,
    },
    {
        'name': 'prnn_nonlin_1HL8_3L_6m',
        'model_type': 'shared_prnn',
        'encoder_type': 'NonLinear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 6,
        'hyper_hidden_sizes': (8,),
        'hyper_activation': 'sigmoid',
        'encoder_n_layers': 3,
        'encoder_activation': 'softplus',
        'hidden_dim': 8,
    },
    {
        'name': 'prnn_nonlin_1HL8_1L_6m',
        'model_type': 'shared_prnn',
        'encoder_type': 'NonLinear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 6,
        'hyper_hidden_sizes': (8,),
        'hyper_activation': 'sigmoid',
        'encoder_n_layers': 1,
        'encoder_activation': 'softplus',
        'hidden_dim': 8,
    },
    {
        'name': 'prnn_nonlin_1L8_3m_sigmoid',
        'model_type': 'shared_prnn',
        'encoder_type': 'NonLinear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 3,
        'hyper_hidden_sizes': (8, 8, 8),
        'hyper_activation': 'sigmoid',
        'encoder_n_layers': 1,
        'encoder_activation': 'sigmoid',
        'hidden_dim': 8,
    },
    {
        'name': 'prnn_nonlin_1L8_6m_sigmoid',
        'model_type': 'shared_prnn',
        'encoder_type': 'NonLinear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 6,
        'hyper_hidden_sizes': (8, 8, 8),
        'hyper_activation': 'sigmoid',
        'encoder_n_layers': 1,
        'encoder_activation': 'sigmoid',
        'hidden_dim': 8,
    },
    {
        'name': 'prnn_nonlin_1HL8_1L_6m_sigmoid',
        'model_type': 'shared_prnn',
        'encoder_type': 'NonLinear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 6,
        'hyper_hidden_sizes': (8,),
        'hyper_activation': 'sigmoid',
        'encoder_n_layers': 1,
        'encoder_activation': 'sigmoid',
        'hidden_dim': 8,
    },
    {
        'name': 'prnn_nonlin_1L8_6m_sigmoid_scaleMu',
        'model_type': 'shared_prnn',
        'encoder_type': 'NonLinear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'stress_scaling_feature': 'mu',
        'mat_points': 6,
        'hyper_hidden_sizes': (8, 8, 8),
        'hyper_activation': 'sigmoid',
        'encoder_n_layers': 1,
        'encoder_activation': 'sigmoid',
        'hidden_dim': 8,
    },
    {
        'name': 'prnn_nonlin_1L8_12m_sigmoid',
        'model_type': 'shared_prnn',
        'encoder_type': 'NonLinear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 12,
        'hyper_hidden_sizes': (8, 8, 8),
        'hyper_activation': 'sigmoid',
        'encoder_n_layers': 1,
        'encoder_activation': 'sigmoid',
        'hidden_dim': 8,
    },
    {
        'name': 'prnn_nonlin_3L8_3m_sigmoid',
        'model_type': 'shared_prnn',
        'encoder_type': 'NonLinear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 3,
        'hyper_hidden_sizes': (8, 8, 8),
        'hyper_activation': 'sigmoid',
        'encoder_n_layers': 3,
        'encoder_activation': 'sigmoid',
        'hidden_dim': 8,
    },
    {
        'name': 'prnn_nonlin_3L8_6m_sigmoid',
        'model_type': 'shared_prnn',
        'encoder_type': 'NonLinear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 6,
        'hyper_hidden_sizes': (8, 8, 8),
        'hyper_activation': 'sigmoid',
        'encoder_n_layers': 3,
        'encoder_activation': 'sigmoid',
        'hidden_dim': 8,
    },
    {
        'name': 'prnn_nonlin_3L8_12m_sigmoid',
        'model_type': 'shared_prnn',
        'encoder_type': 'NonLinear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 12,
        'hyper_hidden_sizes': (8, 8, 8),
        'hyper_activation': 'sigmoid',
        'encoder_n_layers': 3,
        'encoder_activation': 'sigmoid',
        'hidden_dim': 8,
    },
    {
        'name': 'prnn_nonlin_3L8_6m_relu',
        'model_type': 'shared_prnn',
        'encoder_type': 'NonLinear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 6,
        'hyper_hidden_sizes': (8, 8, 8),
        'hyper_activation': 'sigmoid',
        'encoder_n_layers': 3,
        'encoder_activation': 'relu',
        'hidden_dim': 8,
    },
    {
        'name': 'prnn_nonlin_3L8relu_6m',
        'model_type': 'shared_prnn',
        'encoder_type': 'NonLinear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 6,
        'hyper_hidden_sizes': (8, 8, 8),
        'hyper_activation': 'relu',
        'encoder_n_layers': 3,
        'encoder_activation': 'softplus',
        'hidden_dim': 8,
    },
    # - Linear -
    # {
    #     'name': 'prnn_lin_3L_6m3',
    #     'model_type': 'shared_prnn',
    #     'encoder_type': 'Linear',
    #     'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
    #     'norm_matparams': [False, False, True, True],
    #     'shared_micro_features': [2, 3],
    #     'mat_micro_features': [0, 1],
    #     'mat_points': [6, 3],
    #     'hyper_hidden_sizes': (8, 8, 8),
    #     'hyper_activation': 'sigmoid',
    # },
    {
        'name': 'prnn_lin_3L_6m',
        'model_type': 'shared_prnn',
        'encoder_type': 'Linear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 6,
        'hyper_hidden_sizes': (8, 8, 8),
        'hyper_activation': 'sigmoid',
    },
    {
        'name': 'prnn_lin_3L_3m',
        'model_type': 'shared_prnn',
        'encoder_type': 'Linear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 3,
        'hyper_hidden_sizes': (8, 8, 8),
        'hyper_activation': 'sigmoid',
    },
    {
        'name': 'prnn_lin_3L_2m',
        'model_type': 'shared_prnn',
        'encoder_type': 'Linear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 2,
        'hyper_hidden_sizes': (8, 8, 8),
        'hyper_activation': 'sigmoid',
    },
    {
        'name': 'prnn_lin_3L_12m',
        'model_type': 'shared_prnn',
        'encoder_type': 'Linear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 12,
        'hyper_hidden_sizes': (8, 8, 8),
        'hyper_activation': 'sigmoid',
    },
    {
        'name': 'prnn_lin_1L_6m',
        'model_type': 'shared_prnn',
        'encoder_type': 'Linear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 6,
        'hyper_hidden_sizes': (8,),
        'hyper_activation': 'sigmoid',
    },
    {
        'name': 'prnn_lin_1L_3m',
        'model_type': 'shared_prnn',
        'encoder_type': 'Linear',
        'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
        'norm_matparams': [False, False, True, True],
        'shared_micro_features': [2, 3],
        'mat_micro_features': [0, 1],
        'mat_points': 6,
        'hyper_hidden_sizes': (8,),
        'hyper_activation': 'sigmoid',
    },
]


# configurations = [
#     # # --- NN configurations ---
#     # {
#     #     'name': 'nn_8_2',
#     #     'model_type': 'nn',
#     #     'nn_hidden_sizes': (8, 8),
#     #     'nn_activation': 'sigmoid',
#     #     'nn_bias': True,
#     #     'mat_parameters': ['vfrac', 'ratio'],
#     #     'norm_matparams': [True, True],
#     # },
#     # {
#     #     'name': 'nn_16_2',
#     #     'model_type': 'nn',
#     #     'nn_hidden_sizes': (16, 16),
#     #     'nn_activation': 'sigmoid',
#     #     'nn_bias': True,
#     #     'mat_parameters': ['vfrac', 'ratio'],
#     #     'norm_matparams': [True, True],
#     # },
#     # {
#     #     'name': 'nn_64_2',
#     #     'model_type': 'nn',
#     #     'nn_hidden_sizes': (64, 64),
#     #     'nn_activation': 'sigmoid',
#     #     'nn_bias': True,
#     #     'mat_parameters': ['vfrac', 'ratio'],
#     #     'norm_matparams': [True, True],
#     # },
#     # {
#     #     'name': 'nn_32_3',
#     #     'model_type': 'nn',
#     #     'nn_hidden_sizes': (32, 32, 32),
#     #     'nn_activation': 'sigmoid',
#     #     'nn_bias': True,
#     #     'mat_parameters': ['vfrac', 'ratio'],
#     #     'norm_matparams': [True, True],
#     # },
#     # {
#     #     'name': 'nn_64_3',
#     #     'model_type': 'nn',
#     #     'nn_hidden_sizes': (64, 64, 64),
#     #     'nn_activation': 'sigmoid',
#     #     'nn_bias': True,
#     #     'mat_parameters': ['vfrac', 'ratio'],
#     #     'norm_matparams': [True, True],
#     # },
#     # {
#     #     'name': 'nn_128_3',
#     #     'model_type': 'nn',
#     #     'nn_hidden_sizes': (128, 128, 128),
#     #     'nn_activation': 'sigmoid',
#     #     'nn_bias': True,
#     #     'mat_parameters': ['vfrac', 'ratio'],
#     #     'norm_matparams': [True, True],
#     # },
#     # {
#     #     'name': 'nn_64_4',
#     #     'model_type': 'nn',
#     #     'nn_hidden_sizes': (64, 64, 64, 64),
#     #     'nn_activation': 'sigmoid',
#     #     'nn_bias': True,
#     #     'mat_parameters': ['vfrac', 'ratio'],
#     #     'norm_matparams': [True, True],
#     # },
#     # {
#     #     'name': 'nn_tanh_64_3',
#     #     'model_type': 'nn',
#     #     'nn_hidden_sizes': (64, 64, 64),
#     #     'nn_activation': 'tanh',
#     #     'nn_bias': True,
#     #     'mat_parameters': ['vfrac', 'ratio'],
#     #     'norm_matparams': [True, True],
#     # },
#     # {
#     #     'name': 'nn_relu_64_3',
#     #     'model_type': 'nn',
#     #     'nn_hidden_sizes': (64, 64, 64),
#     #     'nn_activation': 'relu',
#     #     'nn_bias': True,
#     #     'mat_parameters': ['vfrac', 'ratio'],
#     #     'norm_matparams': [True, True],
#     # },
#     #
#     # # --- Shared Hyper PRNN configurations ---
#     # # - Nonlinear -
#     # {
#     #     'name': 'prnn_nonlin_3L8_6m',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#     #     'norm_matparams': [False, False, True, True],
#     #     'shared_micro_features': [2, 3],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 6,
#     #     'hyper_hidden_sizes': (8, 8, 8),
#     #     'hyper_activation': 'sigmoid',
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'softplus',
#     #     'hidden_dim': 8,
#     # },
#     # {
#     #     'name': 'prnn_nonlin_3L8_6m3',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#     #     'norm_matparams': [False, False, True, True],
#     #     'shared_micro_features': [2, 3],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': [6, 3],
#     #     'hyper_hidden_sizes': (8, 8, 8),
#     #     'hyper_activation': 'sigmoid',
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'softplus',
#     #     'hidden_dim': 8,
#     # },
#     # {
#     #     'name': 'prnn_nonlin_3L8_3m',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#     #     'norm_matparams': [False, False, True, True],
#     #     'shared_micro_features': [2, 3],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 3,
#     #     'hyper_hidden_sizes': (8, 8, 8),
#     #     'hyper_activation': 'sigmoid',
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'softplus',
#     #     'hidden_dim': 8,
#     # },
#     # {
#     #     'name': 'prnn_nonlin_3L8_2m',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#     #     'norm_matparams': [False, False, True, True],
#     #     'shared_micro_features': [2, 3],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 2,
#     #     'hyper_hidden_sizes': (8, 8, 8),
#     #     'hyper_activation': 'sigmoid',
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'softplus',
#     #     'hidden_dim': 8,
#     # },
#     # {
#     #     'name': 'prnn_nonlin_3L8_12m',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#     #     'norm_matparams': [False, False, True, True],
#     #     'shared_micro_features': [2, 3],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 12,
#     #     'hyper_hidden_sizes': (8, 8, 8),
#     #     'hyper_activation': 'sigmoid',
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'softplus',
#     #     'hidden_dim': 8,
#     # },
#     # {
#     #     'name': 'prnn_nonlin_3L8_24m',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#     #     'norm_matparams': [False, False, True, True],
#     #     'shared_micro_features': [2, 3],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 24,
#     #     'hyper_hidden_sizes': (8, 8, 8),
#     #     'hyper_activation': 'sigmoid',
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'softplus',
#     #     'hidden_dim': 8,
#     # },
#     # {
#     #     'name': 'prnn_nonlin_1L8_6m',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#     #     'norm_matparams': [False, False, True, True],
#     #     'shared_micro_features': [2, 3],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 6,
#     #     'hyper_hidden_sizes': (8,),
#     #     'hyper_activation': 'sigmoid',
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'softplus',
#     #     'hidden_dim': 8,
#     # },
#     {
#         'name': 'prnn_nonlin_3L8_3m_sigmoid',
#         'model_type': 'shared_prnn',
#         'encoder_type': 'NonLinear',
#         'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#         'norm_matparams': [False, False, True, True],
#         'shared_micro_features': [2, 3],
#         'mat_micro_features': [0, 1],
#         'mat_points': 3,
#         'hyper_hidden_sizes': (8, 8, 8),
#         'hyper_activation': 'sigmoid',
#         'encoder_n_layers': 3,
#         'encoder_activation': 'sigmoid',
#         'hidden_dim': 8,
#     },
#     # {
#     #     'name': 'prnn_nonlin_3L8_6m_sigmoid',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#     #     'norm_matparams': [False, False, True, True],
#     #     'shared_micro_features': [2, 3],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 6,
#     #     'hyper_hidden_sizes': (8, 8, 8),
#     #     'hyper_activation': 'sigmoid',
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'sigmoid',
#     #     'hidden_dim': 8,
#     # },
#     {
#         'name': 'prnn_nonlin_1L8_3m_sigmoid',
#         'model_type': 'shared_prnn',
#         'encoder_type': 'NonLinear',
#         'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#         'norm_matparams': [False, False, True, True],
#         'shared_micro_features': [2, 3],
#         'mat_micro_features': [0, 1],
#         'mat_points': 3,
#         'hyper_hidden_sizes': (8, 8, 8),
#         'hyper_activation': 'sigmoid',
#         'encoder_n_layers': 1,
#         'encoder_activation': 'sigmoid',
#         'hidden_dim': 8,
#     },
#     # {
#     #     'name': 'prnn_nonlin_1L8_6m_sigmoid',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#     #     'norm_matparams': [False, False, True, True],
#     #     'shared_micro_features': [2, 3],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 6,
#     #     'hyper_hidden_sizes': (8, 8, 8),
#     #     'hyper_activation': 'sigmoid',
#     #     'encoder_n_layers': 1,
#     #     'encoder_activation': 'sigmoid',
#     #     'hidden_dim': 8,
#     # },
#     {
#         'name': 'prnn_nonlin_1L8_12m_sigmoid',
#         'model_type': 'shared_prnn',
#         'encoder_type': 'NonLinear',
#         'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#         'norm_matparams': [False, False, True, True],
#         'shared_micro_features': [2, 3],
#         'mat_micro_features': [0, 1],
#         'mat_points': 12,
#         'hyper_hidden_sizes': (8, 8, 8),
#         'hyper_activation': 'sigmoid',
#         'encoder_n_layers': 1,
#         'encoder_activation': 'sigmoid',
#         'hidden_dim': 8,
#     },
#     # {
#     #     'name': 'prnn_nonlin_3L16_6m_sigmoid',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#     #     'norm_matparams': [False, False, True, True],
#     #     'shared_micro_features': [2, 3],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 6,
#     #     'hyper_hidden_sizes': (8, 8, 8),
#     #     'hyper_activation': 'sigmoid',
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'sigmoid',
#     #     'hidden_dim': 16,
#     # },
#     # {
#     #     'name': 'prnn_nonlin_3L8_12m_sigmoid',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#     #     'norm_matparams': [False, False, True, True],
#     #     'shared_micro_features': [2, 3],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 12,
#     #     'hyper_hidden_sizes': (8, 8, 8),
#     #     'hyper_activation': 'sigmoid',
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'sigmoid',
#     #     'hidden_dim': 8,
#     # },
#     # {
#     #     'name': 'prnn_nonlin_3L8_6m_relu',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#     #     'norm_matparams': [False, False, True, True],
#     #     'shared_micro_features': [2, 3],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 6,
#     #     'hyper_hidden_sizes': (8, 8, 8),
#     #     'hyper_activation': 'sigmoid',
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'relu',
#     #     'hidden_dim': 8,
#     # },
#     # {
#     #     'name': 'prnn_nonlin_3L8relu_6m',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#     #     'norm_matparams': [False, False, True, True],
#     #     'shared_micro_features': [2, 3],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 6,
#     #     'hyper_hidden_sizes': (8, 8, 8),
#     #     'hyper_activation': 'relu',
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'softplus',
#     #     'hidden_dim': 8,
#     # },
#     # # - Linear -
#     # {
#     #     'name': 'prnn_lin_3L_6m3',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'Linear',
#     #     'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#     #     'norm_matparams': [False, False, True, True],
#     #     'shared_micro_features': [2, 3],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': [6, 3],
#     #     'hyper_hidden_sizes': (8, 8, 8),
#     #     'hyper_activation': 'sigmoid',
#     # },
#     # {
#     #     'name': 'prnn_lin_3L_6m',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'Linear',
#     #     'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#     #     'norm_matparams': [False, False, True, True],
#     #     'shared_micro_features': [2, 3],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 6,
#     #     'hyper_hidden_sizes': (8, 8, 8),
#     #     'hyper_activation': 'sigmoid',
#     # },
#     # {
#     #     'name': 'prnn_lin_3L_3m',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'Linear',
#     #     'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#     #     'norm_matparams': [False, False, True, True],
#     #     'shared_micro_features': [2, 3],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 3,
#     #     'hyper_hidden_sizes': (8, 8, 8),
#     #     'hyper_activation': 'sigmoid',
#     # },
#     # {
#     #     'name': 'prnn_lin_3L_2m',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'Linear',
#     #     'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#     #     'norm_matparams': [False, False, True, True],
#     #     'shared_micro_features': [2, 3],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 2,
#     #     'hyper_hidden_sizes': (8, 8, 8),
#     #     'hyper_activation': 'sigmoid',
#     # },
#     # {
#     #     'name': 'prnn_lin_3L_12m',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'Linear',
#     #     'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#     #     'norm_matparams': [False, False, True, True],
#     #     'shared_micro_features': [2, 3],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 12,
#     #     'hyper_hidden_sizes': (8, 8, 8),
#     #     'hyper_activation': 'sigmoid',
#     # },
#     # {
#     #     'name': 'prnn_lin_1L_6m',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'Linear',
#     #     'mat_parameters': ['mu', 'lambda', 'vfrac', 'ratio'],
#     #     'norm_matparams': [False, False, True, True],
#     #     'shared_micro_features': [2, 3],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 6,
#     #     'hyper_hidden_sizes': (8,),
#     #     'hyper_activation': 'sigmoid',
#     # },
# ]

# configurations = [
#     # --- NN configurations ---
#     # {
#     #     'name': 'nn_8_2',
#     #     'model_type': 'nn',
#     #     'nn_hidden_sizes': (8, 8),
#     #     'nn_activation': 'sigmoid',
#     #     'nn_bias': True,
#     #     'mat_parameters': ['mu'],
#     #     'norm_matparams': [True],
#     # },
#     # {
#     #     'name': 'nn_16_2',
#     #     'model_type': 'nn',
#     #     'nn_hidden_sizes': (16, 16),
#     #     'nn_activation': 'sigmoid',
#     #     'nn_bias': True,
#     #     'mat_parameters': ['mu'],
#     #     'norm_matparams': [True],
#     # },
#     # {
#     #     'name': 'nn_16_3',
#     #     'model_type': 'nn',
#     #     'nn_hidden_sizes': (16, 16, 16),
#     #     'nn_activation': 'sigmoid',
#     #     'nn_bias': True,
#     #     'mat_parameters': ['mu'],
#     #     'norm_matparams': [True],
#     # },
#     # {
#     #     'name': 'nn_64_2',
#     #     'model_type': 'nn',
#     #     'nn_hidden_sizes': (64, 64),
#     #     'nn_activation': 'sigmoid',
#     #     'nn_bias': True,
#     #     'mat_parameters': ['mu'],
#     #     'norm_matparams': [True],
#     # },
#     # {
#     #     'name': 'nn_32_3',
#     #     'model_type': 'nn',
#     #     'nn_hidden_sizes': (32, 32, 32),
#     #     'nn_activation': 'sigmoid',
#     #     'nn_bias': True,
#     #     'mat_parameters': ['mu'],
#     #     'norm_matparams': [True],
#     # },
#     # {
#     #     'name': 'nn_64_3',
#     #     'model_type': 'nn',
#     #     'nn_hidden_sizes': (64, 64, 64),
#     #     'nn_activation': 'sigmoid',
#     #     'nn_bias': True,
#     #     'mat_parameters': ['mu'],
#     #     'norm_matparams': [True],
#     # },
#     # {
#     #     'name': 'nn_128_3',
#     #     'model_type': 'nn',
#     #     'nn_hidden_sizes': (128, 128, 128),
#     #     'nn_activation': 'sigmoid',
#     #     'nn_bias': True,
#     #     'mat_parameters': ['mu'],
#     #     'norm_matparams': [True],
#     # },
#     # {
#     #     'name': 'nn_64_4',
#     #     'model_type': 'nn',
#     #     'nn_hidden_sizes': (64, 64, 64, 64),
#     #     'nn_activation': 'sigmoid',
#     #     'nn_bias': True,
#     #     'mat_parameters': ['mu'],
#     #     'norm_matparams': [True],
#     # },
#     # {
#     #     'name': 'nn_tanh_64_3',
#     #     'model_type': 'nn',
#     #     'nn_hidden_sizes': (64, 64, 64),
#     #     'nn_activation': 'tanh',
#     #     'nn_bias': True,
#     #     'mat_parameters': ['mu'],
#     #     'norm_matparams': [True],
#     # },
#     # {
#     #     'name': 'nn_relu_64_3',
#     #     'model_type': 'nn',
#     #     'nn_hidden_sizes': (64, 64, 64),
#     #     'nn_activation': 'relu',
#     #     'nn_bias': True,
#     #     'mat_parameters': ['mu'],
#     #     'norm_matparams': [True],
#     # },
#     #
#     # # --- Shared Hyper PRNN configurations ---
#     # # - Nonlinear -
#     # {
#     #     'name': 'prnn_nonlin_6m',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda'],
#     #     'norm_matparams': [False, False],
#     #     'shared_micro_features': [],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 6,
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'softplus',
#     #     'hidden_dim': 8,
#     # },
#     # {
#     #     'name': 'prnn_nonlin_6m3',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda'],
#     #     'norm_matparams': [False, False],
#     #     'shared_micro_features': [],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': [6, 3],
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'softplus',
#     #     'hidden_dim': 8,
#     # },
#     # {
#     #     'name': 'prnn_nonlin_3m',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda'],
#     #     'norm_matparams': [False, False],
#     #     'shared_micro_features': [],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 3,
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'softplus',
#     #     'hidden_dim': 8,
#     # },
#     # {
#     #     'name': 'prnn_nonlin_2m',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda'],
#     #     'norm_matparams': [False, False],
#     #     'shared_micro_features': [],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 2,
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'softplus',
#     #     'hidden_dim': 8,
#     # },
#     # {
#     #     'name': 'prnn_nonlin_12m',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda'],
#     #     'norm_matparams': [False, False],
#     #     'shared_micro_features': [],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 12,
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'softplus',
#     #     'hidden_dim': 8,
#     # },
#     # {
#     #     'name': 'prnn_nonlin_24m',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda'],
#     #     'norm_matparams': [False, False],
#     #     'shared_micro_features': [],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 24,
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'softplus',
#     #     'hidden_dim': 8,
#     # },
#     # {
#     #     'name': 'prnn_nonlin_3m_sigmoid',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda'],
#     #     'norm_matparams': [False, False],
#     #     'shared_micro_features': [],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 3,
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'sigmoid',
#     #     'hidden_dim': 8,
#     # },
#     # {
#     #     'name': 'prnn_nonlin_6m_sigmoid',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda'],
#     #     'norm_matparams': [False, False],
#     #     'shared_micro_features': [],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 6,
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'sigmoid',
#     #     'hidden_dim': 8,
#     # },
#     # {
#     #     'name': 'prnn_nonlin_12m_sigmoid',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda'],
#     #     'norm_matparams': [False, False],
#     #     'shared_micro_features': [],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 12,
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'sigmoid',
#     #     'hidden_dim': 8,
#     # },
#     # {
#     #     'name': 'prnn_nonlin_1L8_3m_sigmoid',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda'],
#     #     'norm_matparams': [False, False],
#     #     'shared_micro_features': [],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 3,
#     #     'encoder_n_layers': 1,
#     #     'encoder_activation': 'sigmoid',
#     #     'hidden_dim': 8,
#     # },
#     # {
#     #     'name': 'prnn_nonlin_1L8_6m_sigmoid',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda'],
#     #     'norm_matparams': [False, False],
#     #     'shared_micro_features': [],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 6,
#     #     'encoder_n_layers': 1,
#     #     'encoder_activation': 'sigmoid',
#     #     'hidden_dim': 8,
#     # },
#     {
#         'name': 'prnn_nonlin_1L8_6m_sigmoid_scaleMU',
#         'model_type': 'shared_prnn',
#         'encoder_type': 'NonLinear',
#         'mat_parameters': ['mu', 'lambda'],
#         'norm_matparams': [False, False],
#         'shared_micro_features': [],
#         'mat_micro_features': [0, 1],
#         'stress_scaling_feature': 'mu',
#         'mat_points': 6,
#         'encoder_n_layers': 1,
#         'encoder_activation': 'sigmoid',
#         'hidden_dim': 8,
#     },
#     # {
#     #     'name': 'prnn_nonlin_1L8_12m_sigmoid',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda'],
#     #     'norm_matparams': [False, False],
#     #     'shared_micro_features': [],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 12,
#     #     'encoder_n_layers': 1,
#     #     'encoder_activation': 'sigmoid',
#     #     'hidden_dim': 8,
#     # },
#     # {
#     #     'name': 'prnn_nonlin_3L16_6m_sigmoid',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda'],
#     #     'norm_matparams': [False, False],
#     #     'shared_micro_features': [],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 6,
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'sigmoid',
#     #     'hidden_dim': 16,
#     # },
#     # {
#     #     'name': 'prnn_nonlin_6m_relu',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'NonLinear',
#     #     'mat_parameters': ['mu', 'lambda'],
#     #     'norm_matparams': [False, False],
#     #     'shared_micro_features': [],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 6,
#     #     'encoder_n_layers': 3,
#     #     'encoder_activation': 'relu',
#     #     'hidden_dim': 8,
#     # },
#     # # # - Linear -
#     # # {
#     # #     'name': 'prnn_lin_6m3',
#     # #     'model_type': 'shared_prnn',
#     # #     'encoder_type': 'Linear',
#     # #     'mat_parameters': ['mu', 'lambda'],
#     # #     'norm_matparams': [False, False],
#     # #     'shared_micro_features': [],
#     # #     'mat_micro_features': [0, 1],
#     # #     'mat_points': [6, 3],
#     # # },
#     # {
#     #     'name': 'prnn_lin_6m',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'Linear',
#     #     'mat_parameters': ['mu', 'lambda'],
#     #     'norm_matparams': [False, False],
#     #     'shared_micro_features': [],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 6,
#     # },
#     # {
#     #     'name': 'prnn_lin_3m',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'Linear',
#     #     'mat_parameters': ['mu', 'lambda'],
#     #     'norm_matparams': [False, False],
#     #     'shared_micro_features': [],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 3,
#     # },
#     # {
#     #     'name': 'prnn_lin_2m',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'Linear',
#     #     'mat_parameters': ['mu', 'lambda'],
#     #     'norm_matparams': [False, False],
#     #     'shared_micro_features': [],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 2,
#     # },
#     # {
#     #     'name': 'prnn_lin_12m',
#     #     'model_type': 'shared_prnn',
#     #     'encoder_type': 'Linear',
#     #     'mat_parameters': ['mu', 'lambda'],
#     #     'norm_matparams': [False, False],
#     #     'shared_micro_features': [],
#     #     'mat_micro_features': [0, 1],
#     #     'mat_points': 12,
#     # },
# ]

batch_job = False
sys_args = sys.argv
if len(sys_args) > 1:       # There is a parameter given, so part of batch job.
    batch_job = int(sys_args[1])
    if batch_job > len(configurations) - 1:
        print(f"Batch job index {batch_job} exceeds number of configurations {len(configurations)}")
        sys.exit(1)
    else:
        configurations = [configurations[batch_job]]
else:
    if mode != 'plot':
        print(f"No parameter given to python script - testing all configurations!")
        mode = 'test'

# ============================================================================
# MAIN SCRIPT
# ============================================================================

if not os.path.exists(savefolder):
    os.makedirs(savefolder, exist_ok=True)
    print(f"Created save directory: {savefolder}")

# Load dataset once (shared across all configurations)
dataset = LDDataset(
    base_settings['data_path'],
    seq_length=base_settings['seq_length'],
    num_samples=base_settings['num_samples'],
    mat_file=base_settings['matdata_path'],
    mat_features=configurations[0]['mat_parameters'],  # Will reload per config if needed
    norm_stresses=base_settings['norm_stresses'],
    norm_matparams=configurations[0]['norm_matparams'],
    stress_scaling_feature = base_settings['stress_scaling_feature']
)

# Split indices
all_indices = np.arange(base_settings['num_samples'])
train_indices = all_indices[:int(-2 * base_settings['val_test_samples'])]
val_indices = all_indices[int(-2 * base_settings['val_test_samples']):int(-base_settings['val_test_samples'])]
test_indices = all_indices[int(-base_settings['val_test_samples']):]
print(f"Train/Val/Test sizes: {train_indices.shape[0]}/{val_indices.shape[0]}/{test_indices.shape[0]}")

# Storage for all results
all_results = {}

# Global seed counter
seed = base_settings['seed']

base_runs_per_setting = runs_per_setting
if mode == 'plot':
    # Load saved results without running the testing loop
    print("\n" + "="*60)
    print("Loading saved results from config folders")
    print("="*60)

    for config in configurations:
        config_name = config['name']
        config_folder = f"{savefolder}{config_name}/"
        test_losses_file = f"{config_folder}test_losses.txt"

        if os.path.exists(test_losses_file):
            test_losses = np.loadtxt(test_losses_file, delimiter=',')
            all_results[config_name] = test_losses
            print(f"Loaded results for {config_name}: {test_losses.shape}")
        else:
            print(f"WARNING: No results found for {config_name} at {test_losses_file}")

else:
    for config in configurations:
        runs_per_setting = base_runs_per_setting
        config_name = config['name']
        print(f"\n{'='*60}")
        print(f"Configuration: {config_name}")
        print(f"{'='*60}")
        if mode != 'train':
            if config_name == 'prnn_nonlin_1L8_6m_sigmoid':
                runs_per_setting = 9
            if config_name == 'prnn_lin_3L_6m3':
                runs_per_setting = 9  #4

        stress_scaling = config.get('stress_scaling_feature', base_settings['stress_scaling_feature'])

        # Reload dataset with config-specific mat_parameters if different
        dataset = LDDataset(
            base_settings['data_path'],
            seq_length=base_settings['seq_length'],
            num_samples=base_settings['num_samples'],
            mat_file=base_settings['matdata_path'],
            mat_features=config['mat_parameters'],
            norm_stresses=base_settings['norm_stresses'],
            norm_matparams=config['norm_matparams'],
            stress_scaling_feature = stress_scaling
        )
        valset = dataset.get_subset(val_indices)
        testset = dataset.get_subset(test_indices)

        # Compute stress_scaling_index for PRNN models (index of scaling feature in mat_parameters)
        stress_scaling_index = None
        if stress_scaling is not None and config['mat_parameters'] is not None:
            if stress_scaling in config['mat_parameters']:
                stress_scaling_index = config['mat_parameters'].index(stress_scaling)
                # PRNN models divide internal micro-stresses by this feature's raw value,
                # so it must NOT be normalized (otherwise scale would be wrong)
                if isinstance(config['norm_matparams'], list) and config['norm_matparams'][stress_scaling_index]:
                    print(f"WARNING: stress_scaling_feature '{config['stress_scaling_feature']}' is normalized (norm_matparams[{stress_scaling_index}]=True). PRNN internal scaling requires un-normalized values.")

        # Create config-specific folder
        config_folder = f"{savefolder}{config_name}/"
        if not os.path.exists(config_folder):
            os.makedirs(config_folder, exist_ok=True)

        # Initialize test losses array for this config
        test_losses = np.zeros((len(training_samples_array), runs_per_setting))
        test_losses_rel = np.zeros((len(training_samples_array), runs_per_setting))
        test_losses_L2 = np.zeros((len(training_samples_array), runs_per_setting))

        for run_i in range(runs_per_setting):
            for train_samples in training_samples_array:
                print(f"\n--- {config_name}: {train_samples} samples, run {run_i + 1}/{runs_per_setting} ---")

                # Setup seed
                seed += 1
                key = jax.random.PRNGKey(seed)
                np.random.seed(seed)
                random.seed(seed)

                # Create training subset
                cur_indices = train_indices.copy()
                np.random.shuffle(cur_indices)
                cur_indices = cur_indices[:train_samples]
                trainset = dataset.get_subset(cur_indices)

                # Merge base settings with config
                settings = base_settings.copy()
                settings.update(config)
                settings['train_samples'] = train_samples
                settings['train_batch_size'] = min(2, train_samples)
                if train_samples % 4 != 0 and train_samples % 3 == 0:
                    settings['train_batch_size'] = 3
                settings['savename'] = f"{config_folder}samples{train_samples}_run{run_i}"
                settings['seed'] = seed

                # Create model
                if settings['model_type'] == 'prnn':
                    model, params, material = create_prnn_model(
                        random_key=key,
                        n_matpts=settings['mat_points'],
                        encoder_type=settings['encoder_type'],
                        decoder_type=settings['decoder_type'],
                        stress_normalizer=dataset.stress_normalizer,
                        num_mat_features=len(settings['mat_parameters']),
                        enc_m_feats=settings['encoder_micro_features'],
                        dec_m_feats=settings['decoder_micro_features'],
                        mat_m_feats=settings['mat_micro_features'],
                        stress_scaling_index=stress_scaling_index,
                    )
                elif settings['model_type'] == 'nn':
                    model, params, material = create_nn_model(
                        random_key=key,
                        hidden_sizes=settings['nn_hidden_sizes'],
                        num_mat_features=len(settings['mat_parameters']),
                        activation=settings['nn_activation'],
                        use_bias=settings['nn_bias']
                    )
                elif settings['model_type'] == 'shared_prnn':
                    model, params, material = create_shared_hyper_prnn_model(
                        random_key=key,
                        n_micro_raw=len(settings['mat_parameters']),
                        shared_micro_features=settings['shared_micro_features'],
                        n_matpts=settings['mat_points'],
                        encoder_type=settings['encoder_type'],
                        shared_hidden_mult=settings.get('hyper_hidden_mult', 2),
                        shared_output_mult=settings.get('hyper_output_mult', 4),
                        hidden_dim=settings.get('hidden_dim', 8),
                        stress_normalizer=dataset.stress_normalizer,
                        mat_m_feats=settings['mat_micro_features'],
                        hyper_hidden_sizes=settings.get('hyper_hidden_sizes', None),
                        hyper_activation=settings.get('hyper_activation', 'sigmoid'),
                        encoder_n_layers=settings.get('encoder_n_layers', 3),
                        encoder_activation=settings.get('encoder_activation', 'softplus'),
                        stress_scaling_index=stress_scaling_index
                    )

                lr_config = Config(
                    warmup_epochs=settings['warmup_epochs'],
                    schedule_steps=settings['lr_schedule_steps'],
                    steps_per_epoch=settings['train_samples'] / settings['train_batch_size'],
                    base_learning_rate=settings['base_lr'],
                    min_lr_factor=settings['min_lr_factor']
                )

                train_handler = Trainer(model, params, material=material, lr_config=lr_config, random_key=key, out_norm_factor=1)

                if mode == 'train':
                    train_handler.train(trainset, valset, **settings)
                    train_handler.save(settings['savename'])
                    dataset.saveDataparams(settings['savename'] + '_normparams')
                    save_settings(settings, f"{settings['savename']}_settings")
                    jax.clear_caches()

                elif mode == 'test':
                    params = train_handler.load(settings['savename'])
                    test_loss = train_handler.eval_step_L1(train_handler._state, testset, dataset.stress_normalizer.denormalize, material)
                    test_loss_rel = train_handler.eval_relative_norm(train_handler._state, testset, dataset.stress_normalizer.denormalize, material)
                    test_loss_L2 = train_handler.eval_step_L2(train_handler._state, testset, dataset.stress_normalizer.denormalize, material)
                    test_losses[training_samples_array.index(train_samples), run_i] = test_loss
                    test_losses_rel[training_samples_array.index(train_samples), run_i] = test_loss_rel
                    test_losses_L2[training_samples_array.index(train_samples), run_i] = test_loss_L2
                    print(f"Test loss (L1): {test_loss}, L2: {test_loss_L2}, relative: {test_loss_rel}")

        # Save results for this configuration
        np.savetxt(f"{config_folder}test_losses.txt", test_losses, delimiter=',')
        np.savetxt(f"{config_folder}test_losses_rel.txt", test_losses_rel, delimiter=',')
        np.savetxt(f"{config_folder}test_losses_L2.txt", test_losses_L2, delimiter=',')
        all_results[config_name] = test_losses

        # Save config info
        with open(f"{config_folder}config.txt", 'w') as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\ntraining_samples: {training_samples_array}\n")
            f.write(f"runs_per_setting: {runs_per_setting}\n")

        print(f"\nSaved results for {config_name}")

# ============================================================================
# COMBINED PLOT
# ============================================================================
must_include_name = ''

# if mode in ('test', 'plot'):
if mode in ('plot'):
    print("\n" + "="*60)
    print("Creating combined learning curve plot")
    print("="*60)

    fig, ax = plt.subplots(figsize=(5, 3))

    # Use a colormap for many configurations
    cmap = plt.cm.get_cmap('tab20', len(configurations))

    # Map config names to model types
    config_model_types = {c['name']: c['model_type'] for c in configurations}

    def get_linestyle(config_name, model_type):
        """Get linestyle and marker based on model type and name."""
        if model_type == 'nn':
            marker = 'x'
            base_style = '--'
        elif 'LLT' in config_name:
            marker = 'o'
            base_style = '-'
        elif 'Linear' in config_name:
            marker = 's'
            base_style = '-.'
        else:
            marker = 'o'
            base_style = ':'
        if 'grad' in config_name:
            marker = 'x'
        if 'dec' in config_name:
            marker = 'x'
        return base_style, marker

    for i, (config_name, test_losses) in enumerate(all_results.items()):
        if must_include_name not in config_name:
            continue

        color = cmap(i)
        # Expand test losses to 2 axis
        if len(test_losses.shape) == 1:
            test_losses = test_losses.reshape(-1, 1)

        mean_losses = np.mean(test_losses, axis=1)

        linestyle, marker = get_linestyle(config_name, config_model_types.get(config_name))

        # Plot mean line
        ax.plot(training_samples_array, mean_losses, linestyle=linestyle, marker=marker,
                label=config_name, markersize=4, linewidth=1.5)

        # Plot shaded std region
        # std_losses = np.std(test_losses, axis=1)
        # ax.fill_between(training_samples_array, mean_losses - std_losses, mean_losses + std_losses, color=color, alpha=0.2)

    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('L1 Test Loss')
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.savefig(f"{savefolder}{output_fig_name}.pdf", bbox_inches='tight', format='pdf')
    plt.savefig(f"{savefolder}{output_fig_name}.png", dpi=300, bbox_inches='tight', format='png')
    ax.set_xlim(0, 70)
    plt.savefig(f"{savefolder}{output_fig_name}_xlim.pdf", bbox_inches='tight', format='pdf')
    plt.savefig(f"{savefolder}{output_fig_name}_xlim.png", dpi=300, bbox_inches='tight', format='png')

    # Also save a log-scale version
    fig2, ax2 = plt.subplots(figsize=(5, 3))

    for i, (config_name, test_losses) in enumerate(all_results.items()):
        if must_include_name not in config_name:
            continue
        color = cmap(i)
        # Expand test losses to 2 axis
        if len(test_losses.shape) == 1:
            test_losses = test_losses.reshape(-1, 1)
        mean_losses = np.mean(test_losses, axis=1)

        linestyle, marker = get_linestyle(config_name, config_model_types.get(config_name))

        ax2.semilogy(training_samples_array, mean_losses, linestyle=linestyle, marker=marker,
                     label=config_name, markersize=4, linewidth=1.5)
        # std_losses = np.std(test_losses, axis=1)
        # ax2.fill_between(training_samples_array,mean_losses - std_losses,mean_losses + std_losses,color=color, alpha=0.2)

    ax2.set_xlabel('Number of Training Samples')
    ax2.set_ylabel('L1 Test Loss')
    ax2.set_xlim(xmin=0)
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.savefig(f"{savefolder}{output_fig_name}_log.pdf", bbox_inches='tight', format='pdf')
    plt.savefig(f"{savefolder}{output_fig_name}_log.png", dpi=300, bbox_inches='tight', format='png')

    ax2.set_xlim(0, 70)
    plt.savefig(f"{savefolder}{output_fig_name}_log_xlim.pdf", bbox_inches='tight', format='pdf')
    plt.savefig(f"{savefolder}{output_fig_name}_log_xlim.png", dpi=300, bbox_inches='tight', format='png')

    # Log-scale version with minimum across runs (best case)
    fig3, ax3 = plt.subplots(figsize=(5, 3))

    for i, (config_name, test_losses) in enumerate(all_results.items()):
        if must_include_name not in config_name:
            continue
        color = cmap(i)
        min_losses = np.min(test_losses, axis=1)

        linestyle, marker = get_linestyle(config_name, config_model_types.get(config_name))

        ax3.semilogy(training_samples_array, min_losses, linestyle=linestyle, marker=marker,
                     label=config_name, markersize=4, linewidth=1.5)

    ax3.set_xlabel('Number of Training Samples')
    ax3.set_ylabel('L1 Test Loss (best of 10 runs)')
    ax3.set_xlim(xmin=0)
    ax3.legend(loc='upper right', fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)

    plt.savefig(f"{savefolder}{output_fig_name}_log_min.pdf", bbox_inches='tight', format='pdf')
    plt.savefig(f"{savefolder}{output_fig_name}_log_min.png", dpi=300, bbox_inches='tight', format='png')

    # Save summary statistics
    with open(f"{savefolder}summary.txt", 'w') as f:
        f.write("Configuration Summary\n")
        f.write("="*60 + "\n\n")
        for config_name, test_losses in all_results.items():
            mean_losses = np.mean(test_losses, axis=1)
            f.write(f"{config_name}:\n")
            f.write(f"  Final loss (512 samples): {mean_losses[-1]:.4f} +/- {np.std(test_losses[-1]):.4f}\n")
            f.write(f"  Best loss: {np.min(mean_losses):.4f}\n\n")

    print(f"\nAll plots saved to {savefolder}")
