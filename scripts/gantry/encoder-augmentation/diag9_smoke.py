"""Minimal smoke test for diag9_training_stability.py imports and data load."""
import sys, os
print("step 1: stdlib ok")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
print("step 2: path set")

import numpy as np
import torch
print("step 3: numpy/torch ok")

import deepSI
print("step 4: deepSI ok")

from model_augmentation.utils.utils import normalize_linear_ss_matrices
from model_augmentation.utils.utils import expansion_matrix, selection_matrix
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn
from model_augmentation.fit_systems.interconnect import SSE_Interconnect, Interconnect
from model_augmentation.fit_systems.blocks import Gantry_State_Block, Linear_Output_Block, Static_ANN_Block
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug
from model_augmentation.systems.gantry_ss import Cd, Dd, P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize
print("step 5: model_augmentation imports ok")

from scipy.io import loadmat
D = 5   # 20000/4000
TS_NEW = 1.0/4000
TRAJ_DIR = os.path.join(PROJECT_ROOT, 'data', 'gantry', 'matlab', 'multisine')
d = loadmat(os.path.join(TRAJ_DIR, 'T1_Y_sweep_conservative.mat'), squeeze_me=True)
u = d['u_total'][::D].astype(np.float32)
y = d['y'][::D].astype(np.float32)
print(f"step 6: data load ok  u={u.shape}  y={y.shape}")

print("ALL OK")
