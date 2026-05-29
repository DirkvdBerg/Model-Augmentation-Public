from deepSI.utils.torch_nets import simple_res_net, feed_forward_nn
from deepSI.fit_systems.encoders import SS_encoder_general_hf

import os
import matplotlib .pyplot as plt
import numpy as np
import torch
import deepSI
from scipy.io import loadmat

from model_augmentation.utils.utils import *
from model_augmentation.utils.torch_nets import identity_init_simple_res_net, zero_init_feed_forward_nn, linear_mapping
from model_augmentation.fit_systems.interconnect import *
from model_augmentation.fit_systems.blocks import *
from model_augmentation.fit_systems.pre_encoder import SS_pre_encoder, System_data_with_x
from model_augmentation.systems.mass_spring_damper import *

import time

## ------------- Load data -----------------
sys_dof = 2

data_file_path = os.path.join(os.getcwd(), "data", "mass_spring_damper")
print(data_file_path)
train_data = deepSI.load_system_data(os.path.join(data_file_path, "msd_{0}dof_multisine_train.npz".format(sys_dof)))
val_data = deepSI.load_system_data(os.path.join(data_file_path, "msd_{0}dof_multisine_val.npz".format(sys_dof)))

std_x = train_data.x.std(axis=0)

## ------------- Add noise -----------------
sigma_n = 52e-4 # SNR60:15e-5; SNR30:52e-4; SNR20:15e-3
train_data.y = train_data.y + np.random.normal(0, sigma_n, train_data.y.shape)
val_data.y = val_data.y + np.random.normal(0, sigma_n, val_data.y.shape)

## ------------- Train fit system -----------------
FP_dof = 2
data_file_path = os.path.join(os.getcwd(), "data/mass_spring_damper/msd_{0}dof.mat".format(FP_dof))
mat_contents = loadmat(data_file_path, squeeze_me=False)

nx = mat_contents['nx'][0,0]; ny = mat_contents['ny'][0,0]; nu = mat_contents['nu'][0,0]
Ts = mat_contents['Ts'][0,0]

A_bla = mat_contents['Ad']
B_bla = mat_contents['Bd']
C_bla = mat_contents['Cd']
D_bla = mat_contents['Dd']

FP_state_ix = np.arange(FP_dof*2)
A_bar_bla, B_bar_bla, C_bar_bla, D_bar_bla = normalize_linear_ss_matrices(A_bla, B_bla, C_bla, D_bla, train_data, state_ix=FP_state_ix)

train_data = System_data_with_x(x=train_data.x/std_x, u=train_data.u, y=train_data.y)
val_data = System_data_with_x(x=val_data.x/std_x, u=val_data.u, y=val_data.y)

print(train_data.x.shape)

## ------------- Pre-encoder -----------------
encoder_sys = SS_pre_encoder(nx=nx, na=nx*4+1, nb=nx*4+1)

nf = 1; epochs = 100; batch_size = 512
encoder_sys.fit(train_sys_data=train_data, val_sys_data=val_data, batch_size=batch_size, epochs=epochs, auto_fit_norm=True, loss_kwargs={'nf':nf}, validation_measure="1-step-RMS")

## ------------- Baseline model validation -----------------
nxd = 2*FP_dof

interconnect = Interconnect(nxd, nu, ny, debugging=False)
physical_state_model_block = Linear_State_Block(A=A_bar_bla, B=B_bar_bla)
physical_output_model_block = Linear_Output_Block(C=C_bar_bla, D=D_bar_bla)
interconnect.add_block(physical_state_model_block)
interconnect.add_block(physical_output_model_block)

# ----- residual -------
interconnect.connect_signals("x", physical_state_model_block, "concat", selection_matrix(FP_state_ix, nxd))
interconnect.connect_block_signals(physical_state_model_block, ["u"], [])
interconnect.connect_signals(physical_state_model_block, "xp", "additive", expansion_matrix(FP_state_ix, nxd))

interconnect.connect_signals("x", physical_output_model_block, "concat", selection_matrix(FP_state_ix, nxd))
interconnect.connect_block_signals(physical_output_model_block, ["u"], ["y"])

# ------ fit -------
fit_sys = SSE_Interconnect(interconnect=interconnect, na=nxd*4+1, nb=nxd*4+1)

nf = 1; epochs = 1; batch_size = 100
fit_sys.fit(train_sys_data=train_data, val_sys_data=val_data, batch_size=batch_size, epochs=epochs, auto_fit_norm=True, loss_kwargs={'nf':nf}, validation_measure="1-step-RMS")
fit_sys.bestfit = 10000 # overwrite best fit from above fit procedure
fit_sys.encoder = encoder_sys.encoder

nf = 200; epochs = 500; batch_size = 2000
fit_sys.fit(train_sys_data=train_data, val_sys_data=val_data, batch_size=batch_size, epochs=epochs, auto_fit_norm=True, loss_kwargs={'nf':nf}, validation_measure="sim-RMS")