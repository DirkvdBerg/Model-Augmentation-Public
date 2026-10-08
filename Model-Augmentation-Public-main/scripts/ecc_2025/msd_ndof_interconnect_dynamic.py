import os
import numpy as np
import torch
import deepSI
from scipy.io import loadmat

from model_augmentation.utils.utils import *
from model_augmentation.utils.torch_nets import identity_init_simple_res_net, zero_init_feed_forward_nn
from model_augmentation.fit_systems.interconnect import *
from model_augmentation.fit_systems.blocks import *
from model_augmentation.systems.mass_spring_damper import *

import time

## ------------- Hyper params -----------------
# model structure parameters
FP_type = "approximate" # "ideal" or "approximate"
dynamic_aug = True # True or False
type_aug = "parallel" # "parallel" or "series"
linear_parallel = True # True or False
SNR = 20 # 20, 30, 60

# training parameters
nf = 200; epochs = 3000; batch_size = 2000

# utility parameters
save_flag = True # True or False (dont save model if False, e.g. for debugging)
wait_minutes = 0 # minutes to wait before starting the training (e.g. to not annoy colleagues)

## ------------- Load data -----------------
dof = 3
data_file_path = os.path.join(os.getcwd(), "data", "mass_spring_damper")
print(data_file_path)
train_data = deepSI.load_system_data(os.path.join(data_file_path, "msd_3dof_multisine_train.npz"))
val_data = deepSI.load_system_data(os.path.join(data_file_path, "msd_3dof_multisine_val.npz"))

## ------------- Add noise -----------------
if SNR == 20:
    sigma_n = 15e-3 # SNR20:15e-3
elif SNR == 30:
    sigma_n = 52e-4
elif SNR == 60:
    sigma_n = 15e-5
else:
    raise ValueError("SNR must be either 20, 30 or 60")
train_data.y = train_data.y + np.random.normal(0, sigma_n, train_data.y.shape)
val_data.y = val_data.y + np.random.normal(0, sigma_n, val_data.y.shape)

## ------------- Load FP model -----------------
FP_dof = 2
if FP_type == "ideal":
    data_file_path = os.path.join(os.getcwd(), "data/mass_spring_damper/msd_{0}dof.mat".format(FP_dof))
elif FP_type == "approximate":
    data_file_path = os.path.join(os.getcwd(), "data/mass_spring_damper/msd_{0}dof_non_ideal.mat".format(FP_dof))
else:
    raise ValueError("FP_type must be either 'ideal' or 'approximate'")
mat_contents = loadmat(data_file_path, squeeze_me=False)

nx = mat_contents['nx'][0,0]; ny = mat_contents['ny'][0,0]; nu = mat_contents['nu'][0,0]
Ts = mat_contents['Ts'][0,0]

A_bla = mat_contents['Ad']
B_bla = mat_contents['Bd']
C_bla = mat_contents['Cd']
D_bla = mat_contents['Dd']

A_bar_bla, B_bar_bla, C_bar_bla, D_bar_bla = normalize_linear_ss_matrices(A_bla, B_bla, C_bla, D_bla, train_data, state_ix=np.array([0,1,2,3]))

## ------------- Define augmentations structure -----------------
if dynamic_aug:
    nxd = 2*dof # dynamic aug
else:
    nxd = 2*FP_dof # static aug

interconnect = Interconnect(nxd, nu, ny, debugging=False)
# physical_state_model_block = Linear_State_Block(A=A_bar_bla, B=B_bar_bla)
# physical_state_model_block = Parameterized_Linear_State_Block(A=A_bar_bla, B=B_bar_bla)
physical_state_model_block = Parameterized_MSD_State_Block(nz = 5, nw = 4, FP_type=FP_type)
physical_output_model_block = Linear_Output_Block(C=C_bar_bla, D=D_bar_bla)
interconnect.add_block(physical_state_model_block)
interconnect.add_block(physical_output_model_block)

# ----- (dynamic) parallel -------
if type_aug == "parallel": # works for both static and dynamic augmentation
    if linear_parallel:
        ANN_state_block = Static_ANN_Block(nz=nxd+nu, nw=nxd, n_nodes_per_layer=8, net=zero_init_feed_forward_nn, activation=torch.nn.Identity)
    else:
        ANN_state_block = Static_ANN_Block(nz=nxd+nu, nw=nxd, n_nodes_per_layer=8, net=zero_init_feed_forward_nn, activation=torch.nn.Tanh)
    interconnect.add_block(ANN_state_block)

    interconnect.connect_block_signals(ANN_state_block, ["x", "u"], ["xp"])

    interconnect.connect_signals("x", physical_state_model_block, "concat", selection_matrix(np.array([0,1,2,3]), nxd))
    interconnect.connect_block_signals(physical_state_model_block, ["u"], [])
    interconnect.connect_signals(physical_state_model_block, "xp", "additive", expansion_matrix(np.array([0,1,2,3]), nxd))

    interconnect.connect_signals("x", physical_output_model_block, "concat", selection_matrix(np.array([0,1,2,3]), nxd))
    interconnect.connect_block_signals(physical_output_model_block, ["u"], ["y"])

# ----- (dynamic) series -------
elif type_aug == "series": # works for both static and dynamic augmentation
    ANN_state_block = Static_ANN_Block(nz=nxd+2*FP_dof+nu, nw=nxd, n_nodes_per_layer=8, net=identity_init_simple_res_net, activation=torch.nn.Tanh)

    interconnect.add_block(ANN_state_block)

    interconnect.connect_block_signals(ANN_state_block, [physical_state_model_block, "x", "u"], ["xp"])

    interconnect.connect_signals("x", physical_state_model_block, "concat", selection_matrix(np.array([0,1,2,3]), nxd))
    interconnect.connect_block_signals(physical_state_model_block, ["u"], [])

    interconnect.connect_signals("x", physical_output_model_block, "concat", selection_matrix(np.array([0,1,2,3]), nxd))
    interconnect.connect_block_signals(physical_output_model_block, ["u"], ["y"])
else:
    raise ValueError("type_aug must be either 'parallel' or 'series'")

# ----- Wait Time To Not Annoy Colleagues -------
for t in range(wait_minutes):
    time.sleep(60)
    print(f"Time passed: {t+1} minutes")

## ------------- Train fit system -----------------
fit_sys = SSE_Interconnect(interconnect=interconnect, na=nxd*2+1, nb=nxd*2+1, e_net_kwargs={"n_nodes_per_layer":16})
fit_sys.fit(train_sys_data=train_data, val_sys_data=val_data, batch_size=batch_size, epochs=epochs, auto_fit_norm=True, loss_kwargs={'nf':nf}, validation_measure="sim-RMS")

# ------------- Save fit system -----------------
if save_flag:
    if type_aug == "parallel" and dynamic_aug:
        if linear_parallel:
            model_file_name = "msd_{0}dof_linear_dynamic_parallel_e{1}".format(dof, epochs)
        else:
            model_file_name = "msd_{0}dof_dynamic_parallel_e{1}".format(dof, epochs)
    elif type_aug == "parallel" and not dynamic_aug:
        model_file_name = "msd_{0}dof_static_parallel_e{1}".format(dof, epochs)
    elif type_aug == "series" and dynamic_aug:
        model_file_name = "msd_{0}dof_dynamic_series_e{1}".format(dof, epochs)
    elif type_aug == "series" and not dynamic_aug:
        model_file_name = "msd_{0}dof_static_series_e{1}".format(dof, epochs)
    else:
        raise ValueError("Not a valid model augmentation")

    if FP_type == "ideal":
        interconnect_file_path = os.path.join(os.getcwd(), "models", "ecc_corrected", "ideal", "SNR{0}".format(SNR), model_file_name)
    elif FP_type == "approximate":
        interconnect_file_path = os.path.join(os.getcwd(), "models", "ecc_corrected", "approximate", "SNR{0}".format(SNR), model_file_name)
    else:
        raise ValueError("FP_type must be either 'ideal' or 'approximate'")
    fit_sys.save_system(interconnect_file_path)