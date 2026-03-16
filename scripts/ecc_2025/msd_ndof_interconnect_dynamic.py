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
SNR = 20 # 20, 30, 60

# Phase switches
use_lpv_baseline = False  # Phase-1 LPV milestone: set True
use_augmentation = True   # Phase-1 LPV milestone: set False

# augmentation structure parameters (used only when use_augmentation=True)
dynamic_aug = True  # True or False
type_aug = "parallel"  # "parallel" or "series"
linear_parallel = True  # True or False

# LPV settings (used only when use_lpv_baseline=True)
lpv_p_max = 4.0
lpv_reg_A1_scale = 1.0

# training parameters
nf = 200
epochs = 3000
batch_size = 2000

# utility parameters
save_flag = True  # True or False (dont save model if False, e.g. for debugging)
wait_minutes = 0  # minutes to wait before starting the training (e.g. to not annoy colleagues)

## ------------- Load data -----------------
dof = 3
repo_root = os.getcwd()
data_file_path = os.path.join(repo_root, "data", "mass_spring_damper")
print(data_file_path)
train_data = deepSI.load_system_data(os.path.join(data_file_path, "msd_3dof_multisine_train.npz"))
val_data = deepSI.load_system_data(os.path.join(data_file_path, "msd_3dof_multisine_val.npz"))

## ------------- Add noise -----------------
if SNR == 20:
    sigma_n = 15e-3  # SNR20:15e-3
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
    fp_file_path = os.path.join(repo_root, "data/mass_spring_damper/msd_{0}dof.mat".format(FP_dof))
elif FP_type == "approximate":
    fp_file_path = os.path.join(repo_root, "data/mass_spring_damper/msd_{0}dof_non_ideal.mat".format(FP_dof))
else:
    raise ValueError("FP_type must be either 'ideal' or 'approximate'")
mat_contents = loadmat(fp_file_path, squeeze_me=False)

nx = mat_contents['nx'][0, 0]
ny = mat_contents['ny'][0, 0]
nu = mat_contents['nu'][0, 0]
Ts = mat_contents['Ts'][0, 0]

A_bla = mat_contents['Ad']
B_bla = mat_contents['Bd']
C_bla = mat_contents['Cd']
D_bla = mat_contents['Dd']

A_bar_bla, B_bar_bla, C_bar_bla, D_bar_bla = normalize_linear_ss_matrices(
    A_bla, B_bla, C_bla, D_bla, train_data, state_ix=np.array([0, 1, 2, 3])
)

## ------------- Define model structure -----------------
# Phase-1 LPV milestone uses baseline-only with state dimension equal to FP baseline.
if use_lpv_baseline and not use_augmentation:
    dynamic_aug = False

if dynamic_aug:
    nxd = 2 * dof  # dynamic aug
else:
    nxd = 2 * FP_dof  # static aug

interconnect = Interconnect(nxd, nu, ny, debugging=False)

# Baseline dynamics block
if use_lpv_baseline:
    # Auto-pick scheduling state index from output matrix (SISO): i = argmax(|C[0,:]|)
    sched_state_ix = int(np.argmax(np.abs(C_bar_bla[0, :])))

    physical_state_model_block = Parameterized_LPV_Affine_Linear_State_Block(
        A0=A_bar_bla,
        B0=B_bar_bla,
        A1_init=torch.zeros_like(to_tensor(A_bar_bla)),
        sched_state_ix=sched_state_ix,
        p_max=lpv_p_max,
        RMSE_baseline=1.0,
        reg_A1_scale=lpv_reg_A1_scale,
        flag_loss_reg=True,
        train_B0=False,
    )
else:
    # Existing baseline block used in ECC augmentation work
    physical_state_model_block = Parameterized_MSD_State_Block(nz=5, nw=4, FP_type=FP_type)

physical_output_model_block = Linear_Output_Block(C=C_bar_bla, D=D_bar_bla)
interconnect.add_block(physical_state_model_block)
interconnect.add_block(physical_output_model_block)

# Wire baseline block into xp and y
interconnect.connect_signals("x", physical_state_model_block, "concat", selection_matrix(np.array([0, 1, 2, 3]), nxd))
interconnect.connect_block_signals(physical_state_model_block, ["u"], [])
interconnect.connect_signals(physical_state_model_block, "xp", "additive", expansion_matrix(np.array([0, 1, 2, 3]), nxd))

interconnect.connect_signals("x", physical_output_model_block, "concat", selection_matrix(np.array([0, 1, 2, 3]), nxd))
interconnect.connect_block_signals(physical_output_model_block, ["u"], ["y"])

# Optional augmentation (disabled for Phase-1 LPV milestone)
if use_augmentation:
    if type_aug == "parallel":  # works for both static and dynamic augmentation
        if linear_parallel:
            ANN_state_block = Static_ANN_Block(
                nz=nxd + nu,
                nw=nxd,
                n_nodes_per_layer=8,
                net=zero_init_feed_forward_nn,
                activation=torch.nn.Identity,
            )
        else:
            ANN_state_block = Static_ANN_Block(
                nz=nxd + nu,
                nw=nxd,
                n_nodes_per_layer=8,
                net=zero_init_feed_forward_nn,
                activation=torch.nn.Tanh,
            )
        interconnect.add_block(ANN_state_block)
        interconnect.connect_block_signals(ANN_state_block, ["x", "u"], ["xp"])

    elif type_aug == "series":  # works for both static and dynamic augmentation
        ANN_state_block = Static_ANN_Block(
            nz=nxd + 2 * FP_dof + nu,
            nw=nxd,
            n_nodes_per_layer=8,
            net=identity_init_simple_res_net,
            activation=torch.nn.Tanh,
        )
        interconnect.add_block(ANN_state_block)
        interconnect.connect_block_signals(ANN_state_block, [physical_state_model_block, "x", "u"], ["xp"])

    else:
        raise ValueError("type_aug must be either 'parallel' or 'series'")

# ----- Wait Time To Not Annoy Colleagues -------
for t in range(wait_minutes):
    time.sleep(60)
    print(f"Time passed: {t + 1} minutes")

## ------------- Train fit system -----------------
# Check GPU usage
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

fit_sys = SSE_Interconnect(interconnect=interconnect, na=nxd * 2 + 1, nb=nxd * 2 + 1, e_net_kwargs={"n_nodes_per_layer": 16})
fit_sys.init_model(sys_data=train_data, device=device, auto_fit_norm=True)
fit_sys.fit(
    train_sys_data=train_data,
    val_sys_data=val_data,
    batch_size=batch_size,
    epochs=epochs,
    loss_kwargs={"nf": nf},
    validation_measure="sim-RMS",
)

# ------------- Save fit system -----------------
if save_flag:
    if use_lpv_baseline and not use_augmentation:
        model_file_name = f"msd_{dof}dof_lpv_affine_A0A1_xi2_e{epochs}_SNR{SNR}"
    else:
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
        interconnect_file_path = os.path.join(repo_root, "models", "ecc_corrected", "ideal", "SNR{0}".format(SNR), model_file_name)
    elif FP_type == "approximate":
        interconnect_file_path = os.path.join(repo_root, "models", "ecc_corrected", "approximate", "SNR{0}".format(SNR), model_file_name)
    else:
        raise ValueError("FP_type must be either 'ideal' or 'approximate'")

    os.makedirs(os.path.dirname(interconnect_file_path), exist_ok=True)
    print(f"Saving model to: {interconnect_file_path}")
    fit_sys.save_system(interconnect_file_path)
