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
base_model = "BLA_3" # "FP" or "FP_nonlin" or "BLA_2" or "BLA_3"
nx = 3; # 2 for static or 3 for dynamic
nu = 1; ny = 1
pre_encoder_flag = False
save_flag = True

# input data
data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/uest_multisine.mat")
mat_contents = loadmat(data_file_path, squeeze_me=True)
uEst = mat_contents["u"]

# output data
data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/yest_multisine.mat")
mat_contents = loadmat(data_file_path, squeeze_me=True)
yEst = mat_contents["y"]

# state baseline simulation data
if base_model == "FP":
    data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/baseline_models/states_FP_linear.mat")
    mat_contents = loadmat(data_file_path, squeeze_me=True)
    xEst = mat_contents["xEst_lin"]
    data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/baseline_models/states_val_FP_linear.mat")
    mat_contents = loadmat(data_file_path, squeeze_me=True)
    xTest = mat_contents["xTest_lin"]
elif base_model == "FP_nonlin":
    data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/baseline_models/states_FP_nonlinear.mat")
    mat_contents = loadmat(data_file_path, squeeze_me=True)
    xEst = mat_contents["xOptEst"]
    data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/baseline_models/states__val_FP_nonlinear.mat")
    mat_contents = loadmat(data_file_path, squeeze_me=True)
    xTest = mat_contents["xOptTest"]
elif base_model == "BLA_2":
    data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/baseline_models/states_BLA_2.mat")
    mat_contents = loadmat(data_file_path, squeeze_me=True)
    xEst = mat_contents["xEst_ss"]
    data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/baseline_models/states_val_BLA_2.mat")
    mat_contents = loadmat(data_file_path, squeeze_me=True)
    xTest = mat_contents["xTest_ss"]
elif base_model == "BLA_3":
    data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/baseline_models/states_BLA_3.mat")
    mat_contents = loadmat(data_file_path, squeeze_me=True)
    xEst = mat_contents["xEst_ss"]
    data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/baseline_models/states_val_BLA_3.mat")
    mat_contents = loadmat(data_file_path, squeeze_me=True)
    xTest = mat_contents["xTest_ss"]

data = deepSI.System_data(u=uEst, y=yEst)
train_data, val_data = data.train_test_split(split_fraction=0.25)

train_data = System_data_with_x(x=xEst, u=train_data.u, y=train_data.y)
val_data = System_data_with_x(x=xTest, u=val_data.u, y=val_data.y)

# ------------- Train fit system -----------------

if base_model == "FP":
    data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/baseline_models/sys_FP_linear.mat")
    mat_contents = loadmat(data_file_path, squeeze_me=False)
    A_bla = mat_contents['A']
    B_bla = mat_contents['B']
    C_bla = mat_contents['C']
    D_bla = mat_contents['D']
    RMSE_baseline = 0.0017
    nx_base = 2
elif base_model == "FP_nonlin":
    data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/baseline_models/aL_FP_nonlinear.mat")
    mat_contents = loadmat(data_file_path, squeeze_me=False)
    aL = mat_contents['aL'][0,0]
    data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/baseline_models/sys_FP_nonlinear.mat")
    mat_contents = loadmat(data_file_path, squeeze_me=False)
    A_bla = mat_contents['A']
    B_bla = mat_contents['B']
    C_bla = mat_contents['C']
    D_bla = mat_contents['D']
    RMSE_baseline = 1.95e-4
    nx_base = 2
elif base_model == "BLA_2":
    data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/baseline_models/sys_BLA_2.mat")
    mat_contents = loadmat(data_file_path, squeeze_me=False)
    A_bla = mat_contents['A']
    B_bla = mat_contents['B']
    C_bla = mat_contents['C']
    D_bla = mat_contents['D']
    RMSE_baseline = 1.65e-4
    nx_base = 2
elif base_model == "BLA_3":
    data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/baseline_models/sys_BLA_3.mat")
    mat_contents = loadmat(data_file_path, squeeze_me=False)
    A_bla = mat_contents['A']
    B_bla = mat_contents['B']
    C_bla = mat_contents['C']
    D_bla = mat_contents['D']
    RMSE_baseline = 1.65e-4
    nx_base = 3

FP_state_ix = np.arange(nx_base)
A_bar_bla, B_bar_bla, C_bar_bla, D_bar_bla = normalize_linear_ss_matrices(A_bla, B_bla, C_bla, D_bla, train_data, state_ix=FP_state_ix)

# normalize state data
std_x = train_data.x.std(axis=0)
train_data = System_data_with_x(x=train_data.x/std_x, u=train_data.u, y=train_data.y)
val_data = System_data_with_x(x=val_data.x/std_x, u=val_data.u, y=val_data.y)

## ------------- Pre-encoder -----------------
e_net_kwargs = {"n_hidden_layers": 2, "n_nodes_per_layer": 16}
na = nb = nx*4+1

encoder_sys = SS_pre_encoder(nx=nx_base, na=na, nb=nb, e_net_kwargs=e_net_kwargs)
if pre_encoder_flag:
    nf = 1; epochs = 200; batch_size = 2000
    encoder_sys.fit(train_sys_data=train_data, val_sys_data=val_data, batch_size=batch_size, epochs=epochs, auto_fit_norm=True, loss_kwargs={'nf':nf}, validation_measure="1-step-RMS")

## ------------- Baseline model validation -----------------
interconnect = Interconnect(nx, nu, ny, debugging=False)
# physical_state_model_block = Linear_State_Block(A=A_bar_bla, B=B_bar_bla)
# physical_output_model_block = Linear_Output_Block(C=C_bar_bla, D=D_bar_bla)

physical_state_model_block = Parameterized_Linear_State_Block(A=A_bar_bla, B=B_bar_bla, RMSE_baseline=RMSE_baseline)
physical_output_model_block = Parameterized_Linear_Output_Block(C=C_bar_bla, D=D_bar_bla, RMSE_baseline=RMSE_baseline)

interconnect.add_block(physical_state_model_block)
interconnect.add_block(physical_output_model_block)
if base_model == "FP_nonlin":
    cubic_block = Cubic_Block(nz=1, nw=1, aL=-aL*(std_x[0]**3)/std_x[1])
    interconnect.add_block(cubic_block)

# ----- parallel -------
ANN_state_block = Static_ANN_Block(nz=nx+nu, nw=nx, n_nodes_per_layer=8, net=zero_init_feed_forward_nn, activation=torch.nn.Tanh)
interconnect.add_block(ANN_state_block)

interconnect.connect_block_signals(ANN_state_block, ["x", "u"], ["xp"])

interconnect.connect_signals("x", physical_state_model_block, "concat", selection_matrix(FP_state_ix, nx))
interconnect.connect_block_signals(physical_state_model_block, ["u"], [])
interconnect.connect_signals(physical_state_model_block, "xp", "additive", expansion_matrix(FP_state_ix, nx))

interconnect.connect_signals("x", physical_output_model_block, "concat", selection_matrix(FP_state_ix, nx))
interconnect.connect_block_signals(physical_output_model_block, ["u"], ["y"])

if base_model == "FP_nonlin":
    interconnect.connect_signals("x", cubic_block, "concat", selection_matrix([0], nx))
    interconnect.connect_block_signals(cubic_block, [], [])
    interconnect.connect_signals(cubic_block, "xp", "additive", expansion_matrix([1], nx))

## ------ init fit -------
train_data = System_data(u=train_data.u, y=train_data.y)
val_data = System_data(u=val_data.u, y=val_data.y)

fit_sys = SSE_Interconnect(interconnect=interconnect, na=na, nb=nb)

## ------ expand encoder -------
if pre_encoder_flag:
    nf = 1; epochs = 1; batch_size = 100
    fit_sys.fit(train_sys_data=train_data, val_sys_data=val_data, batch_size=batch_size, epochs=epochs, auto_fit_norm=True, loss_kwargs={'nf':nf}, validation_measure="1-step-RMS")
    fit_sys.bestfit = 10000 # overwrite best fit from above fit procedure


    *_, last_non_lin_layer = encoder_sys.encoder.net.net_non_lin.net.modules()
    lin_layer = encoder_sys.encoder.net.net_lin

    from torch import nn, optim
    new_encoder = simple_res_net(n_in=nb*nu + na*ny, n_out=nx, n_nodes_per_layer=e_net_kwargs["n_nodes_per_layer"], n_hidden_layers=e_net_kwargs["n_hidden_layers"], activation=nn.Tanh)

    *_, new_last_non_lin_layer = new_encoder.net_non_lin.net.modules()
    new_last_non_lin_layer.weight.data[:nx_base,:] = last_non_lin_layer.weight
    new_last_non_lin_layer.bias.data[:nx_base] = last_non_lin_layer.bias

    new_lin_layer = new_encoder.net_lin
    new_lin_layer.weight.data[:nx_base,:] = lin_layer.weight
    new_lin_layer.bias.data[:nx_base] = lin_layer.bias

    fit_sys.encoder = encoder_sys.encoder
    fit_sys.encoder.net = new_encoder

## ------ fit sys -------
nf = 500; epochs = 4000; batch_size = 2000
fit_sys.fit(train_sys_data=train_data, val_sys_data=val_data, batch_size=batch_size, epochs=epochs, auto_fit_norm=True, loss_kwargs={'nf':nf}, validation_measure="sim-RMS")

## ------------- Save fit system -----------------
if base_model == "FP":
    if nx == 2:
        model_file_name = "boucWen-FP_linear-static_parallel_e{0}".format(epochs)
    if nx == 3:
        model_file_name = "boucWen-FP_linear-dynamic_parallel_e{0}".format(epochs)
if base_model == "FP_nonlin":
    if nx == 2:
        model_file_name = "boucWen-FP_nonlinear-static_parallel_e{0}".format(epochs)
    if nx == 3:
        model_file_name = "boucWen-FP_nonlinear-dynamic_parallel_e{0}".format(epochs)
if base_model == "BLA_2":
    if nx == 2:
        model_file_name = "boucWen-BLA_2-static_parallel_e{0}".format(epochs)
    if nx == 3:
        model_file_name = "boucWen-BLA_2-dynamic_parallel_e{0}".format(epochs)
if base_model == "BLA_3":
    model_file_name = "boucWen-BLA_3-parallel_e{0}".format(epochs)

if save_flag:
    interconnect_file_path = os.path.join(os.getcwd(), "models", "bouc_wen", model_file_name)
    fit_sys.save_system(interconnect_file_path)

for m in fit_sys.hfn.connected_blocks:
    if isinstance(m, Parameterized_Linear_State_Block):
        print(m.A)
        print(m.A_init)
        print(m.B)
        print(m.B_init)
    elif isinstance(m, Parameterized_Linear_Output_Block):
        print(m.C)
        print(m.C_init)
        print(m.D)
        print(m.D_init)