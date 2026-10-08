from deepSI.utils.torch_nets import simple_res_net, feed_forward_nn
from deepSI.fit_systems.encoders import SS_encoder_general_hf

import os
import matplotlib .pyplot as plt
import numpy as np
import torch
import deepSI
from scipy.io import loadmat

from model_augmentation.utils.utils import *
from model_augmentation.fit_systems.interconnect import *
from model_augmentation.fit_systems.blocks import *
from model_augmentation.systems.mass_spring_damper import *

## ------------- Load data -----------------
data_file_path = os.path.join(os.getcwd(), "data/cascaded_tanks/CascadedTankParameters.mat")
mat_contents = loadmat(data_file_path, squeeze_me=True)
CT_params = mat_contents["parameters"]

data_file_path = os.path.join(os.getcwd(), "data/cascaded_tanks/dataBenchmark.mat")
mat_contents = loadmat(data_file_path, squeeze_me=True)
uEst = mat_contents["uEst"]
yEst = mat_contents["yEst"]
uTest = mat_contents["uVal"]
yTest = mat_contents["yVal"]

data_file_path = os.path.join(os.getcwd(), "data/cascaded_tanks/xSim.mat")
mat_contents = loadmat(data_file_path, squeeze_me=True)
x_sim = np.array(mat_contents["xOptPS"])
data_file_path = os.path.join(os.getcwd(), "data/cascaded_tanks/xSimTest.mat")
mat_contents = loadmat(data_file_path, squeeze_me=True)
x_sim_test = np.array(mat_contents["xOptPSTest"])

train_data = deepSI.System_data(u=uEst, y=yEst)
test_data = deepSI.System_data(u=uTest, y=yTest)
val_train_split = 0
val_data = train_data[-val_train_split:]
# train_data = train_data[:-val_train_split]

## ------------- Load encoder baseline -----------------
encoder_file_path = os.path.join(os.getcwd(), "models", "cascaded_tanks", "ct_encoder_baseline")
encoder_baseline : SSE_Interconnect = deepSI.load_system(encoder_file_path)

## ------------- Repeat Training -----------------
rmse_train_list = np.array([])
rmse_test_list = np.array([])

for i in range(1):

    ## ------------- Create Model -----------------
    nx = 2; ny = 1; nu = 1
    nxd = 2
    interconnect = Interconnect(nxd, nu, ny, debugging=False)

    Baseline_block = Cascaded_Tanks_State_Block(nz=nx+nu, nw=nx+ny, params=CT_params, sys_data=train_data)
    # Baseline_block = Parameterized_Cascaded_Tanks_State_Block(nz=nx+nu, nw=nx+ny, params=CT_params, sys_data=train_data)
    ANN_state_block_1 = Static_ANN_Block(nz=1, nw=1, n_nodes_per_layer=32, n_hidden_layers=2, net=zero_init_feed_forward_nn, activation=torch.nn.Tanh)
    ANN_state_block_2 = Static_ANN_Block(nz=nx, nw=1, n_nodes_per_layer=32, n_hidden_layers=2, net=zero_init_feed_forward_nn, activation=torch.nn.Tanh)

    interconnect.add_block(Baseline_block)
    interconnect.add_block(ANN_state_block_1)
    interconnect.add_block(ANN_state_block_2)

    interconnect.connect_signals("x", ANN_state_block_1, "concat", selection_matrix(np.array([0]), nxd))
    interconnect.connect_signals(ANN_state_block_1, "xp", "additive", expansion_matrix(np.array([0]), nxd)*0.1)

    interconnect.connect_signals("x", ANN_state_block_2, "concat", selection_matrix(np.array([0,1]), nxd))
    interconnect.connect_signals(ANN_state_block_2, "xp", "additive", expansion_matrix(np.array([1]), nxd)*0.1)

    interconnect.connect_signals("x", Baseline_block, "concat", selection_matrix(np.array([0,1]), nxd))
    interconnect.connect_block_signals(Baseline_block, ["u"], [])
    interconnect.connect_signals(Baseline_block, "xp", "additive", expansion_matrix(np.array([0,1]), nxd) @ selection_matrix(np.array([0,1]), nx+ny))
    interconnect.connect_signals(Baseline_block, "y", "additive", selection_matrix(np.array([2]), nx+ny))

    fit_sys = SSE_Interconnect(interconnect=interconnect, na=nx*2+1, nb=nx*2+1, e_net_kwargs={"n_nodes_per_layer":16}, na_right=0, nb_right=0)

    # for initializing of models
    nf = 1; epochs = 1; batch_size = 100
    fit_sys.fit(train_sys_data=train_data, val_sys_data=val_data, batch_size=batch_size, epochs=epochs, auto_fit_norm=True, loss_kwargs={'nf':nf}, validation_measure="1-step-RMS")
    fit_sys.bestfit = 10000 # overwrite best fit from above fit procedure
    fit_sys.encoder = encoder_baseline.encoder

    nf = 100; epochs = 500; batch_size = 1000
    fit_sys.fit(train_sys_data=train_data, val_sys_data=train_data, batch_size=batch_size, epochs=epochs, auto_fit_norm=True, loss_kwargs={'nf':nf})

    # ------------- Save RMSE -----------------
    train_fit_sys = fit_sys.apply_experiment(train_data) 
    print(f'Train RMS {train_fit_sys.RMS(train_data)}')
    print(f'Train NRMS {train_fit_sys.NRMS(train_data):.2%}')
    test_fit_sys = fit_sys.apply_experiment(val_data)
    print(f'Test RMS {test_fit_sys.RMS(val_data)}')
    print(f'Test NRMS {test_fit_sys.NRMS(val_data):.2%}')

    rmse_train_list = np.append(rmse_train_list, train_fit_sys.RMS(train_data))
    rmse_test_list = np.append(rmse_test_list, test_fit_sys.RMS(val_data))

    # ------------- Save fit system -----------------
    if test_fit_sys.RMS(val_data) <= rmse_test_list.min():
        model_file_name = "ct_state_aug"
        interconnect_file_path = os.path.join(os.getcwd(), "models", "ecc", model_file_name)
        fit_sys.save_system(interconnect_file_path)

print(rmse_test_list.mean())
print(rmse_test_list.min())
print(rmse_test_list.max())

# ------------- Load best model -----------------
model_file_name = "ct_state_aug"
interconnect_file_path = os.path.join(os.getcwd(), "models", "ecc", model_file_name)
fit_sys : SSE_Interconnect = deepSI.load_system(interconnect_file_path)

# ------------- Plots -----------------
train_fit_sys = fit_sys.apply_experiment(train_data) 
print(f'Train RMS {train_fit_sys.RMS(train_data)}')
print(f'Train NRMS {train_fit_sys.NRMS(train_data):.2%}')
test_fit_sys = fit_sys.apply_experiment(test_data)
print(f'Test RMS {test_fit_sys.RMS(test_data)}')
print(f'Test NRMS {test_fit_sys.NRMS(test_data):.2%}')

plt.subplot(121)
plt.plot(train_data.y[:], label="data")
plt.plot(x_sim[:,1] - 0.140145921650978, label="baseline")
plt.plot(train_fit_sys.y[:], label="augmented")
plt.xlabel("time (s)")
plt.ylabel("tank level")
plt.legend(loc='lower left')
plt.title("train tank 2")

plt.subplot(122)
plt.plot(test_data.y[:], label="data")
plt.plot(x_sim_test[:,1] - 0.140145921650978, label="baseline")
plt.plot(test_fit_sys.y[:], label="augmented")
plt.xlabel("time (s)")
plt.legend(loc='lower left')
plt.title("test tank 2")
plt.show()