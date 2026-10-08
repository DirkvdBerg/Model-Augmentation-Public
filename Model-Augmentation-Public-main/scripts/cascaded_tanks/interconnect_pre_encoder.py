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

train_data = deepSI.System_data(u=uEst, y=x_sim)
test_data = deepSI.System_data(u=uTest, y=x_sim_test)
val_data = test_data[:512]
## ------------- Create Model -----------------
nx = 2; ny = 2; nu = 1
nxd = 2
interconnect = Interconnect(nxd, nu, ny, debugging=False)

Baseline_block = Cascaded_Tanks_State_Block(nz=nx+nu, nw=nx+ny, params=CT_params, sys_data=train_data)

interconnect.add_block(Baseline_block)

interconnect.connect_signals("x", Baseline_block, "concat", selection_matrix(np.array([0,1]), nxd))
interconnect.connect_block_signals(Baseline_block, ["u"], [])
interconnect.connect_signals(Baseline_block, "xp", "additive", expansion_matrix(np.array([0,1]), nxd) @ selection_matrix(np.array([0,1]), nx+ny))
interconnect.connect_signals(Baseline_block, "y", "additive", selection_matrix(np.array([2,3]), nx+ny))


encoder_file_path = os.path.join(os.getcwd(), "models", "cascaded_tanks", "ct_encoder_baseline")
fit_sys : SSE_Interconnect = deepSI.load_system(encoder_file_path)

# fit_sys = SSE_Interconnect(interconnect=interconnect, na=nx*2+1, nb=nx*2+1, e_net_kwargs={"n_nodes_per_layer":16}, na_right=0, nb_right=0)

# nf = 1; epochs = 20000; batch_size = 512
# fit_sys.fit(train_sys_data=train_data, val_sys_data=val_data, batch_size=batch_size, epochs=epochs, auto_fit_norm=True, loss_kwargs={'nf':nf}, validation_measure="1-step-RMS")

# ------------- Save fit system -----------------
# model_file_name = "ct_encoder_baseline_test".format(epochs, nf, batch_size)
# interconnect_file_path = os.path.join(os.getcwd(), "models", "cascaded_tanks", model_file_name)
# fit_sys.save_system(interconnect_file_path)

# ------------- Plots -----------------
train_fit_sys = fit_sys.apply_experiment(train_data) 
print(f'Train RMS {train_fit_sys.RMS(train_data)}')
print(f'Train NRMS {train_fit_sys.NRMS(train_data):.2%}')
test_fit_sys = fit_sys.apply_experiment(test_data)
print(f'Test RMS {test_fit_sys.RMS(test_data)}')
print(f'Test NRMS {test_fit_sys.NRMS(test_data):.2%}')

plt.subplot(121)
# plt.plot(train_data.y[:,0], label="data")
plt.plot(x_sim[:,0], label="simulation")
plt.plot(train_fit_sys.y[:,0], label="encoder")
plt.xlabel("time (s)")
plt.ylabel("tank level")
plt.legend(loc='lower left')
plt.title("train tank 2")

plt.subplot(122)
# plt.plot(train_data.y[:,1], label="data")
plt.plot(x_sim[:,1], label="simulation")
plt.plot(train_fit_sys.y[:,1], label="encoder")
plt.xlabel("time (s)")
plt.ylabel("tank level")
plt.legend(loc='lower left')
plt.title("train tank 2")

# plt.subplot(122)
# plt.plot(test_data.y[:], label="data")
# plt.plot(x_sim_test[:,1] - 0.140145921650978, label="baseline")
# plt.plot(test_fit_sys.y[:], label="augmented")
# plt.xlabel("time (s)")
# plt.legend(loc='lower left')
# plt.title("test tank 2")
plt.show()

scaling = 6/10; fig3 =  plt.figure(figsize=[8.9*scaling, 4.5*scaling])
fit_sys.checkpoint_load_system('_last')
plt.semilogy(fit_sys.epoch_id,fit_sys.Loss_val) 
plt.xlabel('batch id (number of updates)')
plt.ylabel('RMS error')
plt.legend(loc="upper center", prop = { "size": 8.1}, ncols=3, columnspacing=0.5)
plt.tight_layout()
plt.grid()
plt.show()

# fig2 = plt.figure()
# nsteperror= fit_sys.n_step_error(test_data, nf=400)
# plt.plot(nsteperror, label='Augmented model n-step-error')
# # plt.plot(nsteperror_ss_enc, label='SS encoder n-step-error')
# plt.xlabel('n step in the future')
# plt.ylabel('NRMS error')
# plt.legend()
# plt.grid()
# plt.show()