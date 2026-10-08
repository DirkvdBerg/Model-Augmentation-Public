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

# train_data = deepSI.System_data(u=uEst, y=yEst)
test_data = deepSI.System_data(u=uTest, y=yTest)
# test_data = test_data[512:]

## ------------- Load fit sys -----------------
model_file_name = "ct_state_aug"
interconnect_file_path = os.path.join(os.getcwd(), "models", "ecc", model_file_name)
fit_sys : SSE_Interconnect = deepSI.load_system(interconnect_file_path)

# ------------- Simulation Plot -----------------
# train_fit_sys = fit_sys.apply_experiment(train_data) 
# print(f'Train RMS {train_fit_sys.RMS(train_data)}')
# print(f'Train NRMS {train_fit_sys.NRMS(train_data):.2%}')
test_fit_sys = fit_sys.apply_experiment(test_data)
print(f'Test RMS {test_fit_sys.RMS(test_data)}')
print(f'Test NRMS {test_fit_sys.NRMS(test_data):.2%}')

scaling = 2.5/10; fig1 = plt.figure(figsize=[26*scaling, 9.3*scaling])
# plt.subplot(121)
# plt.plot(train_data.y[:], label="data")
# plt.plot(x_sim[:,1] - 0.140145921650978, label="baseline")
# plt.plot(train_fit_sys.y[:], label="augmented")
# plt.xlabel("time (s)")
# plt.ylabel("tank level")
# plt.legend(loc='lower left')
# plt.title("train tank 2")

# plt.subplot(122)
plt.plot(test_data.y[:], label="data")
plt.plot(x_sim_test[:,1] - 0.140145921650978, label="baseline")
plt.plot(test_fit_sys.y[:], label="augmented")
plt.xlabel("time (s)")
plt.ylabel("magnitude (V)")
plt.legend(loc="lower center", prop = { "size": 8.1}, ncols=3, columnspacing=0.5)
# plt.legend(loc='lower left')
# plt.title("test tank 2")
plt.grid()

fig_file_path = os.path.join(os.getcwd(), "figures", "cascaded_tanks")
plt.savefig(os.path.join(fig_file_path, "fitted_models_pred_error.svg"), transparent=True, format="svg")
plt.savefig(os.path.join(fig_file_path, "fitted_models_pred_error.png"), transparent=True, format="png")
plt.savefig(os.path.join(fig_file_path, "fitted_models_pred_error.eps"), transparent=True, format="eps")

plt.show()

# # ------------- Validation Loss Plot -----------------
# scaling = 6/10; fig2 = plt.figure(figsize=[20*scaling, 10*scaling])
# fit_sys.checkpoint_load_system('_last')
# plt.semilogy(fit_sys.epoch_id,fit_sys.Loss_val) 
# plt.xlabel('batch id (number of updates)')
# plt.ylabel('RMS error')
# plt.legend(loc="upper center", prop = { "size": 8.1}, ncols=3, columnspacing=0.5)
# plt.tight_layout()
# plt.grid()

# plt.savefig(os.path.join(fig_file_path, "val_loss_plot_file_name.svg"), format="svg")
# plt.savefig(os.path.join(fig_file_path, "val_loss_plot_file_name.png"), format="png")
# plt.savefig(os.path.join(fig_file_path, "val_loss_plot_file_name.eps"), format="eps")

# plt.show()

# # ------------- n-step error Plot -----------------
# scaling = 6/10; fig3 = plt.figure(figsize=[20*scaling, 10*scaling])
# nsteperror= fit_sys.n_step_error(test_data, nf=100)
# plt.plot(nsteperror)
# plt.xlabel('n step in the future')
# plt.ylabel('NRMS error')
# plt.legend()
# plt.grid()

# plt.savefig(os.path.join(fig_file_path, "n_step_error.svg"), format="svg")
# plt.savefig(os.path.join(fig_file_path, "n_step_error.png"), format="png")
# plt.savefig(os.path.join(fig_file_path, "n_step_error.eps"), format="eps")

# plt.show()