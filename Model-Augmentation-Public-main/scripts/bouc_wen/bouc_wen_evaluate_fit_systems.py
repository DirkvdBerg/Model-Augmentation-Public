from deepSI.utils.torch_nets import simple_res_net
from deepSI.fit_systems.encoders import SS_encoder_general_hf
import deepSI

import os
import numpy as np
import torch
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from model_augmentation.utils.utils import *
from model_augmentation.fit_systems.interconnect import *
from model_augmentation.fit_systems.blocks import *
from model_augmentation.systems.mass_spring_damper import *

# flags
test_type = "sinesweep" # "multisine" or "sinesweep"

## ------------- Load data -----------------
# input data
if test_type == "multisine":
    data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/uval_multisine.mat")
    mat_contents = loadmat(data_file_path, squeeze_me=True)
    uTest = mat_contents["uval_multisine"]
elif test_type == "sinesweep":
    data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/uval_sinesweep.mat")
    mat_contents = loadmat(data_file_path, squeeze_me=True)
    uTest = mat_contents["uval_sinesweep"]
else:
    raise ValueError("test_type must be either 'multisine' or 'sinesweep'")


# output data
if test_type == "multisine":
    data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/yval_multisine.mat")
    mat_contents = loadmat(data_file_path, squeeze_me=True)
    yTest = mat_contents["yval_multisine"]
elif test_type == "sinesweep":
    data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/yval_sinesweep.mat")
    mat_contents = loadmat(data_file_path, squeeze_me=True)
    yTest = mat_contents["yval_sinesweep"]
else:
    raise ValueError("test_type must be either 'multisine' or 'sinesweep'")


test_data = deepSI.System_data(u=uTest, y=yTest)

## ------------- Load fit system -----------------
# fit_sys_file_name_list = ["boucWen-FP_linear-dynamic_parallel",
#                           "boucWen-FP_linear-static_parallel",
#                           "boucWen-FP_nonlinear-dynamic_parallel_e3000",
#                           "boucWen-FP_nonlinear-static_parallel",
#                           "boucWen-BLA_2-dynamic_parallel_e3000",
#                           "boucWen-BLA_2-static_parallel",
#                           "boucWen-BLA_3-parallel_e3000",
#                           "boucWen-ANN_SS_e3700"]

# paper
fit_sys_file_name_list = ["boucWen-FP_linear-dynamic_parallel_e4000",
                        #   "boucWen-FP_linear-static_parallel",
                          "boucWen-FP_nonlinear-dynamic_parallel_e4000",
                        #   "boucWen-FP_nonlinear-static_parallel",
                          "boucWen-BLA_3-parallel_e4000",
                          "boucWen-ANN_SS_e3700"]

# fit_sys_file_name_list = ["boucWen-FP_linear-dynamic_parallel_e4000"]

# loss plots
# paper
# fit_sys_file_name_list = ["boucWen-FP_linear-dynamic_parallel_e4000",
#                           "boucWen-FP_linear-static_parallel",
#                           "boucWen-ANN_SS_e3700"]



fit_sys_list = []
fit_sys_name_list = []
for fit_sys_file_name in fit_sys_file_name_list:
    fit_sys_file_path = os.path.join(os.getcwd(), "models", "bouc_wen", fit_sys_file_name)
    fit_sys_list.append(deepSI.load_system(fit_sys_file_path))
    fit_sys_name_list.append(fit_sys_file_name.split("Wen-")[1])

# for fit_sys in fit_sys_list:
#     # print(fit_sys.hfn.nx)
#     for m in fit_sys_list[0].hfn.connected_blocks:
#         if isinstance(m, Parameterized_MSD_State_Block):
#             print(m.params)
#             print(m.init_params)
#         if isinstance(m, Parameterized_Linear_State_Block):
#             print(m.A)
#             print(m.A_init)
#             print(m.B)
#             print(m.B_init)
#             # break
#         if isinstance(m, Parameterized_Linear_Output_Block):
#             print(m.C)
#             print(m.C_init)
#             # break
#         if isinstance(m, Cubic_Block):
#             print(m.aL)
#             # break

## ------------- RMSE scores -----------------
test_list = []
for fit_sys in fit_sys_list:
    test_list.append(fit_sys.apply_experiment(test_data))

for i, test in enumerate(test_list):
    print(f'{fit_sys_name_list[i]} RMS {test.RMS(test_data)}; NRMS {test.NRMS(test_data)*100}')

## ------------- Plot prediction error -----------------
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
scaling = 6/10; fig1 = plt.figure(figsize=[9.2*scaling, 3*scaling])

plt.plot(test_data.y[:],label="Measured data")
plt.plot(test_data.y[:] - test_list[0].y[:], label=fit_sys_name_list[0])

plt.grid()
plt.ylabel(r"$y$ [m]", labelpad=0.0)
plt.xlim([0,test_data.N_samples])
plt.legend()
plt.xlabel(r"$k$", labelpad=0.0)
plt.tight_layout()

# fig_file_path = os.path.join(os.getcwd(), "figures\\bouc_wen\\")
# plt.savefig(os.path.join(fig_file_path, "fitted_models_pred_error.svg"), transparent=True, format="svg")
# plt.savefig(os.path.join(fig_file_path, "fitted_models_pred_error.png"), transparent=True, format="png")
# plt.savefig(os.path.join(fig_file_path, "fitted_models_pred_error.eps"), transparent=True, format="eps")

plt.show()

## ------------- Plot val loss -----------------
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
scaling = 6/10; fig3 =  plt.figure(figsize=[8.9*scaling, 3.5*scaling])
n_epochs_plt = 1000

for fit_sys, name in zip(fit_sys_list, fit_sys_name_list):
    fit_sys.checkpoint_load_system('_last')
    plt.semilogy(fit_sys.epoch_id,fit_sys.Loss_val, label=name) 

plt.ylabel(r"RMSE")
plt.xlabel(r"epochs")
plt.xlim([0,n_epochs_plt])
# plt.ylim([8e-6,10e-3])
plt.grid()
plt.legend()
plt.tight_layout()

fig_file_path = os.path.join(os.getcwd(), "figures\\bouc_wen\\")
plt.savefig(os.path.join(fig_file_path, "val_loss_plot_file_name.svg"), format="svg")
plt.savefig(os.path.join(fig_file_path, "val_loss_plot_file_name.png"), format="png")
plt.savefig(os.path.join(fig_file_path, "val_loss_plot_file_name.eps"), format="eps")

plt.show()