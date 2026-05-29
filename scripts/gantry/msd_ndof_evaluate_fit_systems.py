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

## ------------- Load data -----------------
dof = 3; nxd = 2*dof
SNR = 30 # 20, 30, 60

data_file_path = "data/mass_spring_damper"
test_data = deepSI.load_system_data(os.path.join(os.getcwd(), data_file_path, "msd_3dof_multisine_test.npz"))
# test_data = deepSI.load_system_data(os.path.join(os.getcwd(), data_file_path, "msd_3dof_multisine_extrapolate.npz"))

## ------------- Add noise -----------------
if SNR == 20:
    sigma_n = 15e-3 # SNR20:15e-3
elif SNR == 30:
    sigma_n = 52e-4
elif SNR == 60:
    sigma_n = 15e-5
else:
    raise ValueError("SNR must be either 20, 30 or 60")
test_data.y = test_data.y + np.random.normal(0, sigma_n, test_data.y.shape)

## ------------- Load fit system -----------------
# all ideal models
# FP_type = "ideal"
# fit_sys_file_name_list = ["msd_3dof_dynamic_parallel_e3000",
#                           "msd_3dof_linear_dynamic_parallel_e3000",
#                           "msd_3dof_static_parallel_e3000",
#                           "msd_3dof_dynamic_series_e3000",
#                           "msd_3dof_static_series_e3000",
#                           "msd_3dof_ANN_SS_e10000"]

# all approximate models
# FP_type = "approximate"
# fit_sys_file_name_list = ["msd_3dof_dynamic_parallel_e3000",
#                         #   "msd_3dof_linear_dynamic_parallel_e3000",
#                           "msd_3dof_static_parallel_e3000",
#                           "msd_3dof_dynamic_series_e3000",
#                           "msd_3dof_static_series_e3000"]

# loss epochs plot
FP_type = "ideal"
fit_sys_file_name_list = ["msd_3dof_dynamic_parallel_e3000",
                          "msd_3dof_static_parallel_e3000",
                          "msd_3dof_linear_dynamic_parallel_e1000",
                          "msd_3dof_ANN_SS_e10000"]

fit_sys_list = []
fit_sys_name_list = []
for fit_sys_file_name in fit_sys_file_name_list:
    if FP_type == "ideal":
        fit_sys_file_path = os.path.join(os.getcwd(), "models", "ecc_corrected", "ideal", "SNR{0}".format(SNR), fit_sys_file_name)
    elif FP_type == "approximate":
        fit_sys_file_path = os.path.join(os.getcwd(), "models", "ecc_corrected", "approximate", "SNR{0}".format(SNR), fit_sys_file_name)
    fit_sys_list.append(deepSI.load_system(fit_sys_file_path))
    fit_sys_name_list.append(fit_sys_file_name.split("_e", 1)[0].split("dof_", 1)[1])

for fit_sys in fit_sys_list:
    print(fit_sys.hfn.nx)
    for m in fit_sys_list[0].hfn.connected_blocks:
        if isinstance(m, Parameterized_MSD_State_Block):
            print(m.params)
            print(m.init_params)
            print(m.epsilon)
            break

## ------------- Load 2dof -----------------
FP_dof = 2
data_file_path = os.path.join(os.getcwd(), "data", "mass_spring_damper", "msd_{0}dof.mat".format(FP_dof))
# data_file_path = os.path.join(os.getcwd(), "data", "mass_spring_damper", "msd_{0}dof_non_ideal.mat".format(FP_dof))

mat_contents = loadmat(data_file_path, squeeze_me=False)

nx = mat_contents['nx'][0,0]; ny = mat_contents['ny'][0,0]; nu = mat_contents['nu'][0,0]
Ts = mat_contents['Ts'][0,0]

A_bla = mat_contents['Ad']
B_bla = mat_contents['Bd']
C_bla = mat_contents['Cd']
D_bla = mat_contents['Dd']

## ------------- Simulate 2dof -----------------
N = test_data.N_samples

ylog = np.zeros(N)
ulog = np.zeros(N)
x0 = np.zeros((4,1))
for i in range(N):
    u0 = test_data.u[i]
    ulog[i] = u0

    x1 = A_bla @ x0 + B_bla * u0
    ylog[i] = (C_bla @ x0)[0,0]

    x0 = x1.copy()

test_2dof = deepSI.System_data(u=ulog, y=ylog)

## ------------- RMSE scores -----------------
test_list = []
for fit_sys in fit_sys_list:
    test_list.append(fit_sys.apply_experiment(test_data))

print(f'Baseline RMS {test_2dof.RMS(test_data)}; NRMS {test_2dof.NRMS(test_data)*100}')
for i, test in enumerate(test_list):
    print(f'{fit_sys_name_list[i]} RMS {test.RMS(test_data)}; NRMS {test.NRMS(test_data)*100}')

# ## ------------- FP model params -----------------
# for i, fit_sys in enumerate(fit_sys_list):
#     for m in fit_sys.hfn.connected_blocks:
#         if isinstance(m, Parameterized_MSD_State_Block):
#             print(f'{fit_sys_name_list[i]} params: {m.params}')
#             break

# ## ------------- Plot prediction error -----------------
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
scaling = 6/10; fig1 = plt.figure(figsize=[9.2*scaling, 3*scaling])
# scaling = 2.5/10; fig1 = plt.figure(figsize=[30*scaling, 13*scaling], dpi=400)

plt.plot(test_data.y[:],label="Measured data")
# plt.plot(test_data.y[:] - test_2dof.y[:])
plt.plot(test_data.y[:] - test_list[0].y[:])

plt.grid()
plt.ylabel(r"$y$ [m]".format(i+1), labelpad=0.0)
plt.xlim([0,test_data.N_samples])
# plt.legend(loc="lower center", prop = { "size": 8.3}, ncols=2, columnspacing=0.5)
plt.xlabel(r"$k$", labelpad=0.0)
plt.tight_layout()

fig_file_path = os.path.join(os.getcwd(), "figures\\ecc_corrected\\")
plt.savefig(os.path.join(fig_file_path, "fitted_models_pred_error.svg"), transparent=True, format="svg")
plt.savefig(os.path.join(fig_file_path, "fitted_models_pred_error.png"), transparent=True, format="png")
plt.savefig(os.path.join(fig_file_path, "fitted_models_pred_error.eps"), transparent=True, format="eps")

plt.show()

## ------------- Plot val loss -----------------
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
scaling = 6/10; fig3 =  plt.figure(figsize=[8.9*scaling, 3.5*scaling])
n_epochs_plt = 500

for fit_sys, name in zip(fit_sys_list, fit_sys_name_list):
    fit_sys.checkpoint_load_system('_last')
    plt.semilogy(fit_sys.epoch_id,fit_sys.Loss_val, label=name) 

plt.plot(np.arange(n_epochs_plt), np.ones(n_epochs_plt)*sigma_n, "r--", label="noise floor")
plt.ylabel(r"RMSE")
plt.xlabel(r"epochs")
plt.xlim([0,n_epochs_plt])
# plt.legend()
plt.grid()
plt.tight_layout()

fig_file_path = os.path.join(os.getcwd(), "figures\\ecc_corrected\\")
plt.savefig(os.path.join(fig_file_path, "val_loss_plot_file_name.svg"), format="svg")
plt.savefig(os.path.join(fig_file_path, "val_loss_plot_file_name.png"), format="png")
plt.savefig(os.path.join(fig_file_path, "val_loss_plot_file_name.eps"), format="eps")

plt.show()