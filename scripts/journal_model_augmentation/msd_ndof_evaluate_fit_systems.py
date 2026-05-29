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

root_dir = os.getcwd()
expected_folder = "Model Augmentation"
while True:
    if os.path.basename(root_dir) == expected_folder:
        break  # found it!
    parent = os.path.dirname(root_dir)
    if parent == root_dir:
        raise FileNotFoundError(
            f"Folder '{expected_folder}' not found above {os.getcwd()}"
        )
    root_dir = parent

## ------------- User Input -----------------
# system parameters
system_dof = 3; nxd = 2*system_dof
FP_dof = 2
FP_type = "ideal" # "ideal" or "approximate"
system_folder_name = "msd_3dof_nonlinear_input_saturation_multisine"

# flags
save_flag = False # True or False (dont save model if False, e.g. for debugging)
flag_select_models = True
flag_print_parameters = True
flag_select_epoch_count = False; n_epochs_flag = 3000

## ------------- Load data -----------------
data_file_path = os.path.join(root_dir, "data", "journal_model_augmentation", system_folder_name)
test_data = deepSI.load_system_data(os.path.join(root_dir, data_file_path, "test.npz"))

## ------------- Add noise -----------------
sigma_n = 52e-4
test_data.y = test_data.y + np.random.normal(0, sigma_n, test_data.y.shape)

## ------------- Load fit system -----------------
baseline_model_descriptors = ["FP", "{0}dof".format(FP_dof)]
baseline_model_descriptors.append("ideal" if FP_type == "ideal" else "approximate")
baseline_model_folder_name = "_".join(baseline_model_descriptors)

model_folder_path = os.path.join(root_dir, "models", "journal_model_augmentation", system_folder_name, baseline_model_folder_name)
if flag_select_models:
    # fit_sys_file_name_list = [
    #                          "dynamic_parallel_e3000",
    #                          "static_parallel_e3000",
    #                          "dynamic_series_out_e3000",
    #                          "static_series_out_e3000",
    #                          "dynamic_linear_parallel_e3000",

                            #   "resnet_hf__e15000",
                            #   "resnet_hf_7states_e15000"
                            #   ]
    # fit_sys_file_name_list = ["dynamic_parallel_static_input_e1600",
    #                           "dynamic_parallel_e3000",
    #                           "resnet_hf_e3000"]
    fit_sys_file_name_list = [
                                "static_parallel_static_input_e3000",
                                "dynamic_parallel_static_input_e3000"
                                ]
else:
    fit_sys_file_name_list = os.listdir(model_folder_path)


fit_sys_list = []
fit_sys_name_list = []
if not flag_select_epoch_count: max_n_epochs = 0
for fit_sys_file_name in fit_sys_file_name_list:
    fit_sys_file_path = os.path.join(model_folder_path, fit_sys_file_name)

    if flag_select_epoch_count:
        if int(fit_sys_file_name.split("_e", 1)[1]) == n_epochs_flag:
            fit_sys_list.append(deepSI.load_system(fit_sys_file_path))
            fit_sys_name_list.append(fit_sys_file_name.split("_e", 1)[0])
    else:
        fit_sys_list.append(deepSI.load_system(fit_sys_file_path))
        fit_sys_name_list.append(fit_sys_file_name.split("_e", 1)[0])
        if int(fit_sys_file_name.split("_e", 1)[1]) > max_n_epochs:
            max_n_epochs = int(fit_sys_file_name.split("_e", 1)[1])

## ------------- print baseline model parameters -----------------
if flag_print_parameters:
    for fit_sys in fit_sys_list:
        if isinstance(fit_sys, SSE_Interconnect):
            print(fit_sys.hfn.nx)
            for m in fit_sys.hfn.connected_blocks:
                if isinstance(m, Parameterized_Linear_State_Block):
                    print(m.A)
                    print(m.A_init)
                    print(m.B)
                    print(m.B_init)
                if isinstance(m, Parameterized_Linear_Output_Block):
                    print(m.C)
                    print(m.C_init)
                    print(m.D)
                    print(m.D_init)
                if isinstance(m, Parameterized_MSD_State_Block):
                    print(m.params)

## ------------- RMSE scores -----------------
test_list = []
for fit_sys in fit_sys_list:
    test_list.append(fit_sys.apply_experiment(test_data))

for i, test in enumerate(test_list):
    print(f'{fit_sys_name_list[i]} RMS {test.RMS(test_data)}; NRMS {test.NRMS(test_data)*100}')

# ## ------------- Plot prediction error -----------------
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
scaling = 6/10; fig1 = plt.figure(figsize=[9.2*scaling, 3*scaling])
# scaling = 2.5/10; fig1 = plt.figure(figsize=[30*scaling, 13*scaling], dpi=400)

plt.plot(test_data.y[:],label="Measured data")
plt.plot(test_data.y[:] - test_list[0].y[:])

plt.grid()
plt.ylabel(r"$y$ [m]".format(i+1), labelpad=0.0)
plt.xlim([0,test_data.N_samples])
# plt.legend(loc="lower center", prop = { "size": 8.3}, ncols=2, columnspacing=0.5)
plt.xlabel(r"$k$", labelpad=0.0)
plt.tight_layout()

if save_flag:
    model_folder_path = os.path.join(root_dir, "figures", "journal_model_augmentation", system_folder_name, baseline_model_folder_name)
    os.makedirs(model_folder_path, exist_ok=True)

    fig_file_name = "fitted_models_pred_error"
    plt.savefig(os.path.join(model_folder_path, fig_file_name + ".svg"), format="svg")
    plt.savefig(os.path.join(model_folder_path, fig_file_name + ".png"), format="png")
    plt.savefig(os.path.join(model_folder_path, fig_file_name + ".eps"), format="eps")
plt.show()


## ------------- Plot val loss -----------------
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
scaling = 6/10; fig3 =  plt.figure(figsize=[8.9*scaling, 3.5*scaling])
if flag_select_epoch_count:
    n_epochs_plt = 3000
else:
    n_epochs_plt = max_n_epochs
# n_epochs_plt = 1000

for fit_sys, name in zip(fit_sys_list, fit_sys_name_list):
    fit_sys.checkpoint_load_system('_last')
    plt.semilogy(fit_sys.epoch_id,fit_sys.Loss_val, label=name) 

plt.plot(np.arange(n_epochs_plt), np.ones(n_epochs_plt)*sigma_n, "r--", label="noise floor")
plt.ylabel(r"RMSE")
plt.xlabel(r"epochs")
plt.xlim([0,n_epochs_plt])
plt.ylim([sigma_n*0.8,12e-2])
# plt.legend()
plt.grid()
plt.tight_layout()

if save_flag:
    model_folder_path = os.path.join(root_dir, "figures", "journal_model_augmentation", system_folder_name, baseline_model_folder_name)
    os.makedirs(model_folder_path, exist_ok=True)

    fig_file_name = "val_loss_plot_file_name"
    plt.savefig(os.path.join(model_folder_path, fig_file_name + ".svg"), format="svg")
    plt.savefig(os.path.join(model_folder_path, fig_file_name + ".png"), format="png")
    plt.savefig(os.path.join(model_folder_path, fig_file_name + ".eps"), format="eps")
plt.show()
