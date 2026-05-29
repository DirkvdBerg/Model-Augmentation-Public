from deepSI.utils.torch_nets import simple_res_net
from deepSI.fit_systems.encoders import SS_encoder_general_hf
import deepSI

import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from model_augmentation.utils.utils import *
from model_augmentation.fit_systems.interconnect import *
from model_augmentation.fit_systems.blocks import *
from model_augmentation.systems.mass_spring_damper import *

## ------------- Hyper params -----------------
# system parameters
system_dof = 3 # 2 or 3 (number of dof in system)
flag_linear_msd = False # True or False (use linear mass spring damper system)
flag_low_pass_filter = "None" # "Input", "Output" or "None" (use low pass filter in system)
flag_input_saturation = False # True or False (use input saturation in system)

# FP (baseline) model parameters
FP_dof = 2 # 2 or 3 (number of dof in FP model)
FP_type = "ideal" # "ideal" or "approximate"

# model augmentation parameters
state_aug_type = "parallel" # "parallel", "series" or "None"
series_type = "in" # "in" or "out"
dynamic_state_aug = True # True or False
additional_dynamic_aug_states = 2 # number of additional states for dynamic augmentation
output_aug = False # True or False
input_aug = False # True or False
# more specific augmentation types
state_augment_specific_states = False
if dynamic_state_aug:
    state_aug_state_indices_output = np.array([1,3,4,5]); nx_aug_model_out = 4
else:
    state_aug_state_indices_output = np.array([1,3]); nx_aug_model_out = 2
parallel_augment_u_in_z = True

# training parameters
epochs = 3000

# flags
save_flag = True # True or False (dont save model if False, e.g. for debugging)

## ------------- Load data -----------------
ny = 1; nu = 1 # other values are not supported yet

system_descriptors = ["msd", "{0}dof".format(system_dof)]
system_descriptors.append("linear" if flag_linear_msd else "nonlinear")
if flag_low_pass_filter == "Input": system_descriptors.append("lpf_input")
elif flag_low_pass_filter == "Output": system_descriptors.append("lpf_output")
if flag_input_saturation: system_descriptors.append("input_saturation")
system_descriptors.append("multisine")
system_folder_name = "_".join(system_descriptors)

data_file_path = os.path.join(os.getcwd(), "data", "journal_model_augmentation", system_folder_name)
test_data = deepSI.load_system_data(os.path.join(os.getcwd(), data_file_path, "test.npz"))

## ------------- Add noise -----------------
sigma_n = 52e-4
test_data.y = test_data.y + np.random.normal(0, sigma_n, test_data.y.shape)

## ------------- Load fit system -----------------
baseline_model_descriptors = ["FP", "{0}dof".format(FP_dof)]
baseline_model_descriptors.append("ideal" if FP_type == "ideal" else "approximate")
baseline_model_folder_name = "_".join(baseline_model_descriptors)

model_augmentation_descriptors = []
if state_aug_type != "None":
    if dynamic_state_aug:
        model_augmentation_descriptors.append("dynamic")
    else:
        model_augmentation_descriptors.append("static")
    if state_augment_specific_states:
        model_augmentation_descriptors.append("specific")
    if not parallel_augment_u_in_z:
            model_augmentation_descriptors.append("only_x_as_z")
    model_augmentation_descriptors.append(state_aug_type)
    if state_aug_type == "series":
            model_augmentation_descriptors.append(series_type)

if output_aug:
        model_augmentation_descriptors.append("dynamic_output")
if input_aug:
    model_augmentation_descriptors.append("static_input")
model_augmentation_descriptors.append("e{0}".format(epochs))  
model_augmentation_file_name = "_".join(model_augmentation_descriptors)
print(model_augmentation_file_name)

interconnect_file_path = os.path.join(os.getcwd(), "models", "journal_model_augmentation", system_folder_name, baseline_model_folder_name, model_augmentation_file_name)

interconnect : SSE_Interconnect = deepSI.load_system(interconnect_file_path)

# determine number of model states
if dynamic_state_aug:
    nx_model = 2*FP_dof + additional_dynamic_aug_states
else:
    nx_model = 2*FP_dof
if output_aug:
    nx_model = nx_model + 1

## ------------- print baseline model parameters -----------------
for m in interconnect.hfn.connected_blocks:
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

## ------------- Plot state component comparison -----------------
interconnect.hfn.reset_saved_signals()
test_interconnection = interconnect.apply_experiment(test_data)
saved_input_signals = interconnect.hfn.saved_input_signals
saved_output_signals = interconnect.hfn.saved_output_signals

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
scaling = 6/10
fig1 =  plt.figure(figsize=[9.2*scaling, 8.2*scaling])
plt_range = 400
plt_centre = 200

if state_augment_specific_states: aug_state_ix_counter = 0
for i in range(nx_model):
    plt.subplot(nx_model,1,i+1)
    plt.plot(saved_output_signals[i, plt_centre - plt_range //2 : plt_centre + plt_range //2 ], label="state")
    if output_aug:
        if not state_augment_specific_states:
            plt.plot(saved_input_signals[-nx_model+i-1, plt_centre - plt_range //2 : plt_centre + plt_range //2 ], label="augmentation")
        else:
            if i in state_aug_state_indices_output or i == nx_model-1:
                plt.plot(saved_input_signals[-nx_aug_model_out+aug_state_ix_counter-2, plt_centre - plt_range //2 : plt_centre + plt_range //2 ], label="augmentation")
                aug_state_ix_counter = aug_state_ix_counter + 1
    else:
        plt.plot(saved_input_signals[-nx_model+i, plt_centre - plt_range //2 : plt_centre + plt_range //2 ], label="augmentation")

    plt.ylabel(r"$x_{0}$".format(i+1))
    plt.xlim([0,plt_range])
    plt.grid()
    plt.xticks([100, 200, 300, 400], [])

plt.xticks([100, 200, 300, 400],[100, 200, 300, 400])
plt.xlabel(r"$k$")
# plt.tight_layout()

if save_flag:
    model_folder_path = os.path.join(os.getcwd(), "figures", "journal_model_augmentation", system_folder_name, baseline_model_folder_name, model_augmentation_file_name)
    os.makedirs(model_folder_path, exist_ok=True)

    fig_file_name = "state_component_comparison"
    plt.savefig(os.path.join(model_folder_path, fig_file_name + ".svg"), format="svg")
    plt.savefig(os.path.join(model_folder_path, fig_file_name + ".png"), format="png")
    plt.savefig(os.path.join(model_folder_path, fig_file_name + ".eps"), format="eps")
plt.show()