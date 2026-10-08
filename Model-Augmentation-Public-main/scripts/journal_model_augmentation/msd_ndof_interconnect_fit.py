import os
import time

import deepSI
import numpy as np
import torch
from scipy.io import loadmat

from model_augmentation.fit_systems.blocks import (
    Linear_Output_Block,
    Linear_State_Block,
    Parameterized_Linear_Output_Block,
    Parameterized_Linear_State_Block,
    Parameterized_MSD_State_Block,
    Static_ANN_Block,
)
from model_augmentation.fit_systems.interconnect import Interconnect, SSE_Interconnect
from model_augmentation.systems.mass_spring_damper import *
from model_augmentation.utils.torch_nets import (
    identity_init_simple_res_net,
    linear_mapping,
    zero_init_feed_forward_nn,
    zero_init_linear_mapping,
)
from model_augmentation.utils.utils import (
    expansion_matrix,
    normalize_linear_ss_matrices,
    selection_matrix,
)

## ------------- Hyper params -----------------
# system parameters
system_dof = 3  # 2 or 3 (number of dof in system)
flag_linear_msd = False  # True or False (use linear mass spring damper system)
flag_low_pass_filter = (
    "None"  # "Input", "Output" or "None" (use low pass filter in system)
)
flag_input_saturation = False  # True or False (use input saturation in system)

# FP (baseline) model parameters
FP_dof = 2  # 2 or 3 (number of dof in FP model)
FP_type = "ideal"  # "ideal" or "approximate"

# model augmentation parameters
state_aug_type = "parallel"  # "parallel", "series" or "None"
series_type = "out"  # "in" or "out"
dynamic_state_aug = True  # True or False
additional_dynamic_aug_states = (
    2  # number of additional states for dynamic augmentation
)
output_aug = False  # True or False
input_aug = False  # True or False
# more specific augmentation types
state_augment_specific_states = False
if dynamic_state_aug:
    state_aug_state_indices_output = np.array([1, 3, 4, 5])
    nx_aug_model_out = 4
else:
    state_aug_state_indices_output = np.array([1, 3])
    nx_aug_model_out = 2
parallel_augment_u_in_z = True


# training parameters
nf = 200
epochs = 3002
batch_size = 2000
# utility parameters
save_flag = True  # True or False (dont save model if False, e.g. for debugging)
wait_minutes = (
    0  # minutes to wait before starting the training (e.g. to not annoy colleagues)
)

## ------------- Invalid user inputs -----------------
if state_aug_type == "None" and dynamic_state_aug:
    raise ValueError("type_aug 'None' cannot be dynamic augmentation")
if input_aug and state_aug_type == "series":
    raise ValueError(
        "Both input_aug and series aug work on the inputs of the baseline model"
    )

## ------------- Load data -----------------
ny = 1
nu = 1  # other values are not supported yet

# This has been modified for new simulations
system_descriptors = ["msd", "{0}dof".format(system_dof)]
system_descriptors.append("linear" if flag_linear_msd else "nonlinear")
if flag_low_pass_filter == "Input":
    system_descriptors.append("lpf_input")
elif flag_low_pass_filter == "Output":
    system_descriptors.append("lpf_output")
if flag_input_saturation:
    system_descriptors.append("input_saturation")
system_descriptors.append("multisine")
system_folder_name = "_".join(system_descriptors)

data_file_path = os.path.join(
    os.getcwd(), "data", "journal_model_augmentation", system_folder_name
)
train_data = deepSI.load_system_data(os.path.join(data_file_path, "train.npz"))
val_data = deepSI.load_system_data(os.path.join(data_file_path, "val.npz"))

## ------------- Add noise -----------------
sigma_n = 52e-4  # SNR60:15e-5; SNR30:52e-4; SNR20:15e-3
train_data.y = train_data.y + np.random.normal(0, sigma_n, train_data.y.shape)  # type: ignore
val_data.y = val_data.y + np.random.normal(0, sigma_n, val_data.y.shape)  # type: ignore

## ------------- Load FP model -----------------
nx_FP = 2 * FP_dof  # number of states in FP model (2dof: 4; 3dof: 6)
FP_state_indices = np.arange(
    0, nx_FP
)  # state indices for the FP model (2dof: [0,1,2,3]; 3dof: [0,1,2,3,4,5])

if FP_type == "ideal":
    data_file_path = os.path.join(
        os.getcwd(), "data", "mass_spring_damper", "msd_{0}dof.mat".format(FP_dof)
    )
elif FP_type == "approximate":
    data_file_path = os.path.join(
        os.getcwd(),
        "data",
        "mass_spring_damper",
        "msd_{0}dof_non_ideal.mat".format(FP_dof),
    )
else:
    raise ValueError("FP_type must be either 'ideal' or 'approximate'")
mat_contents = loadmat(data_file_path, squeeze_me=False)

A_bla = mat_contents["Ad"]
B_bla = mat_contents["Bd"]
C_bla = mat_contents["Cd"]
D_bla = mat_contents["Dd"]
A_bar_bla, B_bar_bla, C_bar_bla, D_bar_bla = normalize_linear_ss_matrices(
    A_bla,
    B_bla,
    C_bla,
    D_bla,
    train_data,  # type: ignore
    state_ix=FP_state_indices,
)

## ------------- Define augmentations structure -----------------
# determine number of model states
if dynamic_state_aug:
    nx_model = nx_aug_model = nx_FP + additional_dynamic_aug_states
else:
    nx_model = nx_aug_model = nx_FP
state_aug_state_indices = np.arange(
    0, nx_aug_model
)  # state indices for the state augmentation component model (2dof: [0,1,2,3]; 3dof: [0,1,2,3,4,5])
if not state_augment_specific_states:
    nx_aug_model_out = nx_aug_model
    state_aug_state_indices_output = state_aug_state_indices
if output_aug:
    nx_model = nx_model + 1
    output_aug_state_indices = np.array([nx_model - 1])

# define interconnection and baseline model blocks
interconnect = Interconnect(nx_model, nu, ny, debugging=False)
if FP_type == "ideal":
    physical_state_model_block = Linear_State_Block(A=A_bar_bla, B=B_bar_bla)
    physical_output_model_block = Linear_Output_Block(C=C_bar_bla, D=D_bar_bla)
elif FP_type == "approximate":
    physical_state_model_block = Parameterized_MSD_State_Block(
        nz=5, nw=4, FP_type=FP_type
    )
    physical_output_model_block = Linear_Output_Block(C=C_bar_bla, D=D_bar_bla)
else:
    raise ValueError("FP_type must be either 'ideal' or 'approximate'")
interconnect.add_block(physical_state_model_block)
interconnect.add_block(physical_output_model_block)

# ----- (dynamic) parallel -------
if state_aug_type == "parallel":  # works for both static and dynamic augmentation
    if parallel_augment_u_in_z:
        ANN_state_block = Static_ANN_Block(
            nz=nx_aug_model + nu,
            nw=nx_aug_model_out,
            n_nodes_per_layer=8,
            net=zero_init_linear_mapping,
            activation=torch.nn.Tanh,
        )

    else:
        ANN_state_block = Static_ANN_Block(
            nz=nx_aug_model,
            nw=nx_aug_model_out,
            n_nodes_per_layer=8,
            net=zero_init_linear_mapping,
            activation=torch.nn.Tanh,
        )
    interconnect.add_block(ANN_state_block)

    interconnect.connect_signals(
        "x",
        ANN_state_block,
        "concat",
        selection_matrix(state_aug_state_indices, nx_model),
    )
    if parallel_augment_u_in_z:
        interconnect.connect_block_signals(ANN_state_block, ["u"], [])
    interconnect.connect_signals(
        ANN_state_block,
        "xp",
        "additive",
        expansion_matrix(state_aug_state_indices_output, nx_model),
    )

    interconnect.connect_signals(
        "x",
        physical_state_model_block,
        "concat",
        selection_matrix(FP_state_indices, nx_model),
    )
    interconnect.connect_signals(
        physical_state_model_block,
        "xp",
        "additive",
        expansion_matrix(FP_state_indices, nx_model),
    )

    interconnect.connect_signals(
        "x",
        physical_output_model_block,
        "concat",
        selection_matrix(FP_state_indices, nx_model),
    )

# ----- (dynamic) series out -------
elif state_aug_type == "series":
    if series_type == "out":  # works for both static and dynamic augmentation
        ANN_state_block = Static_ANN_Block(
            nz=nx_aug_model + 2 * FP_dof + nu,
            nw=nx_aug_model,
            n_nodes_per_layer=8,
            net=identity_init_simple_res_net,
            activation=torch.nn.Tanh,
        )
        interconnect.add_block(ANN_state_block)

        interconnect.connect_signals(
            "x",
            ANN_state_block,
            "concat",
            selection_matrix(state_aug_state_indices, nx_model),
        )
        interconnect.connect_block_signals(
            ANN_state_block, [physical_state_model_block, "u"], []
        )
        interconnect.connect_signals(
            ANN_state_block,
            "xp",
            "additive",
            expansion_matrix(state_aug_state_indices, nx_model),
        )

        interconnect.connect_signals(
            "x",
            physical_state_model_block,
            "concat",
            selection_matrix(FP_state_indices, nx_model),
        )

        interconnect.connect_signals(
            "x",
            physical_output_model_block,
            "concat",
            selection_matrix(FP_state_indices, nx_model),
        )

    # ----- (dynamic) series in -------
    elif series_type == "in":  # works for both static and dynamic augmentation
        ANN_state_block = Static_ANN_Block(
            nz=nx_aug_model + nu,
            nw=nx_aug_model + nu,
            n_nodes_per_layer=8,
            net=identity_init_simple_res_net,
            activation=torch.nn.Tanh,
        )
        interconnect.add_block(ANN_state_block)

        interconnect.connect_signals(
            "x",
            ANN_state_block,
            "concat",
            selection_matrix(state_aug_state_indices, nx_model),
        )
        interconnect.connect_block_signals(ANN_state_block, ["u"], [])
        interconnect.connect_signals(
            ANN_state_block,
            physical_state_model_block,
            "concat",
            selection_matrix(np.arange(0, nx_FP + 1), ANN_state_block.nw),
        )
        if dynamic_state_aug:
            interconnect.connect_signals(
                ANN_state_block,
                "xp",
                "additive",
                torch.matmul(
                    expansion_matrix(np.arange(nx_FP, nx_FP + 2), nx_model),
                    selection_matrix(
                        np.arange(nx_FP + 1, nx_FP + 3), ANN_state_block.nw
                    ),
                ),
            )

        interconnect.connect_signals(
            physical_state_model_block,
            "xp",
            "additive",
            expansion_matrix(FP_state_indices, nx_model),
        )

        interconnect.connect_signals(
            "x",
            physical_output_model_block,
            "concat",
            selection_matrix(FP_state_indices, nx_model),
        )

    else:
        raise ValueError("series_type must be either 'in' or 'out'")

# ----- No augmentation -------
elif state_aug_type == "None":
    interconnect.connect_signals(
        "x",
        physical_state_model_block,
        "concat",
        selection_matrix(FP_state_indices, nx_model),
    )
    interconnect.connect_signals(
        physical_state_model_block,
        "xp",
        "additive",
        expansion_matrix(FP_state_indices, nx_model),
    )

    interconnect.connect_signals(
        "x",
        physical_output_model_block,
        "concat",
        selection_matrix(FP_state_indices, nx_model),
    )
else:
    raise ValueError("type_aug must be either 'parallel', 'series' or 'None'")

# ----- input -------
if input_aug:
    # ----- static input augmentation -------
    ANN_input_block = Static_ANN_Block(
        nz=nu,
        nw=nu,
        n_nodes_per_layer=4,
        net=identity_init_simple_res_net,
        activation=torch.nn.Tanh,
    )
    interconnect.add_block(ANN_input_block)

    interconnect.connect_block_signals(
        ANN_input_block, ["u"], [physical_state_model_block]
    )
elif state_aug_type == "series" and series_type == "in":
    # This is handled in the series in augmentation above
    pass
else:
    # ----- No input augmentation -------
    interconnect.connect_block_signals(physical_state_model_block, ["u"], [])

# ----- output -------
if output_aug:
    # ----- dynamic output augmentation -------
    lpf_state_model_block = Parameterized_Linear_State_Block(
        A=torch.Tensor([[1.0]]), B=torch.Tensor([[0.0]]), flag_loss_reg=False
    )
    lpf_output_model_block = Parameterized_Linear_Output_Block(
        C=torch.Tensor([[0.0]]), D=torch.Tensor([[1.0]]), flag_loss_reg=False
    )
    interconnect.add_block(lpf_state_model_block)
    interconnect.add_block(lpf_output_model_block)

    interconnect.connect_block_signals(physical_output_model_block, ["u"], [])

    interconnect.connect_signals(
        "x",
        lpf_state_model_block,
        "concat",
        selection_matrix(output_aug_state_indices, nx_model),
    )
    interconnect.connect_block_signals(
        lpf_state_model_block, [physical_output_model_block], []
    )
    interconnect.connect_signals(
        lpf_state_model_block,
        "xp",
        "additive",
        expansion_matrix(output_aug_state_indices, nx_model),
    )

    interconnect.connect_signals(
        "x",
        lpf_output_model_block,
        "concat",
        selection_matrix(output_aug_state_indices, nx_model),
    )
    interconnect.connect_block_signals(
        lpf_output_model_block, [physical_output_model_block], []
    )
    interconnect.connect_block_signals(lpf_output_model_block, [], ["y"])
else:
    # ----- No output augmentation -------
    interconnect.connect_block_signals(physical_output_model_block, ["u"], ["y"])


## ------------- Wait Time To Not Annoy Colleagues -----------------
for t in range(wait_minutes):
    time.sleep(60)
    print(f"Time passed: {t + 1} minutes")

## ------------- Train fit system -----------------
fit_sys = SSE_Interconnect(
    interconnect=interconnect,  # type: ignore
    na=nx_model * 2 + 1,
    nb=nx_model * 2 + 1,
    e_net_kwargs={"n_nodes_per_layer": 16},
)
fit_sys.fit(
    train_sys_data=train_data,
    val_sys_data=val_data,
    batch_size=batch_size,
    epochs=epochs,
    auto_fit_norm=True,
    loss_kwargs={"nf": nf},
    validation_measure="sim-RMS",
)

## ------------- Save fit system -----------------
if save_flag:
    baseline_model_descriptors = ["FP", "{0}dof".format(FP_dof)]
    baseline_model_descriptors.append("ideal" if FP_type == "ideal" else "approximate")
    baseline_model_folder_name = "_".join(baseline_model_descriptors)

    model_augmentation_descriptors = []
    if state_aug_type != "None":
        if dynamic_state_aug:
            model_augmentation_descriptors.append("dynamic_linear")
        else:
            model_augmentation_descriptors.append("static_linear")
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

    model_folder_path = os.path.join(
        os.getcwd(),
        "models",
        "journal_model_augmentation",
        system_folder_name,
        baseline_model_folder_name,
    )
    os.makedirs(model_folder_path, exist_ok=True)
    fit_sys.save_system(os.path.join(model_folder_path, model_augmentation_file_name))
