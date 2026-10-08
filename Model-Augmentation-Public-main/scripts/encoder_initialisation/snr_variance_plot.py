from deepSI import load_system_data, load_system

import torch
import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from model_augmentation.fit_systems.pre_encoder import (
    System_data_with_x,
    System_data_norm_with_x,
)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

## ------------- Determine root directory -----------------
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

## --- Plotting format --
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times New Roman"

## ------------- User Input -----------------
# select specific system
flag_select_system = True
# select specific FP model
flag_select_FP_model_type = True
# select specific encoder initialisation
flag_select_encoder_init_type = True
encoder_initialisation_types = [
    "pre_trained",
    "linear_map",
]  # "pre_trained", "linear_map", "none"
# select specific models
flag_select_augmentation_types = True
augmentation_types = ["baseline"]  # "static_parallel" or "baseline"

# filter model flags
flag_select_epoch_count = True
n_epochs_flag = 0
snr_levels = list(range(5, 51, 5)) # in dB

# select system parameters
system_dof = 2  # 2 or 3 (number of dof in system)
flag_linear_msd = True  # True or False (use linear mass spring damper system)

# select FP (baseline) model parameters
FP_dof = 2  # 2 or 3 (number of dof in FP model)
FP_linear = True  # True or False (use linear FP model)
FP_type = "ideal"  # "ideal" or "approximate"

# debug flags
save_flag = True  # True or False (dont save model if False, e.g. for debugging)

## ------------- Load data -----------------
nx = 4
std_list = [[] for i in range(nx)]

for snr_level in snr_levels:
    if flag_select_system:
        system_descriptors = ["msd", "{0}dof".format(system_dof)]
        system_descriptors.append("linear" if flag_linear_msd else "nonlinear")
        system_descriptors.append("multisine")
        system_folder_name = "_".join(system_descriptors)
    else:
        raise NotImplementedError("Only flag_select_system=True is implemented")

    data_file_path = os.path.join(
        root_dir,
        "data",
        "encoder_initialisation",
        system_folder_name,
    )
    true_test_data = load_system_data(os.path.join(root_dir, data_file_path, "test.npz"))

    data_file_path = os.path.join(
        root_dir,
        "data",
        "encoder_initialisation",
        system_folder_name,
        "SNR_levels",
        "SNR{0}".format(snr_level),
    )
    test_data = load_system_data(os.path.join(root_dir, data_file_path, "test.npz"))

    with open(os.path.join(root_dir, data_file_path, "parameters.json"), "r") as file:
        data_parameters = json.load(file)
    sigma_n = data_parameters["sigma_n"]

    std_x = np.std(true_test_data.x, axis=0)  # type: ignore
    test_data_with_x = System_data_with_x(
        y=test_data.y, u=test_data.u, x=true_test_data.x / std_x
    )

    ## ------------- Load fit system -----------------
    if flag_select_FP_model_type:
        baseline_model_descriptors = ["FP", "{0}dof".format(FP_dof)]
        baseline_model_descriptors.append("linear" if FP_linear else "nonlinear")
        baseline_model_descriptors.append("ideal" if FP_type == "ideal" else "approximate")
        baseline_model_folder_name = "_".join(baseline_model_descriptors)
    else:
        raise NotImplementedError("Only flag_select_FP_model_type=True is implemented")

    model_folder_path = os.path.join(
        root_dir,
        "models",
        "encoder_initialisation",
        system_folder_name,
        baseline_model_folder_name,
        "SNR_levels",
        "SNR{0}".format(snr_level),
    )

    # reorder encoder_initialisation_types to model path order
    encoder_initialisation_types = sorted(
        encoder_initialisation_types,
        key=lambda x: {
            item: index for index, item in enumerate(os.listdir(model_folder_path))
        }[x],
    )

    fit_sys_list = []
    fit_sys_name_list = []
    number_of_models_per_encoder_init_type = np.zeros((len(encoder_initialisation_types),))
    for encoder_initialisation_folder_name in os.listdir(model_folder_path):
        if (
            not flag_select_encoder_init_type
            or encoder_initialisation_folder_name in encoder_initialisation_types
        ):
            encoder_initialisation_folder_path = os.path.join(
                model_folder_path, encoder_initialisation_folder_name
            )
            for fit_sys_file_name in os.listdir(encoder_initialisation_folder_path):
                if (
                    not flag_select_augmentation_types
                    or fit_sys_file_name.split("_e", 1)[0] in augmentation_types
                ):
                    if (
                        not flag_select_epoch_count
                        or int(fit_sys_file_name.split("_e", 1)[1].split("_v", 1)[0])
                        == n_epochs_flag
                    ):
                        if flag_select_encoder_init_type:
                            number_of_models_per_encoder_init_type[
                                encoder_initialisation_types.index(
                                    encoder_initialisation_folder_name
                                )
                            ] += 1
                        fit_sys_file_path = os.path.join(
                            encoder_initialisation_folder_path, fit_sys_file_name
                        )
                        fit_sys_list.append(load_system(fit_sys_file_path))
                        fit_sys_name_list.append(
                            encoder_initialisation_folder_name
                            + "_"
                            + fit_sys_file_name.split("_e", 1)[0]
                            + "_v"
                            + fit_sys_file_name.split("_v", 1)[1]
                        )

    print(fit_sys_name_list)
    print(number_of_models_per_encoder_init_type)

    ## ------------- Run models -----------------
    temp_fit_sys = fit_sys_list[0]  # it should not matter which fit_sys is used here

    norm = System_data_norm_with_x(
        u0=temp_fit_sys.norm.u0,
        y0=temp_fit_sys.norm.y0,
        ystd=temp_fit_sys.norm.ystd,
        ustd=temp_fit_sys.norm.ustd,
    )
    normed_data = norm.transform(test_data_with_x)  # type: ignore
    up, yp, uf, yf, xf = temp_fit_sys.make_training_data(normed_data, **{"nf": 1})
    upt = torch.tensor(up, dtype=torch.float32)
    ypt = torch.tensor(yp, dtype=torch.float32)
    xft = torch.tensor(xf[:, 0, :], dtype=torch.float32)


    nx = xft.size()[1]
    sim_len = xft.size()[0]

    x_prediction_list = []
    error_list = []
    test_rms_list = []
    for fit_sys, name in zip(fit_sys_list, fit_sys_name_list):
        x_pred = fit_sys.encoder(upt, ypt)
        error = xft - x_pred

        x_prediction_list.append(x_pred.detach().numpy())
        e = error.detach().numpy()
        error_list.append(e)
        rmse = np.sqrt(np.mean(np.power(e, 2), axis=0))
        test_rms_list.append(rmse)

    # switches the 'none' initialization to be last in the list
    if "none" in encoder_initialisation_types:
        a = encoder_initialisation_types.index("none")
        b = len(encoder_initialisation_types) - 1
        encoder_initialisation_types[a], encoder_initialisation_types[b] = (
            encoder_initialisation_types[b],
            encoder_initialisation_types[a],
        )
        error_list[a], error_list[b] = (
            error_list[b],
            error_list[a],
        )
        test_rms_list[a], test_rms_list[b] = (
            test_rms_list[b],
            test_rms_list[a],
        )
    
    for i in range(nx):
        per_x_errors = []
        for error in error_list:
            per_x_errors.append(error[:, i])

        std_list[i].append(np.std(np.array(per_x_errors),axis=1))

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times New Roman"
scaling = 0.5
fig2 = plt.figure(figsize=[8.9 * scaling, 3.5 * scaling])  # type: ignore

for i in range(1):
    # plt.subplot(nx,1,i+1)
    # plt.yscale("log")
    plt.plot(np.array(snr_levels), np.array(std_list[i]))
    plt.ylabel(r"$\sigma_{\epsilon_{%d}}$" % (i+1))
    plt.grid()
    
plt.xlabel(r"SNR [dB]")
# plt.subplot(nx,1,1)
plt.legend(["model-based", "data-based"])
plt.tight_layout()


if save_flag:
    model_folder_path = os.path.join(
        root_dir,
        "figures",
        "encoder_initialisation",
        system_folder_name,
        baseline_model_folder_name,
        "SNR_levels"
    )
    os.makedirs(model_folder_path, exist_ok=True)

    fig_file_name = "state_error_std"
    plt.savefig(os.path.join(model_folder_path, fig_file_name + ".svg"), format="svg")
    plt.savefig(os.path.join(model_folder_path, fig_file_name + ".png"), format="png")
    plt.savefig(os.path.join(model_folder_path, fig_file_name + ".eps"), format="eps")
plt.show()
