from deepSI import load_system_data, load_system

from torch import no_grad
import sys
import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
    "none",
]  # "pre_trained", "linear_map", "none"
encoder_initialisation_types_plot_naming = {
    "pre_trained": "Data-based",
    "linear_map": "Model-based",
    "none": "Random init",
}
# select specific models
flag_select_augmentation_types = True
augmentation_types = ["static_parallel"]  # "static_parallel" or "baseline"

# filter model flags
flag_select_epoch_count = True
n_epochs_flag = 2000
snr_level = 20  # in dB
flag_only_load_first = False

multi_vall_loss_nf = [5, 20, 100] # selected from [1, 5, 20, 50, 100] interconncet_fit.py. Need to be selected during training

# select system parameters
system_dof = 2  # 2 or 3 (number of dof in system)
flag_linear_msd = False  # True or False (use linear mass spring damper system)

# select FP (baseline) model parameters
FP_dof = 2  # 2 or 3 (number of dof in FP model)
FP_linear = False  # True or False (use linear FP model)
FP_type = "ideal"  # "ideal" or "approximate"

# debug flags
save_flag = True  # True or False (dont save model if False, e.g. for debugging)

## ------------- Validation measures -----------------
# plot flags
flag_plot_validation_loss = True
flag_plot_all_loss = True

## ------------- Load data -----------------
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
    "SNR_levels",
    "SNR{0}".format(snr_level),
)
test_data = load_system_data(os.path.join(root_dir, data_file_path, "test.npz"))

with open(os.path.join(root_dir, data_file_path, "parameters.json"), "r") as file:
    data_parameters = json.load(file)
sigma_n = data_parameters["sigma_n"]

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
# list the plot naming in the same order as above
encoder_initialisation_types_plot_naming_list = [
    encoder_initialisation_types_plot_naming[type]
    for type in encoder_initialisation_types
]

model_simulations_folder_path = os.path.join(
    root_dir,
    "model_simulations",
    "encoder_initialisation",
    system_folder_name,
    baseline_model_folder_name,
    "SNR_levels",
    "SNR{0}".format(snr_level),
)

fit_sys_list = []
fit_sys_name_list = []
test_list = []
number_of_models_per_encoder_init_type = np.zeros((len(encoder_initialisation_types),))
for encoder_initialisation_folder_name in os.listdir(model_folder_path):
    if (
        not flag_select_encoder_init_type
        or encoder_initialisation_folder_name in encoder_initialisation_types
    ):
        encoder_initialisation_folder_path = os.path.join(
            model_folder_path, encoder_initialisation_folder_name
        )
        encoder_initialisation_simulations_folder_path = os.path.join(
            model_simulations_folder_path, encoder_initialisation_folder_name
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
                    with no_grad():
                        fit_sys = load_system(fit_sys_file_path)
                    fit_sys_list.append(fit_sys)
                    fit_sys_name_list.append(
                        encoder_initialisation_folder_name
                        + "_"
                        + fit_sys_file_name.split("_e", 1)[0]
                        + "_v"
                        + fit_sys_file_name.split("_v", 1)[1]
                    )

                    ## ------------- Run models -----------------
                    # test_list.append(fit_sys.apply_experiment(test_data))

                    fit_sys_simulation_file_path = os.path.join(
                        encoder_initialisation_simulations_folder_path,
                        fit_sys_file_name + ".npz",
                    )

                    # if os.path.exists(fit_sys_simulation_file_path):
                    try:
                        test_list.append(load_system_data(fit_sys_simulation_file_path))
                        print("Loaded test simulation for model: " + fit_sys_file_name)
                    except Exception:
                        # else:
                        print(
                            "Test simulation for model: "
                            + encoder_initialisation_folder_name
                            + " "
                            + fit_sys_file_name
                            + " does not exist. Currently generating."
                        )
                        os.makedirs(
                            encoder_initialisation_simulations_folder_path,
                            exist_ok=True,
                        )
                        with no_grad(): # type: ignore
                            test = fit_sys.apply_experiment(test_data)
                        test.save(fit_sys_simulation_file_path)
                        test_list.append(test)

                    if flag_select_epoch_count and flag_only_load_first:
                        break

print(fit_sys_name_list)
print(number_of_models_per_encoder_init_type)

## ------------- Plot val loss -----------------
if "none" in encoder_initialisation_types:
    a = encoder_initialisation_types.index("none")
    b = len(encoder_initialisation_types) - 1
    encoder_initialisation_types[a], encoder_initialisation_types[b] = (
        encoder_initialisation_types[b],
        encoder_initialisation_types[a],
    )
    encoder_initialisation_types_plot_naming_list[a], encoder_initialisation_types_plot_naming_list[b] = (
        encoder_initialisation_types_plot_naming_list[b],
        encoder_initialisation_types_plot_naming_list[a],
    )

ixs_val_nf = [multi_vall_loss_nf.index(nf) for nf in multi_vall_loss_nf]

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times New Roman"
scaling = 0.9
fig2 = plt.figure(figsize=[8.9 * scaling, 7.5 * scaling])  # type: ignore


for ix, ix_vall_nf, nf_val in zip(range(len(encoder_initialisation_types)), ixs_val_nf, multi_vall_loss_nf):
    plt.subplot(len(encoder_initialisation_types),1,ix+1)
    plt.yscale("log")
    plt.xscale("log")

    # get default color list
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    max_epoch = 0
    min_epoch = sys.maxsize
    for color, encoder_init_type in zip(colors, encoder_initialisation_types):
        val_loss_list = []
        max_epoch = 0
        min_epoch = sys.maxsize
        for fit_sys, name in zip(fit_sys_list, fit_sys_name_list):
            if encoder_init_type in name:
                fit_sys.checkpoint_load_system("_last")

                # val_loss_list.append(fit_sys.Loss_val)

                val_loss_list.append(fit_sys.multi_loss_val[ix_vall_nf::5])

                # for i in range(5):
                #     val_loss_list.append(fit_sys.multi_loss_val[i::5])

                if int(max(fit_sys.epoch_id)) > max_epoch:
                    max_epoch = int(max(fit_sys.epoch_id))
                if int(max(fit_sys.epoch_id) < min_epoch):
                    min_epoch = int(max(fit_sys.epoch_id))
            for i, val_loss in enumerate(val_loss_list):
                val_loss_list[i] = val_loss_list[i][
                    : min_epoch + 1
                ]  # + 1 is for 0 epoch that is also included
        val_loss_array = np.array(val_loss_list)

        if flag_plot_all_loss:
            # for i, nf in enumerate([1, 5, 20, 50, 100]):
            #     plt.plot(np.arange(val_loss_array.shape[1]), val_loss_array.T[:,i], label=str(nf), alpha=0.8)
            for i in range(val_loss_array.shape[0]):
                if i == 0:
                    plt.plot(
                        np.arange(val_loss_array.shape[1]),
                        val_loss_array.T[:, i],
                        color=color,
                        label=encoder_initialisation_types_plot_naming[
                            encoder_init_type
                        ],
                        alpha=0.3,
                    )
                else:
                    plt.plot(
                        np.arange(val_loss_array.shape[1]),
                        val_loss_array.T[:, i],
                        color=color,
                        alpha=0.3,
                    )

    # plt.plot(
    #     np.arange(min_epoch + 1),
    #     np.ones(min_epoch + 1) * sigma_n,
    #     "r--",
    #     label="noise floor",
    # )
    # plt.ylabel(r"{0} step RMSE".format(nf_val))
    if ix == len(encoder_initialisation_types) - 1:
        plt.xlabel(r"epochs")
    if ix == 0:
        plt.legend()
        
    plt.xlim([0, min_epoch])
    # plt.ylim([0.01, 0.5])
    plt.ylabel(r"{0}-step RMSE".format(nf_val))
    plt.grid()
    
plt.tight_layout()

if save_flag:
    model_folder_path = os.path.join(
        root_dir,
        "figures",
        "encoder_initialisation",
        system_folder_name,
        baseline_model_folder_name,
        "SNR_levels",
        "SNR{0}".format(snr_level),
    )
    os.makedirs(model_folder_path, exist_ok=True)

    fig_file_name = "multi_nf_val_loss_plot"
    plt.savefig(
        os.path.join(model_folder_path, fig_file_name + ".svg"), format="svg"
    )
    plt.savefig(
        os.path.join(model_folder_path, fig_file_name + ".png"), format="png"
    )
    plt.savefig(
        os.path.join(model_folder_path, fig_file_name + ".eps"), format="eps"
    )
plt.show()