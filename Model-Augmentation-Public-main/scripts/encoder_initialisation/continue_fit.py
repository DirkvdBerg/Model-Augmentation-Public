import argparse
import os
import sys
import warnings

from deepSI import load_system_data, load_system

# disable some of the warnings from "incorrect" use of functions and modules
warnings.filterwarnings("ignore")

## ------------- Parse input arguments -----------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--encoder_type", nargs="?", default="linear_map", type=str, help="Encoder type"
)
parser.add_argument("--lr", nargs="?", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--epochs", nargs="?", type=int, default=2, help="Epoch count")
parser.add_argument(
    "--snr", nargs="?", type=int, default=20, help="Signa-to-noise ratio"
)
parser.add_argument(
    "--verbose", nargs="?", type=bool, default=True, help="Print code output"
)
parser.add_argument(
    "--version", nargs="?", type=int, default=1, help="Model version (iteration) to continue training"
)

args, unknown = parser.parse_known_args()

# changes the output window for print commands such that it does not print in the terminal
if not args.verbose:
    save_stdout = sys.stdout
    sys.stdout = open("trash", "w")

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

## ------------- Hyper params -----------------
# system parameters
system_dof = 2  # 2 or 3 (number of dof in system)
flag_linear_msd = False  # True or False (use linear mass spring damper system)
snr = args.snr

# FP (baseline) model parameters
FP_dof = 2  # 2 or 3 (number of dof in FP model)
FP_linear = True  # True or False (use linear FP model)
FP_type = "ideal"  # "ideal" or "approximate"

# encoder intialisation method
encoder_initialisation_type = args.encoder_type  # "pre_trained", "linear_map" or "none"

# model augmentation parameters
state_aug_type = "parallel"  # "parallel", "baseline"

# current epoch count model
current_epoch_count = 800

# model version
model_version_number = args.version

# training parameters
nf = 100
epochs = args.epochs
batch_size = 3000
# utility parameters
save_flag = True  # True or False (dont save model if False, e.g. for debugging)
wait_minutes = (
    0  # minutes to wait before starting the training (to not annoy colleagues)
)

optimizer_kwargs = {"lr": args.lr, "eps": 1e-8}

## ------------- Load data -----------------
# These should depend on the data generated in msd_ndof_data_generation.py, but other values are not supported yet
ny = 1
nu = 1

system_descriptors = ["msd", "{0}dof".format(system_dof)]
system_descriptors.append("linear" if flag_linear_msd else "nonlinear")
system_descriptors.append("multisine")
system_folder_name = "_".join(system_descriptors)

data_file_path = os.path.join(
    root_dir,
    "data",
    "encoder_initialisation",
    system_folder_name,
    "SNR_levels",
    "SNR{0}".format(snr),
)
train_data = load_system_data(os.path.join(data_file_path, "train.npz"))
val_data = load_system_data(os.path.join(data_file_path, "val.npz"))

## ------------- Load fit system -----------------
baseline_model_descriptors = ["FP", "{0}dof".format(FP_dof)]
baseline_model_descriptors.append("linear" if FP_linear else "nonlinear")
baseline_model_descriptors.append("ideal" if FP_type == "ideal" else "approximate")
baseline_model_folder_name = "_".join(baseline_model_descriptors)

encoder_initialiation_descriptors = [encoder_initialisation_type]
encoder_initialisation_folder_name = "_".join(encoder_initialiation_descriptors)

model_folder_path = os.path.join(
    root_dir,
    "models",
    "encoder_initialisation",
    system_folder_name,
    baseline_model_folder_name,
    "SNR_levels",
    "SNR{0}".format(snr),
    encoder_initialisation_folder_name,
)

model_augmentation_descriptors = []
if state_aug_type != "baseline":
    model_augmentation_descriptors.append("static")
    model_augmentation_descriptors.append(state_aug_type)
elif state_aug_type == "baseline":
    model_augmentation_descriptors.append("baseline")

# model_augmentation_descriptors.append("e{0}".format(epochs))
model_augmentation_file_name = "_".join(model_augmentation_descriptors)

# Determine version number for model file
model_file_path = os.path.join(model_folder_path, model_augmentation_file_name)
versioned_file_name = model_augmentation_file_name + "_e" + str(current_epoch_count) + "_v" + str(model_version_number)
fit_sys = load_system(os.path.join(model_folder_path, versioned_file_name))

print(versioned_file_name)

## ------------- Fit system -----------------
fit_sys.fit(
    train_sys_data=train_data,
    val_sys_data=val_data,
    batch_size=batch_size,
    epochs=epochs,
    auto_fit_norm=True,
    loss_kwargs={"nf": nf},
    # validation_measure="sim-RMS",
    validation_measure="{0}-step-RMS".format(nf),
    optimizer_kwargs=optimizer_kwargs,
)

## ------------- Save fit system -----------------
if save_flag:
    versioned_file_name = model_augmentation_file_name + "_e" + str(current_epoch_count + epochs) + "_v" + str(model_version_number)
    fit_sys.save_system(os.path.join(model_folder_path, versioned_file_name))

if not args.verbose:
    sys.stdout = save_stdout
