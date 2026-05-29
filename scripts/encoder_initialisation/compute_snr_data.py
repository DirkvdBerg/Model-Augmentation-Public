import os
import json

from deepSI import load_system_data, System_data
import numpy as np

from model_augmentation.utils.utils import (
    compute_noisy_y_from_SNR,
    determine_yes_no_query_binary_output,
)

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

## ------------- Flags -----------------
system_dof = 2  # 2 or 3 (number of dof in system)
flag_linear_msd = False  # True or False (use linear mass spring damper system)

snr_levels = [10, 20, 30, 50]  # in dB

flag_save = True  # True or False (dont save data if False, e.g. for debugging)

## ------------- Load data -----------------
system_descriptors = ["msd", "{0}dof".format(system_dof)]
system_descriptors.append("linear" if flag_linear_msd else "nonlinear")
system_descriptors.append("multisine")
system_folder_name = "_".join(system_descriptors)

data_file_path = os.path.join(
    root_dir, "data", "encoder_initialisation", system_folder_name
)
train = load_system_data(os.path.join(data_file_path, "train.npz"))
val = load_system_data(os.path.join(data_file_path, "val.npz"))
test = load_system_data(os.path.join(data_file_path, "test.npz"))

train = load_system_data(os.path.join(data_file_path, "train.npz"))
val = load_system_data(os.path.join(data_file_path, "val.npz"))
test = load_system_data(os.path.join(data_file_path, "test.npz"))

## ------------- Add noise -----------------
yes_no_query = determine_yes_no_query_binary_output("Should this data be saved? [y/n] ")

## seed random number generator for reproducibility
if type(snr_levels) is not list:
    snr_levels = [snr_levels]

for snr in snr_levels:
    np.random.seed(42 * snr)

    train_y_noisy, _ = compute_noisy_y_from_SNR(train.y, snr)
    val_y_noisy, _ = compute_noisy_y_from_SNR(val.y, snr)
    test_y_noisy, sigma_n = compute_noisy_y_from_SNR(test.y, snr)

    snr_train = 10 * np.log10(
        np.mean(train.y**2) / np.mean((train_y_noisy - train.y) ** 2)  # type: ignore
    )
    snr_val = 10 * np.log10(np.mean(val.y**2) / np.mean((val_y_noisy - val.y) ** 2))  # type: ignore
    snr_test = 10 * np.log10(np.mean(test.y**2) / np.mean((test_y_noisy - test.y) ** 2))  # type: ignore

    print(
        "Actual SNR values {0}dB: {1}, {2}, {3}".format(
            snr, snr_train, snr_val, snr_test
        )
    )

    if flag_save and yes_no_query:
        data_file_path = os.path.join(
            root_dir,
            "data",
            "encoder_initialisation",
            system_folder_name,
            "SNR_levels",
            "SNR{0}".format(snr),
        )
        os.makedirs(data_file_path, exist_ok=True)

        train_noisy = System_data(u=train.u, y=train_y_noisy)
        val_noisy = System_data(u=val.u, y=val_y_noisy)
        test_noisy = System_data(u=test.u, y=test_y_noisy)
        
        train_noisy.save(os.path.join(root_dir, data_file_path, "train.npz"))
        val_noisy.save(os.path.join(root_dir, data_file_path, "val.npz"))
        test_noisy.save(os.path.join(root_dir, data_file_path, "test.npz"))
        
        data = {"sigma_n": sigma_n}
        json_str = json.dumps(data, indent=4)
        with open(os.path.join(root_dir, data_file_path, "parameters.json"), "w") as f:
            json.dump(data, f)
