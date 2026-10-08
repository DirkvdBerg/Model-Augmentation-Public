from deepSI.fit_systems.encoders import SS_encoder_general_hf, default_state_net, default_output_net
import deepSI
import os
import numpy as np
import time

## ------------- Hyper params -----------------
# system parameters
system_dof = 3 # 2 or 3 (number of dof in system)
flag_linear_msd = False # True or False (use linear mass spring damper system)
flag_low_pass_filter = "None" # "Input", "Output" or "None" (use low pass filter in system)
flag_input_saturation = True # True or False (use input saturation in system)

# training parameters
nf = 200; epochs = 5000; batch_size = 2000
# utility parameters
save_flag = True # True or False (dont save model if False, e.g. for debugging)
wait_minutes = 0 # minutes to wait before starting the training (e.g. to not annoy colleagues)

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
train_data = deepSI.load_system_data(os.path.join(data_file_path, "train.npz"))
val_data = deepSI.load_system_data(os.path.join(data_file_path, "val.npz"))

## ------------- Add noise -----------------
sigma_n = 52e-4 # SNR60:15e-5; SNR30:52e-4; SNR20:15e-3
train_data.y = train_data.y + np.random.normal(0, sigma_n, train_data.y.shape)
val_data.y = val_data.y + np.random.normal(0, sigma_n, val_data.y.shape)


## ------------- Define fit sys -----------------
h_net_kwargs = f_net_kwargs = {"n_hidden_layers": 2, "n_nodes_per_layer": 8}
e_net_kwargs = {"n_hidden_layers": 2, "n_nodes_per_layer": 16}
hf_net_kwargs = dict(f_net=default_state_net, f_net_kwargs=f_net_kwargs, h_net_kwargs=h_net_kwargs, h_net=default_output_net)
n_states = system_dof*2
fit_sys = SS_encoder_general_hf(nx=n_states, na=n_states*2+1, nb=n_states*2+1, e_net_kwargs=e_net_kwargs, hf_net_kwargs=hf_net_kwargs)

## ------------- Wait Time To Not Annoy Colleagues -----------------
for t in range(wait_minutes):
    time.sleep(60)
    print(f"Time passed: {t+1} minutes")

## ------------- Train fit system -----------------
fit_sys.fit(train_sys_data=train_data, val_sys_data=val_data, batch_size=batch_size, epochs=epochs, auto_fit_norm=True, loss_kwargs={'nf':nf}, validation_measure="sim-RMS")

## ------------- Save fit system -----------------
if save_flag:
    model_name_descriptors = ["resnet_hf_"]
    model_name_descriptors.append("e{0}".format(epochs))  
    model_file_name = "_".join(model_name_descriptors)

    model_folder_path = os.path.join(os.getcwd(), "models", "journal_model_augmentation", system_folder_name)
    os.makedirs(model_folder_path, exist_ok=True)
    fit_sys.save_system(os.path.join(model_folder_path, model_file_name))