from deepSI.fit_systems.encoders import SS_encoder_general_hf, default_state_net, default_output_net
import deepSI
import os
import numpy as np

## ------------- Load data -----------------
dof = 3
SNR = 20 # 20, 30, 60

data_file_path = os.path.join(os.getcwd(), "data", "mass_spring_damper")
train_data = deepSI.load_system_data(os.path.join(data_file_path, "msd_3dof_multisine_train.npz"))
val_data = deepSI.load_system_data(os.path.join(data_file_path, "msd_3dof_multisine_val.npz"))

## ------------- Add noise -----------------
if SNR == 20:
    sigma_n = 15e-3 # SNR20:15e-3
elif SNR == 30:
    sigma_n = 52e-4
elif SNR == 60:
    sigma_n = 15e-5
else:
    raise ValueError("SNR must be either 20, 30 or 60")
train_data.y = train_data.y + np.random.normal(0, sigma_n, train_data.y.shape)
val_data.y = val_data.y + np.random.normal(0, sigma_n, val_data.y.shape)

## ------------- Train fit system -----------------
h_net_kwargs = f_net_kwargs = {"n_hidden_layers": 2, "n_nodes_per_layer": 8}
e_net_kwargs = {"n_hidden_layers": 2, "n_nodes_per_layer": 16}
hf_net_kwargs = dict(f_net=default_state_net, f_net_kwargs=f_net_kwargs, h_net_kwargs=h_net_kwargs, h_net=default_output_net)
fit_sys = SS_encoder_general_hf(nx=dof*2, na=dof*4+1, nb=dof*4+1, e_net_kwargs=e_net_kwargs, hf_net_kwargs=hf_net_kwargs)
    
nf = 200; epochs = 10000; batch_size = 2000
fit_sys.fit(train_sys_data=train_data, val_sys_data=val_data, batch_size=batch_size, epochs=epochs, auto_fit_norm=True, loss_kwargs={'nf':nf}, validation_measure="sim-RMS")

# ------------- Save fit system -----------------
model_file_name = "msd_{0}dof_ANN_SS_e10000".format(dof)
interconnect_file_path = os.path.join(os.getcwd(), "models", "ecc_corrected", "ideal", "SNR{0}".format(SNR), model_file_name)
fit_sys.save_system(interconnect_file_path)