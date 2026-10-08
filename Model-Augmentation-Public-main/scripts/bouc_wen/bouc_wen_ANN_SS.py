from deepSI.fit_systems.encoders import SS_encoder_general_hf, default_state_net, default_output_net

import os
import deepSI
from scipy.io import loadmat

from model_augmentation.utils.utils import *
from model_augmentation.fit_systems.interconnect import *
from model_augmentation.fit_systems.blocks import *
from model_augmentation.systems.mass_spring_damper import *

## ------------- Load data -----------------
# input data
data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/uest_multisine.mat")
mat_contents = loadmat(data_file_path, squeeze_me=True)
uEst = mat_contents["u"]

# output data
data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/yest_multisine.mat")
mat_contents = loadmat(data_file_path, squeeze_me=True)
yEst = mat_contents["y"]

data = deepSI.System_data(u=uEst, y=yEst)
train_data, val_data = data.train_test_split(split_fraction=0.25)

## ------------- Create Model -----------------
nx = 3
h_net_kwargs = f_net_kwargs = {"n_hidden_layers": 2, "n_nodes_per_layer": 16}
e_net_kwargs = {"n_hidden_layers": 2, "n_nodes_per_layer": 8}
hf_net_kwargs = dict(f_net=default_state_net, f_net_kwargs=f_net_kwargs, h_net_kwargs=h_net_kwargs, h_net=default_output_net)
fit_sys = SS_encoder_general_hf(nx=nx, na=nx*2+1, nb=nx*2+1, e_net_kwargs=e_net_kwargs, hf_net_kwargs=hf_net_kwargs)
    
## ------------- Train Model -----------------
nf = 500; epochs = 5000; batch_size = 2000
fit_sys.fit(train_sys_data=train_data, val_sys_data=val_data, batch_size=batch_size, epochs=epochs, auto_fit_norm=True, loss_kwargs={'nf':nf}, validation_measure="sim-RMS")

model_file_name = "boucWen-ANN_SS"

interconnect_file_path = os.path.join(os.getcwd(), "models", "bouc_wen", model_file_name)
fit_sys.save_system(interconnect_file_path)