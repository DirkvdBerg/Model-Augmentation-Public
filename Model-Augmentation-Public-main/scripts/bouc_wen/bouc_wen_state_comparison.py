from deepSI.utils.torch_nets import simple_res_net
from deepSI.fit_systems.encoders import SS_encoder_general_hf
import deepSI

import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from model_augmentation.utils.utils import *
from model_augmentation.fit_systems.interconnect import *
from model_augmentation.fit_systems.blocks import *
from model_augmentation.systems.mass_spring_damper import *

## ------------- Load data -----------------
# input data
data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/uval_multisine.mat")
mat_contents = loadmat(data_file_path, squeeze_me=True)
uTest = mat_contents["uval_multisine"]

# output data
data_file_path = os.path.join(os.getcwd(), "data/bouc_wen/yval_multisine.mat")
mat_contents = loadmat(data_file_path, squeeze_me=True)
yTest = mat_contents["yval_multisine"]

test_data = deepSI.System_data(u=uTest, y=yTest)

## ------------- Load fit system -----------------
fit_sys_file_name_list = ["boucWen-FP_linear-dynamic_parallel",
                          "boucWen-FP_linear-static_parallel",
                          "boucWen-FP_nonlinear-dynamic_parallel_e3000",
                          "boucWen-FP_nonlinear-static_parallel",
                          "boucWen-BLA_2-dynamic_parallel_e3000",
                          "boucWen-BLA_2-static_parallel",
                          "boucWen-BLA_3-parallel_e3000",
                          "boucWen-ANN_SS_e3700"]
sys_file_name = "boucWen-BLA_3-parallel_e4000"
interconnect_file_path =os.path.join(os.getcwd(), "models", "bouc_wen", sys_file_name)

interconnect : SSE_Interconnect = deepSI.load_system(interconnect_file_path)

## ------------- Plot state component comparison -----------------
interconnect.hfn.reset_saved_signals()
test_interconnection = interconnect.apply_experiment(test_data)
saved_input_signals = interconnect.hfn.saved_input_signals
saved_output_signals = interconnect.hfn.saved_output_signals

# print(saved_input_signals.shape)
# print(saved_output_signals.shape)

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
scaling = 6/10
fig1 =  plt.figure(figsize=[9.2*scaling, 8.2*scaling])
plt_range = 400
plt_centre = 200

nxd = 3

for i in range(nxd):
    plt.subplot(nxd,1,i+1)
    plt.plot(saved_output_signals[i, plt_centre - plt_range //2 : plt_centre + plt_range //2 ], label="state")
    plt.plot(saved_input_signals[-nxd+i, plt_centre - plt_range //2 : plt_centre + plt_range //2 ], label="augmentation")
    # plt.plot(saved_output_signals[i,:], label="state")
    # plt.plot(saved_input_signals[-nxd+i,:], label="augmentation")
    

    plt.ylabel(r"$x_{0}$".format(i+1))
    # plt.xlim([0,plt_range])
    plt.grid()
    # plt.xticks([100, 200, 300, 400], [])

# plt.xticks([100, 200, 300, 400],[100, 200, 300, 400])
plt.xlabel(r"$k$")
# plt.tight_layout()

# state_comp_file_name = "state_component_comparison"
fig_file_path = os.path.join(os.getcwd(), "figures\\bouc_wen\\")
plt.savefig(os.path.join(fig_file_path, "state_component_comparison.svg"), format="svg")
plt.savefig(os.path.join(fig_file_path, "state_component_comparison.png"), format="png")
plt.savefig(os.path.join(fig_file_path, "state_component_comparison.eps"), format="eps")

plt.show()