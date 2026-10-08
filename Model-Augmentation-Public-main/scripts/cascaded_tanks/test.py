import os
import matplotlib .pyplot as plt
import numpy as np
import torch
import deepSI
from scipy.io import loadmat

from model_augmentation.fit_systems.blocks import *
from model_augmentation.utils.utils import *
from model_augmentation.fit_systems.interconnect import *

fig_file_path = os.path.join(os.getcwd(), "figures", "cascaded_tanks")

u = deepSI.deepSI.exp_design.multisine(400, pmax=10, n_crest_factor_optim=1)

# scaling = 6/10; fig1 = plt.figure(figsize=[10*scaling, 5*scaling])
# plt.plot(u)
# plt.xticks([], []); plt.yticks([], [])
# plt.ylabel("u")
# plt.xlabel("k")
# plt.savefig(os.path.join(fig_file_path, "demo_u.svg"), format="svg")
# plt.show()

scaling = 6/10; fig2 = plt.figure(figsize=[10*scaling, 5*scaling], dpi=600)
for i in range(4):
    plt.plot(np.arange(100*i,100*(1+i)), u[100*i:100*(1+i)])
plt.xticks([], []); plt.yticks([], [])
# plt.ylabel("u")
# plt.xlabel("k")
plt.savefig(os.path.join(fig_file_path, "demo_u_split.png"), format="png", transparent=True)
plt.show()


# y = deepSI.deepSI.exp_design.multisine(400, pmax=18, n_crest_factor_optim=1)

# scaling = 6/10; fig1 = plt.figure(figsize=[10*scaling, 5*scaling])
# plt.plot(y)
# plt.xticks([], []); plt.yticks([], [])
# plt.ylabel("y")
# plt.xlabel("k")
# plt.savefig(os.path.join(fig_file_path, "demo_y.svg"), format="svg")
# plt.show()

# scaling = 6/10; fig1 = plt.figure(figsize=[10*scaling, 5*scaling])
# for i in range(4):
#     plt.plot(np.arange(100*i,100*(1+i)), y[100*i:100*(1+i)])
# plt.xticks([], []); plt.yticks([], [])
# plt.ylabel("y")
# plt.xlabel("k")
# plt.savefig(os.path.join(fig_file_path, "demo_y_split.svg"), format="svg")
# plt.show()

# nx = 2; ny = 1; nu = 1

# data_file_path = os.path.join(os.getcwd(), "data/cascaded_tanks/CascadedTankParameters.mat")
# mat_contents = loadmat(data_file_path, squeeze_me=True)
# CT_params = mat_contents["parameters"]

## ------------- Load fit sys -----------------
# model_file_name = "ct_state_aug_new_best"
# interconnect_file_path = os.path.join(os.getcwd(), "models", "cascaded_tanks", model_file_name)
# fit_sys : SSE_Interconnect = deepSI.load_system(interconnect_file_path)

# base_block = fit_sys.hfn.connected_blocks[0]
# updated_params = [base_block.k1.item(), base_block.k2.item(), base_block.k3.item(), base_block.k4.item(), base_block.k5.item(), base_block.k6.item(), 6.415797753620802, 10, base_block.yoffset.item()]
# print(updated_params)
# print(CT_params)
# print((updated_params - CT_params))


# data_file_path = os.path.join(os.getcwd(), "data/cascaded_tanks/dataBenchmark.mat")
# mat_contents = loadmat(data_file_path, squeeze_me=True)
# uEst = mat_contents["uEst"]
# yEst = mat_contents["yEst"]
# train_data = deepSI.System_data(u=uEst, y=yEst)
# uTest = mat_contents["uVal"]
# yTest = mat_contents["yVal"]
# test_data = deepSI.System_data(u=uTest, y=yTest)


# data_file_path = os.path.join(os.getcwd(), "data/cascaded_tanks/xSim.mat")
# mat_contents = loadmat(data_file_path, squeeze_me=True)
# x_sim = np.array(mat_contents["xOptPS"])
# data_file_path = os.path.join(os.getcwd(), "data/cascaded_tanks/xSimTest.mat")
# mat_contents = loadmat(data_file_path, squeeze_me=True)
# x_sim_test = np.array(mat_contents["xOptPSTest"])

# print(np.std(yEst[:]))
# plt.plot(yEst)
# plt.plot(x_sim[:,1])
# plt.show()

# Baseline_block = Cascaded_Tanks_State_Block(nz=nx+nu, nw=nx+ny, params=CT_params, sys_data=train_data)

# batch = 1; N = 1024
# y_log = np.zeros(N)
# x1_log = np.zeros(N)
# x2_log = np.zeros(N)

# x0 = to_tensor([CT_params[6] ,yEst[0]]).unsqueeze(-1).unsqueeze(0)
# x1_log[0] = CT_params[6]
# x2_log[0] = yEst[0]

# for i in range(N-1):
#     u0 = to_tensor([uEst[i]]).unsqueeze(-1).unsqueeze(0)
#     z0 = torch.hstack((x0,u0))

#     w = Baseline_block.nonlinear_function(z0)

#     x1 = w[:,:nx,:]; x0 = x1.clone()
#     y0 = w[:,nx:,:]

#     x1_log[i+1] = w[0,0,0].numpy()
#     x2_log[i+1] = w[0,1,0].numpy()
#     # y_log[i] = y0[0,0,0].numpy()

# time = np.arange(N)*4.0
# plt.subplot(121)
# plt.plot(time, x_sim[:,0], label = "matlab sim")
# plt.plot(time, x1_log, label = "python sim")
# plt.xlabel("time (s)")
# plt.ylabel("tank level")
# plt.legend(loc='lower left')
# plt.title("train tank 1")

# plt.subplot(122)
# plt.plot(time, x_sim[:,1], label = "matlab sim")
# plt.plot(time, x2_log, label = "python sim")
# plt.xlabel("time (s)")
# # plt.ylabel("tank level")
# plt.legend(loc='lower left')
# plt.title("train tank 2")
# plt.show()