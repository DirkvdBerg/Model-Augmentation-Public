import torch
pt = torch.load('simulations/param_recovery_telica_xpos_-60_ypos-40/lfr_param_recovery_telica_xpos_-60_ypos-40_ETEL_e1_20260628_200050.pt', weights_only=False)
print('Keys:', list(pt.keys()))
print('best_epoch:', pt['best_epoch'])
print('best_log_params is None:', pt['best_log_params'] is None)
print('history length:', len(pt['history']))
print('history[0] keys:', list(pt['history'][0].keys()))
print('eval_train_rmse:', pt['eval_train_rmse'])
print('eval_train_rmse_ch:', pt['eval_train_rmse_ch'])
