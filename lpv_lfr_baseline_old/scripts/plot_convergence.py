"""
plot_convergence.py  —  training convergence from logged epoch data
Run: conda run -n GraduationProject python lpv_lfr_baseline/scripts/plot_convergence.py
"""

import matplotlib.pyplot as plt
import numpy as np

# Pasted from training log (every 25 epochs)
epochs     = [  0,  25,  50,  75, 100, 125, 150, 175, 200, 225, 250, 275,
               300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 599]
train_rmse = [7.2700e-04, 1.1306e-03, 8.0086e-04, 1.2764e-03, 1.2606e-03,
              7.7504e-04, 5.3206e-04, 5.0833e-04, 4.3892e-05, 2.4081e-04,
              2.3543e-04, 2.3910e-04, 1.3377e-04, 1.0495e-04, 7.9861e-05,
              2.1103e-05, 4.1844e-05, 7.6448e-05, 4.5412e-05, 3.4602e-06,
              7.1189e-05, 6.9836e-05, 2.8859e-05, 1.9663e-05, 1.7039e-05]
val_rmse   = [1.4941e-03, 1.4094e-03, 1.2244e-03, 9.9758e-04, 8.4384e-04,
              6.6194e-04, 5.1219e-04, 3.6347e-04, 2.6692e-04, 2.0449e-04,
              1.5908e-04, 1.1950e-04, 9.1240e-05, 7.9536e-05, 7.6968e-05,
              6.2200e-05, 6.2639e-05, 5.2881e-05, 5.3592e-05, 4.6972e-05,
              4.6972e-05, 4.5923e-05, 4.3682e-05, 4.1552e-05, 4.3660e-05]

epochs     = np.array(epochs)
train_rmse = np.array(train_rmse)
# val_rmse   = np.array(val_rmse)

fig, ax = plt.subplots(figsize=(7, 4))

ax.semilogy(epochs, train_rmse, marker='o', markersize=3, label='Train RMSE')
# ax.semilogy(epochs, val_rmse,   marker='s', markersize=3, label='Val RMSE')

ax.set_xlabel('Epoch')
ax.set_ylabel('RMSE [m]')
ax.set_title('Training convergence  (600 epochs, lr=1e-3, batch=8x4000)')
ax.legend()
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

fig.tight_layout()
out = 'simulations/param_recovery/convergence.pdf'
fig.savefig(out, dpi=150)
print(f'Saved: {out}')
plt.show()
