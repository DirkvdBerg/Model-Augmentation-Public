import torch
import matplotlib.pyplot as plt
from pathlib import Path

PT_FILE = Path(__file__).parents[1] / "simulations/param_recovery/lfr_param_recovery_T1-T6-T2-T3-T4-T5_e600.pt"

data = torch.load(PT_FILE, map_location="cpu", weights_only=False)

eval_history = [h for h in data["history"] if "full_traj_rmse_m" in h]
epochs = [h["epoch"] for h in eval_history]
rmse   = [h["full_traj_rmse_m"] for h in eval_history]

# best_epoch = data["best_epoch"]
# best_rmse  = data["best_full_traj_rmse"]

fig, ax = plt.subplots()
ax.plot(epochs, rmse, label="Full-trajectory RMSE")
# ax.axvline(best_epoch, color="r", linestyle="--", label=f"Best epoch {best_epoch}")
# ax.axhline(best_rmse,  color="g", linestyle="--", label=f"Best RMSE {best_rmse:.2e}")
ax.set_xlabel("Epoch")
ax.set_ylabel("RMSE")
ax.set_title("Parameter recovery - convergence")
ax.legend()
plt.tight_layout()
plt.show()
