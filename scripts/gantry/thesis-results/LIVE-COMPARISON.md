# Live added-state comparison

Four seed-1 runs use `Coulomb-tanh-and-MSD-lowpass-noise`, two added
states, the detuned physical start, and the shared thesis hyperparameters.
All four are unconstrained (`THESIS_ARM=u`, no OBC or orthogonal penalty).

`THESIS_ARCH=resnet|mlp` selects the learning function independently of
`THESIS_PS2=0|1`. Physical correction rows start at zero in both architectures,
and all output biases start at zero. Within an architecture, plain and PS2
receive identical complete initial parameters for the same seed.

- ResNet: zero nonlinear output and a live bias-free added-state linear bypass.
  Its live weights use `nn.init.kaiming_uniform_(a=sqrt(5))`, the `nn.Linear`
  default: uniform bounds `1/sqrt(n_in)` (approximately 0.302 for 11 inputs).
  This explicitly replaces the paper's unconstrained `U(-1,1)` initialization
  after a seed-1 overflow was observed. It is not a general stability guarantee.
- Tanh MLP: the existing two-hidden-layer network, with live added-state final
  weights using the same PyTorch rule with the final layer's fan-in (16, bounds
  0.25). Physical final weights remain zero. There is no linear bypass.
- Encoder: the analytical physical map is unchanged; the complete added-state
  history map retains Xavier uniform, gain 1, before splitting into Wy and Wu.

The architecture comparison therefore also includes their different live
initialization paths. It does not isolate architecture from initialization.
Run metadata records `AUGMENTATION_INIT`, `LIVE_G_WEIGHT_INIT`,
`WA_ENCODER_INIT`, and the resolved `data_dir`.

Focused local preflight (temporary checkpoint round-trips, no full validation):

```bash
python -B scripts/gantry/thesis-results/test_live_comparison.py
```

The preflight checks nonzero gradients to live g weights and both added-state
encoder history maps after the initial physical coupling update. Initially these
gradients are zero, as expected from baseline equivalence. It also checks the
physical encoder against the legacy initialization and round-trips both checkpoint
architectures. Test output is saved in `logs/test_live_comparison.log`.

Inspection requires the run's `config.json` beside each checkpoint. Both
`inspect_checkpoint.py` and `band_compare.py` reconstruct from this metadata;
missing metadata or conflicting arm/seed arguments fail explicitly. Band comparisons
require matching architecture and dataset. There is no default-ResNet guess.

Submit each matched pair from the server checkout, with at most two jobs:

Slurm logs are written to `/home/dirk_van_den_berg/logs/augmentation/live-init-g-aug/`.
Model and checkpoint output directories are unchanged.

```bash
bash scripts/gantry/thesis-results/submit.sh compare-resnet 2
bash scripts/gantry/thesis-results/submit.sh compare-mlp 2
```

Run the ResNet pair first. These commands submit jobs; implementing this
comparison does not itself submit anything. The new table supplies architecture
and PS2 explicitly, overriding inherited values for these four jobs.
