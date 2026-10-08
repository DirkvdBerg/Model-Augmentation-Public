## Contact

Feel free to contact me directly for any question or issues.

Main developer: PhD candidate Jan Hoekstra at the TU/e. Control Systems. j.h.hoekstra@tue.nl

## Citation

When citing please use for model augmentation

> Jan H. Hoekstra, B. Györök, Roland Toth, Maarten Schoukens. Learning-based augmentation of first-principle models: A linear fractional representation-based approach; Submitted to Automatica, [arXiv](https://arxiv.org/abs/2602.17297)

and for encoder initialisation

> Jan H. Hoekstra, B. Györök, Roland Toth, Maarten Schoukens. Encoder initialisation methods in the model augmentation setting; Submitted to IFAC WC 2026, [arXiv](https://arxiv.org/abs/2602.13108)

Previous work on model augmentation

> Jan H. Hoekstra, Chris Verhoek, Roland Toth, Maarten Schoukens. Learning-based model augmentation with LFRs; European Journal of Control 2025, [Published version](https://www.sciencedirect.com/science/article/pii/S0947358025001335)

## Funding

Funded by the European Union (ERC, COMPLETE, 101075836). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## License

BSD 3-Clause License

# Brief explainer of project files
## model_augmentation
### fit_systems
  - **interconnect.py** contains the code for creating the "SS_encoder_general" required for training with "deepSI"
  - **blocks.py** contains the baseline and augmentation static function blocks that are connected by the interconnect
  - **pre_encoder.py** contains the code for pre training the encoder with a dataset contain state data
### systems
This folder contains system descriptions for generating data
### utils
  - **torch_nets.py** contains code for parameterising the learning functions in augmentation blocks
  - **utils.py** contains various helper functions and plot functions
## scripts / ecc_2025
Each folder here contains scripts for generating data, estimating models, and evaluating models. For example purpose, the **journal_model_augmentation** scripts are most recent and annotated. The scripts do the following
- **msd_ndof_data_generation.py** generates the dataset with the system description in the systems folder with flags determining what system data is generated for
- **msd_ndof_deepSI_encoder.py** estimates a black box deepSI encoder
- **msd_ndof_interconnect_fit.py** estimates the variety of different model augmentation structures with fags determining which model augmentation is trained from which dataset
- **msd_ndof_evaluate_fit_system.py** evaluate the RMSE and NRMSE scores of the models specified by the flags and returns their loss function curves
- **msd_ndof_state_comparison.py** compares the total model state with the augmentation on those states for select models as specified by the flags
