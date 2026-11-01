# Defense question overview

Generated 2026-10-31 by `defense_db.py overview`; do not edit.

Active questions 24 | evidence 99 | due today 21

| id | type | voice | last result | next due | answer | question |
|-|-|-|-|-|-|-|
| Q-001 | claim | Schoukens | shaky | 2026-10-31 | yes | Your conclusion states that the trained parameters are not interpretable as you  |
| Q-002 | claim | Toth | wrong | 2026-10-30 | yes | OBC makes the estimate of theta_base unique for the first-order one-step approxi |
| Q-003 | considered | Hoekstra | shaky | 2026-11-01 | yes | You minimise the simulation error over short windows, each started by the encode |
| Q-004 | discussion | Shakib | shaky | 2026-10-31 | yes | You train and evaluate in closed loop with the same controller that generated th |
| Q-005 | discussion | Gerritsen | solid | 2026-11-03 | yes | Your abstract opens with using the model to evaluate a changed feedback controll |
| Q-006 | considered | Shakib |  |  |  | Kessels simply runs the known controller over the measured record to get its sta |
| Q-007 | considered | Gerritsen | shaky | 2026-10-31 | yes | Why did you discretise the controller with the Tustin rule and not with zero-ord |
| Q-008 | fact | Gerritsen |  |  |  | At what sampling rate did you train, validate and test, and at what rate does th |
| Q-009 | explain | Hoekstra | shaky | 2026-10-31 | yes | Take one training window and walk me through it: what goes in, how the window ge |
| Q-010 | discussion | Schoukens |  |  | yes | Why a neural state-space augmentation and not a Volterra or polynomial nonlinear |
| Q-011 | discussion | Schoukens |  |  | yes | Your data are closed loop. How would you validate the model, and what criterion  |
| Q-012 | discussion | Toth |  |  | yes | Could the augmentation itself be LPV, with the network weights or outputs schedu |
| Q-013 | discussion | Toth |  |  | yes | Is the augmented model still well-posed over the full Y range, and can you say a |
| Q-014 | discussion | Toth |  |  | yes | How certain are your parameter estimates? Could you put uncertainty bounds on th |
| Q-015 | discussion | Hoekstra |  |  | yes | Why parallel augmentation and not a series or feedback LFR augmentation from my  |
| Q-016 | discussion | Hoekstra |  |  | yes | Now that you know the added states matter, what would you change in the encoder? |
| Q-017 | discussion | Hoekstra |  |  | yes | Could physics-informed neural networks have been an alternative to model augment |
| Q-018 | discussion | Gerritsen |  |  | yes | Could this run online or adaptively on the machine, for example re-estimating th |
| Q-019 | discussion | Gerritsen | wrong | 2026-10-30 | yes | What would it take to use this model with our real controller code and real mach |
| Q-020 | discussion | Gerritsen |  |  | yes | Would the trained model transfer to another gantry of the same type, or would yo |
| Q-021 | discussion | Shakib | wrong | 2026-10-30 | yes | Friction is static and nonlinear. Why not a Lur'e structure with an explicit fri |
| Q-022 | discussion | Shakib |  |  | yes | Why not classical model updating of the physical model instead of augmenting it  |
| Q-023 | discussion | Schoukens |  |  | yes | Did you look at Gaussian-process or Koopman-based models for the missing dynamic |
| Q-024 | explain | Schoukens | shaky | 2026-11-01 | yes | Explain how OBC works in your model, step by step: what is projected, onto what, |
