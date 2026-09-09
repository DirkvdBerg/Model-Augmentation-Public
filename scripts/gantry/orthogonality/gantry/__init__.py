"""Gantry-side integration for the fixed-reference trajectory orthogonality method.

Everything here is INTEGRATION. The mathematics lives in
`model_augmentation/fit_systems/trajectory_orth_projection.py` and is shared with the synthetic
testbed; the contract it is reached through lives in `trajectory_adapter.py`. Nothing in this
package re-implements a rollout: it calls the production encoder, the production interconnect
and the production closed-loop simulator, which is the single requirement the specification
places on a gantry adapter (Sect. 9.1).
"""
__project_origin__ = "added"
