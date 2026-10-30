"""Small helpers: the @added marker, tensor conversion and the selection and expansion matrices."""
# CHANGED: restricted to the thesis training path; full implementation in
# model_augmentation/utils/utils.py at the repository root
import torch
import numpy as np


def added(obj):
    """Mark a class or function as added by this project, not part of Jan's framework (grep "@added")."""
    return obj


def to_tensor(input) -> torch.Tensor:
    """input as a tensor; a tensor is returned as it is."""
    # CHANGED: None is no longer passed through (no caller passes it)
    return input if isinstance(input, torch.Tensor) else torch.Tensor(input)


def expansion_matrix(ix, n):
    """Routing matrix E.

    E (n, len(ix)) with E[ix[j], j] = 1: puts a len(ix) signal on the rows ix of an n signal.

    Args:
        ix: target rows; n: size of the target signal.
    Returns:
        (n, len(ix)) E.
    """
    S = np.zeros((n, len(ix)))
    S[ix, np.arange(len(ix)), ] = 1
    return torch.Tensor(S)

def selection_matrix(ix, n):
    """Selection matrix S.

    S (len(ix), n) with S[j, ix[j]] = 1: selects the rows ix of an n signal.

    Args:
        ix: selected rows; n: size of the source signal.
    Returns:
        (len(ix), n) S.
    """
    E = np.zeros((len(ix), n))
    E[np.arange(len(ix)), ix] = 1
    return torch.Tensor(E)
