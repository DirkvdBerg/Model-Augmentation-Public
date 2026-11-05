"""Project attribution marker, tensor conversion and signal selection and expansion matrices."""
# CHANGED: removed helpers unused by the thesis path; original: Model-Augmentation-Public-main
# CHANGED: algebraic-loop detection is handled in Interconnect.init_forward.
import torch
import numpy as np


def added(obj):
    """Project attribution marker; returns the class or function unchanged."""
    return obj


def to_tensor(input) -> torch.Tensor:
    """Convert to a tensor; existing tensors are returned unchanged."""
    # CHANGED: removed unused None handling; keep PyTorch's conversion errors instead of wrapping them.
    return input if isinstance(input, torch.Tensor) else torch.Tensor(input)


# CHANGED: local matrix names swapped to match the notation, E for expansion and S for selection (Jan's were reversed).
def expansion_matrix(ix, n):
    """E places values at rows ix, with zeros elsewhere: E[ix[j], j] = 1.

    Args:
        ix: destination row indices, one per input value.
        n: destination signal length.
    Returns:
        (n, len(ix)) expansion matrix E.
    """
    E = np.zeros((n, len(ix)))
    E[ix, np.arange(len(ix)), ] = 1
    return torch.Tensor(E)


def selection_matrix(ix, n):
    """S selects entries from a signal: S @ x = x[ix], with S[j, ix[j]] = 1.

    Args:
        ix: source indices, in the requested order.
        n: source signal length.
    Returns:
        (len(ix), n) selection matrix S.
    """
    S = np.zeros((len(ix), n))
    S[np.arange(len(ix)), ix] = 1
    return torch.Tensor(S)
