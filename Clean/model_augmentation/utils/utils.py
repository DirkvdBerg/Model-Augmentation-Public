import torch
import numpy as np


def added(obj):
    """Marks this class or function as added by this project (not Jan's original framework).
    Grep: grep -r "@added" model_augmentation/
    """
    return obj


@added
def to_tensor(input) -> torch.Tensor:
    if isinstance(input, type(None)):
        return None
    if isinstance(input, torch.Tensor):
        return input
    else:
        try: return torch.Tensor(input)
        except: raise TypeError("Input could not be converted to tensor")


def expansion_matrix(ix, n):
    S = np.zeros((n, len(ix)))
    S[ix, np.arange(len(ix)), ] = 1
    return torch.Tensor(S)

def selection_matrix(ix, n):
    E = np.zeros((len(ix), n))
    E[np.arange(len(ix)), ix] = 1
    return torch.Tensor(E)
