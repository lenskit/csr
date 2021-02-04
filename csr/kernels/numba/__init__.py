"""
Kernel implementing matrix operations in pure Numba.
"""


import numpy as np
from numba import njit

from csr.layout import _CSR
from csr.native_ops import row_extent
from .multiply import mult_ab

__all__ = [
    'to_handle',
    'from_handle',
    'release_handle',
    'mult_ab',
    'mult_vec'
]


@njit
def to_handle(csr):
    """
    Convert a native CSR to a handle.  The caller must arrange for the CSR last at
    least as long as the handle.  The handle must be explicitly released.
    """
    assert csr is not None
    return csr


@njit
def from_handle(h):
    """
    Convert a handle to a CSR.  The handle may be released after this is called.
    """

    assert h is not None
    return h


@njit
def release_handle(h):
    """
    Release a handle.
    """
    pass


@njit
def mult_vec(h: _CSR, v):
    res = np.zeros(h.nrows)

    for i in range(h.nrows):
        sp, ep = row_extent(h, i)
        for j in range(sp, ep):
            x = h.values[j] if h.values.size > 0 else 1
            res[i] += x * v[h.colinds[j]]

    return res
