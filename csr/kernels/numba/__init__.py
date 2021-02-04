"""
Kernel implementing matrix operations in pure Numba.
"""


import numpy as np
from numba import njit

from csr.layout import _CSR
from .multiply import mult_ab, mult_abt

__all__ = [
    'to_handle',
    'from_handle',
    'release_handle',
    'mult_ab',
    'mult_abt',
    'mult_vec'
]


@njit
def to_handle(csr):
    """
    Convert a native CSR to a handle.  The caller must arrange for the CSR last at
    least as long as the handle.  The handle must be explicitly released.
    """
    return csr


@njit
def from_handle(h):
    """
    Convert a handle to a CSR.  The handle may be released after this is called.
    """

    return h


@njit
def release_handle(h):
    """
    Release a handle.
    """
    pass


@njit(nogil=True)
def mult_vec(h: _CSR, v):
    res = np.zeros(h.nrows)
    have_values = h.values.size > 0

    row = 0
    for i in range(h.nnz):
        # advance the row if necessary
        while i == h.rowptrs[row+1]:
            row += 1
        col = h.colinds[i]
        if have_values:
            res[row] += v[col] * h.values[i]
        else:
            res[row] += v[col]

    return res
