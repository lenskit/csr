"""
Kernel implementing matrix operations in pure Numba.
"""


import numpy as np
from numba import njit

from csr import CSR
import csr.native_ops as _ops
from .multiply import mult_ab, mult_abt


@njit
def to_handle(csr):
    """
    Convert a native CSR to a handle.  The caller must arrange for the CSR last at
    least as long as the handle.  The handle must be explicitly released.

    Handles are opaque as far as callers are concerned.
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


@njit
def order_columns(h):
    """
    Sort matrix rows in increasing column order.
    """
    _ops.sort_rows(h)


@njit(nogil=True)
def mult_vec(h: CSR, v):
    res = np.zeros(h.nrows)

    row = 0
    for i in range(h.nnz):
        # advance the row if necessary
        while i == h.rowptrs[row+1]:
            row += 1
        col = h.colinds[i]
        if h.has_values:
            res[row] += v[col] * h.values[i]
        else:
            res[row] += v[col]

    return res
