"""
Kernel implementing matrix operations in pure Numba.
"""


import numpy as np
from numba import njit
import numba.types as nt
from numba.experimental import jitclass

from csr.layout import _CSR
from csr.native_ops import row_extent, row_cs, row_vs
from .multiply import mult_ab, mult_abt

__all__ = [
    'to_handle',
    'from_handle',
    'release_handle',
    'mult_ab',
    'mult_abt',
    'mult_vec'
]


@jitclass([
    ('nrows', nt.intc),
    ('ncols', nt.intc),
    ('nnz', nt.intc),
    ('has_values', nt.boolean),
    ('rowptrs', nt.intc[:]),
    ('colinds', nt.intc[:]),
    ('values', nt.float64[:])
])
class CSR_H:
    def __init__(self, csr):
        self.nrows = csr.nrows
        self.ncols = csr.ncols
        self.nnz = csr.nnz
        self.rowptrs = csr.rowptrs
        self.colinds = csr.colinds
        self.values = csr.values
        self.has_values = self.values.size > 0


@njit
def to_handle(csr):
    """
    Convert a native CSR to a handle.  The caller must arrange for the CSR last at
    least as long as the handle.  The handle must be explicitly released.
    """
    return CSR_H(csr)


@njit
def from_handle(h):
    """
    Convert a handle to a CSR.  The handle may be released after this is called.
    """

    return _CSR(h.nrows, h.ncols, h.nnz, h.rowptrs, h.colinds, h.values)


@njit
def release_handle(h):
    """
    Release a handle.
    """
    pass


@njit(nogil=True)
def mult_vec(h: CSR_H, v):
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
