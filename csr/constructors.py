"""
Numba-accessible constructors for CSRs.
"""

from .csr import CSR

import numpy as np
from numba import njit


@njit
def create_empty(nrows, ncols):
    """
    Create an empty CSR of the specified size.

    .. note:: This function can be used from Numba.
    """
    nrows = np.int32(nrows)
    ncols = np.int32(ncols)
    rowptrs = np.zeros(nrows + 1, dtype=np.intc)
    colinds = np.zeros(0, dtype=np.intc)
    values = np.zeros(0)
    return CSR(np.int32(nrows), np.int32(ncols), 0, rowptrs, colinds, values)


@njit
def create_novalues(nrows, ncols, nnz, rowptrs, colinds):
    """
    Create a CSR without values.
    """
    nrows = np.int32(nrows)
    ncols = np.int32(ncols)
    return CSR(nrows, ncols, nnz, rowptrs, colinds, None)


@njit
def create(nrows, ncols, nnz, rowptrs, colinds, values):
    """
    Create a CSR.
    """
    nrows = np.int32(nrows)
    ncols = np.int32(ncols)
    return CSR(nrows, ncols, nnz, rowptrs, colinds, values)


@njit
def create_from_sizes(nrows, ncols, sizes):
    """
    Create a CSR with uninitialized values and specified row sizes.

    This function is Numba-accessible, but is limited to creating matrices with fewer
    than :math:`2^{31}` nonzero entries and present value arrays.

    Args:
        nrows(int): the number of rows
        ncols(int): the number of columns
        sizes(numpy.ndarray): the number of nonzero values in each row
    """
    nrows = np.int32(nrows)
    ncols = np.int32(ncols)
    nnz = np.sum(sizes)
    assert nnz >= 0
    rowptrs = np.zeros(nrows + 1, dtype=np.int32)
    for i in range(nrows):
        rowptrs[i + 1] = rowptrs[i] + sizes[i]
    colinds = np.full(nnz, -1, dtype=np.intc)
    values = np.full(nnz, np.nan)
    return CSR(nrows, ncols, nnz, rowptrs, colinds, values)
