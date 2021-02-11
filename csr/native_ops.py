"""
Backend implementations of Numba operations.
"""

import logging
import numpy as np
from numba import njit

from .csr import CSR, _row_extent

_log = logging.getLogger(__name__)


@njit
def subset_rows(csr, begin, end):
    "Take a subset of the rows of a CSR."
    st = csr.rowptrs[begin]
    ed = csr.rowptrs[end]
    rps = csr.rowptrs[begin:(end+1)] - st

    cis = csr.colinds[st:ed]
    if csr.has_values:
        vs = csr.values[st:ed]
    else:
        vs = None
    return CSR(end - begin, csr.ncols, ed - st, rps, cis, csr.has_values, vs)


@njit(nogil=True)
def center_rows(csr):
    "Mean-center the nonzero values of each row of a CSR."
    means = np.zeros(csr.nrows)
    for i in range(csr.nrows):
        sp, ep = row_extent(csr, i)
        if sp == ep:
            continue  # empty row
        vs = row_vs(csr, i)
        m = np.mean(vs)
        means[i] = m
        csr.values[sp:ep] -= m

    return means


@njit(nogil=True)
def unit_rows(csr):
    "Normalize the rows of a CSR to unit vectors."
    norms = np.zeros(csr.nrows)
    for i in range(csr.nrows):
        sp, ep = row_extent(csr, i)
        if sp == ep:
            continue  # empty row
        vs = row_vs(csr, i)
        m = np.linalg.norm(vs)
        norms[i] = m
        csr.values[sp:ep] /= m

    return norms


@njit(nogil=True)
def transpose(csr, include_values):
    "Transpose a CSR."
    brp = np.zeros(csr.ncols + 1, csr.rowptrs.dtype)
    bci = np.zeros(csr.nnz, np.int32)
    if include_values and csr.has_values:
        bvs = np.zeros(csr.nnz, np.float64)
    else:
        bvs = np.zeros(0)

    # count elements
    for i in range(csr.nrows):
        ars, are = row_extent(csr, i)
        for jj in range(ars, are):
            j = csr.colinds[jj]
            brp[j+1] += 1

    # convert to pointers
    for j in range(csr.ncols):
        brp[j+1] = brp[j] + brp[j+1]

    # construct results
    for i in range(csr.nrows):
        ars, are = row_extent(csr, i)
        for jj in range(ars, are):
            j = csr.colinds[jj]
            bci[brp[j]] = i
            if include_values and csr.has_values:
                bvs[brp[j]] = csr.values[jj]
            brp[j] += 1

    # restore pointers
    for i in range(csr.ncols - 1, 0, -1):
        brp[i] = brp[i-1]
    brp[0] = 0

    if not include_values or not csr.has_values:
        return make_structure(csr.ncols, csr.nrows, csr.nnz, brp, bci)
    else:
        return make_complete(csr.ncols, csr.nrows, csr.nnz, brp, bci, bvs)


@njit(nogil=True)
def sort_rows(csr):
    "Sort the rows of a CSR by increasing column index"
    for i in range(csr.nrows):
        sp, ep = row_extent(csr, i)
        # bubble-sort so it's super-fast on sorted arrays
        swapped = True
        while swapped:
            swapped = False
            for j in range(sp, ep - 1):
                if csr.colinds[j] > csr.colinds[j+1]:
                    _swap(csr.colinds, j, j+1)
                    if csr.has_values:
                        _swap(csr.values, j, j+1)
                    swapped = True
