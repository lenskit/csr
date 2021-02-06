"""
Backend implementations of Numba operations.
"""

import logging
import numpy as np
from numba import njit

from .layout import _CSR, EMPTY_VALUES

_log = logging.getLogger(__name__)


@njit
def make_empty(nrows, ncols):
    rowptrs = np.zeros(nrows + 1, dtype=np.intc)
    colinds = np.zeros(0, dtype=np.intc)
    values = np.zeros(0)
    return _CSR(nrows, ncols, 0, rowptrs, colinds, values)


@njit
def make_unintialized(nrows, ncols, sizes):
    nnz = np.sum(sizes)
    rowptrs = np.zeros(nrows + 1, dtype=np.intc)
    for i in range(nrows):
        rowptrs[i+1] = rowptrs[i] + sizes[i]
    colinds = np.full(nnz, -1, dtype=np.intc)
    values = np.full(nnz, np.nan)
    return _CSR(nrows, ncols, nnz, rowptrs, colinds, values)


@njit
def row_extent(csr, row):
    "Get the extent of a row in the matrix storage."
    sp = csr.rowptrs[row]
    ep = csr.rowptrs[row+1]
    return sp, ep


@njit
def row(csr, row):
    "Get a row as a dense vector."
    v = np.zeros(csr.ncols)
    if csr.nnz == 0:
        return v

    sp, ep = row_extent(csr, row)
    cols = csr.colinds[sp:ep]
    if csr.has_values > 0:
        v[cols] = csr.values[sp:ep]
    else:
        v[cols] = 1

    return v


@njit
def row_cs(csr, row):
    "Get the column indices for a row."
    sp = csr.rowptrs[row]
    ep = csr.rowptrs[row + 1]

    return csr.colinds[sp:ep]


@njit
def row_vs(csr, row):
    "Get the nonzero values for a row."
    sp = csr.rowptrs[row]
    ep = csr.rowptrs[row + 1]

    if csr.values.size == 0:
        return np.full(ep - sp, 1.0)
    else:
        return csr.values[sp:ep]


@njit
def rowinds(csr):
    "Get the row indices for the nonzero values in a matrix."
    ris = np.zeros(csr.nnz, np.intc)
    for i in range(csr.nrows):
        sp, ep = row_extent(csr, i)
        ris[sp:ep] = i
    return ris


@njit
def subset_rows(csr, begin, end):
    "Take a subset of the rows of a CSR."
    st = csr.rowptrs[begin]
    ed = csr.rowptrs[end]
    rps = csr.rowptrs[begin:(end+1)] - st

    cis = csr.colinds[st:ed]
    if csr.values.size == 0:
        vs = EMPTY_VALUES
    else:
        vs = csr.values[st:ed]
    return _CSR(end - begin, csr.ncols, ed - st, rps, cis, vs)


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

    return _CSR(csr.ncols, csr.nrows, csr.nnz, brp, bci, bvs)


@njit(nogil=True)
def _csr_align(rowinds, nrows, rowptrs, align):
    rcts = np.zeros(nrows, dtype=rowptrs.dtype)
    for r in rowinds:
        rcts[r] += 1

    rowptrs[1:] = np.cumsum(rcts)
    rpos = rowptrs[:-1].copy()

    for i in range(len(rowinds)):
        row = rowinds[i]
        pos = rpos[row]
        align[pos] = i
        rpos[row] += 1
