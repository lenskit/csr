import numpy as np

import numba as n
from numba import njit
try:
    from numba.experimental import jitclass
except ImportError:
    from numba import jitclass

from .csr import _CSR

_CSR64 = type('_CSR64', _CSR.__bases__, dict(_CSR.__dict__))
_CSR = jitclass({
    'nrows': n.intc,
    'ncols': n.intc,
    'nnz': n.intc,
    'rowptrs': n.intc[::1],
    'colinds': n.intc[::1],
    'values': n.optional(n.float64[::1])
})(_CSR)
_CSR64 = jitclass({
    'nrows': n.intc,
    'ncols': n.intc,
    'nnz': n.int64,
    'rowptrs': n.int64[::1],
    'colinds': n.intc[::1],
    'values': n.optional(n.float64[::1])
})(_CSR64)


@njit
def _empty_csr(nrows, ncols, sizes):
    nnz = np.sum(sizes)
    rowptrs = np.zeros(nrows + 1, dtype=np.intc)
    for i in range(nrows):
        rowptrs[i+1] = rowptrs[i] + sizes[i]
    colinds = np.full(nnz, -1, dtype=np.intc)
    values = np.full(nnz, np.nan)
    return _CSR(nrows, ncols, nnz, rowptrs, colinds, values)


@njit
def _subset_rows(csr, begin, end):
    st = csr.rowptrs[begin]
    ed = csr.rowptrs[end]
    rps = csr.rowptrs[begin:(end+1)] - st

    cis = csr.colinds[st:ed]
    if csr.values.size == 0:
        vs = csr.values
    else:
        vs = csr.values[st:ed]
    return _CSR(end - begin, csr.ncols, ed - st, rps, cis, vs)
