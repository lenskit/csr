import numpy as np
from numba import njit
import numba.types as nt
from numba.experimental import jitclass

from csr import _CSR
from ._api import *

__all__ = [
    'mkl_h',
    'to_handle',
    'from_handle',
    'release_handle'
]


@jitclass([
    ('H', nt.intp),
    ('nrows', nt.intc),
    ('ncols', nt.intc),
    ('values', nt.float64[::1])
])
class mkl_h:
    """
    Type for MKL handles.  Do not use this directly.
    """
    def __init__(self, H, nrows, ncols, vs):
        self.H = H
        self.nrows = nrows
        self.ncols = ncols
        self.values = vs


@njit
def to_handle(csr: _CSR) -> mkl_h:
    _sp = ffi.from_buffer(csr.rowptrs)
    _cols = ffi.from_buffer(csr.colinds)
    if csr.has_values:
        vs = csr.values
    else:
        vs = np.ones(csr.nnz)
    _vals = ffi.from_buffer(vs)
    h = lk_mkl_spcreate(csr.nrows, csr.ncols, _sp, _cols, _vals)
    return mkl_h(h, csr.nrows, csr.ncols, vs)


@njit
def from_handle(h: mkl_h) -> _CSR:
    rvp = lk_mkl_spexport_p(h.H)
    if rvp is None:
        return None

    nrows = lk_mkl_spe_nrows(rvp)
    ncols = lk_mkl_spe_ncols(rvp)

    sp = lk_mkl_spe_row_sp(rvp)
    ep = lk_mkl_spe_row_ep(rvp)
    cis = lk_mkl_spe_colinds(rvp)
    vs = lk_mkl_spe_values(rvp)

    rowptrs = np.zeros(nrows + 1, dtype=np.intc)
    nnz = 0
    for i in range(nrows):
        nnz += ep[i] - sp[i]
        rowptrs[i+1] = nnz

    colinds = np.zeros(nnz, dtype=np.intc)
    values = np.zeros(nnz)

    for i in range(nrows):
        rs = rowptrs[i]
        re = rowptrs[i+1]
        ss = sp[i]
        for j in range(re - rs):
            colinds[rs + j] = cis[ss + j]
            values[rs + j] = vs[ss + j]

    lk_mkl_spe_free(rvp)

    return _CSR(nrows, ncols, nnz, rowptrs, colinds, values)


@njit
def release_handle(h: mkl_h):
    lk_mkl_spfree(h.H)
    h.H = 0
