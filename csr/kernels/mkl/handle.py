import sys
from typing import Optional
import dataclasses
import numpy as np
from numba import njit, config
from numba.core.types import StructRef
from numba.experimental import structref

from csr import CSR
from ._api import *  # noqa: F403
from csr.constructors import create_empty

__all__ = [
    'mkl_h',
    'to_handle',
    'from_handle',
    'release_handle'
]


@structref.register
class mkl_h_type(StructRef):
    "Internal Numba type for MKL handles"
    pass


if config.DISABLE_JIT:
    @dataclasses.dataclass
    class mkl_h:
        H: int
        nrows: int
        ncols: int
        values: Optional[np.ndarray]
else:
    class mkl_h(structref.StructRefProxy):
        """
        Type for MKL handles.  Opaque, do not use directly.
        """

    structref.define_proxy(mkl_h, mkl_h_type, ['H', 'nrows', 'ncols', 'values'])


@njit
def to_handle(csr: CSR) -> mkl_h:
    if csr.nnz == 0:
        # empty matrices don't really work
        return mkl_h(0, csr.nrows, csr.ncols, np.zeros(0))

    _sp = ffi.from_buffer(csr.rowptrs)
    _cols = ffi.from_buffer(csr.colinds)
    vs = csr._required_values()
    assert vs.size == csr.nnz
    _vals = ffi.from_buffer(vs)
    h = lk_mkl_spcreate(csr.nrows, csr.ncols, _sp, _cols, _vals)
    lk_mkl_spopt(h)
    return mkl_h(h, csr.nrows, csr.ncols, vs)


@njit
def from_handle(h: mkl_h) -> CSR:
    if not h.H:
        return create_empty(h.nrows, h.ncols)

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
        rowptrs[i + 1] = nnz
    assert nnz == ep[nrows - 1]

    colinds = np.zeros(nnz, dtype=np.intc)
    values = np.zeros(nnz)

    for i in range(nrows):
        rs = rowptrs[i]
        re = rowptrs[i + 1]
        ss = sp[i]
        for j in range(re - rs):
            colinds[rs + j] = cis[ss + j]
            values[rs + j] = vs[ss + j]

    lk_mkl_spe_free(rvp)

    return CSR(nrows, ncols, nnz, rowptrs, colinds, values)


@njit
def order_columns(h):
    """
    Sort matrix rows in increasing column order.
    """
    if h.H:
        lk_mkl_sporder(h.H)


@njit
def release_handle(h: mkl_h):
    if h.H:
        lk_mkl_spfree(h.H)
    h.H = 0
