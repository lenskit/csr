import numpy as np
from numba import njit

from csr import _CSR
from ._api import *

__all__ = [
    'mult_ab',
    'mult_abt'
]


@njit(nogil=True)
def mult_ab(a_h, b_h):
    pass


@njit(nogil=True)
def mult_abt(a_h, b_h):
    pass


@njit(nogil=True)
def mult_vec(a_h, x):
    y = np.zeros(a_h.nrows, dtype=np.float64)

    _x = ffi.from_buffer(x)
    _y = ffi.from_buffer(y)

    lk_mkl_spmv(1.0, a_h.H, _x, 0.0, _y)
    return y
