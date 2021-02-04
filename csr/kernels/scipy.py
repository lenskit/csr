"""
SciPy "kernel".  This kernel is not Numba-compatible, and will never be
selected as the default.  It primarily exists for ease in testing and
benchmarking CSR operations.
"""

import numpy as np
from scipy.sparse import csr_matrix
from csr import CSR, _CSR

__all__ = [
    'to_handle',
    'from_handle',
    'release_handle',
    'mult_ab'
]


def to_handle(csr: _CSR):
    values = csr.values if csr.values.size > 0 else np.ones(csr.nnz)
    return csr_matrix((values, csr.colinds, csr.rowptrs), (csr.nrows, csr.ncols))


def from_handle(h):
    m: csr_matrix = h.tocsr()
    nr, nc = m.shape
    return _CSR(nr, nc, m.nnz, m.indptr, m.indices, m.data)


def release_handle(h):
    pass


def mult_ab(A, B):
    return A @ B
