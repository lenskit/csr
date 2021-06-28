"""
SciPy "kernel".  This kernel is not Numba-compatible, and will never be
selected as the default.  It primarily exists for ease in testing and
benchmarking CSR operations.
"""

import numpy as np
from scipy.sparse import csr_matrix
from csr import CSR

max_nnz = np.iinfo('i8').max


def to_handle(csr: CSR):
    values = csr.values
    if values is None:
        values = np.ones(csr.nnz)
    return csr_matrix((values, csr.colinds, csr.rowptrs), (csr.nrows, csr.ncols))


def from_handle(h):
    m: csr_matrix = h.tocsr()
    nr, nc = m.shape
    return CSR(nr, nc, m.nnz, m.indptr, m.indices, m.data)


def order_columns(h):
    h.sort_indices()


def release_handle(h):
    pass


def mult_ab(A, B):
    return A @ B


def mult_abt(A, B):
    return A @ B.T


def mult_vec(A, v):
    return A @ v
