"""
SciPy "kernel".  This kernel is not Numba-compatible, and will never be
selected as the default.  It primarily exists for ease in testing and
benchmarking CSR operations.
"""

from csr import CSR

__all__ = [
    'to_handle',
    'from_handle',
    'release_handle',
    'mult_ab'
]


def to_handle(csr):
    return csr.to_scipy()


def from_handle(h):
    return CSR.from_scipy(h, False)


def release_handle(h):
    pass


def mult_ab(A, B):
    return A @ B
