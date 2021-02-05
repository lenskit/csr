"""
Tests for creating kernel handles.
"""

from csr import CSR
from csr.test_utils import csrs, sparse_matrices

from hypothesis import given, settings


@settings(deadline=2000)
@given(sparse_matrices())
def test_make_handle(kernel, mat):
    csr = CSR.from_scipy(mat)
    h = kernel.to_handle(csr.R)
    try:
        assert h is not None
        c2 = kernel.from_handle(h)
        assert c2.nrows == csr.nrows
        assert c2.ncols == csr.ncols
        assert c2.nnz == csr.nnz
    finally:
        kernel.release_handle(h)
