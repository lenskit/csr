import numpy as np
import scipy.sparse as sps

from csr import CSR
from csr.test_utils import csrs, csr_slow, matrices, sparse_matrices

from pytest import mark, approx, raises
from hypothesis import given, assume, settings, HealthCheck
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph


def test_csr_rowinds():
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_)
    csr = CSR.from_coo(rows, cols, vals)

    ris = csr.rowinds()
    assert all(ris == rows)


def test_csr_str():
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_)

    csr = CSR.from_coo(rows, cols, vals)

    assert '4x3' in str(csr)


def test_csr_row_extent_fixed():
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_) + 1
    csr = CSR.from_coo(rows, cols, vals)

    assert csr.row_extent(0) == (0, 2)
    assert csr.row_extent(1) == (2, 3)
    assert csr.row_extent(2) == (3, 3)
    assert csr.row_extent(3) == (3, 4)


@given(sparse_matrices())
def test_csr_row_extent(smat):
    csr = CSR.from_scipy(smat)

    for i in range(csr.nrows):
        sp, ep = csr.row_extent(i)
        assert sp == csr.rowptrs[i]
        assert ep == csr.rowptrs[i+1]


def test_csr_row_fixed():
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_) + 1

    csr = CSR.from_coo(rows, cols, vals)
    assert all(csr.row(0) == np.array([0, 1, 2], dtype=np.float_))
    assert all(csr.row(1) == np.array([3, 0, 0], dtype=np.float_))
    assert all(csr.row(2) == np.array([0, 0, 0], dtype=np.float_))
    assert all(csr.row(3) == np.array([0, 4, 0], dtype=np.float_))


@settings(deadline=500)
@given(sparse_matrices())
def test_csr_row(smat):
    csr = CSR.from_scipy(smat)

    for i in range(csr.nrows):
        other = smat[i, :].toarray().ravel()
        row = csr.row(i)

        assert row.size == other.size

        assert all(row == other)


def test_csr_sparse_row():
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_)

    csr = CSR.from_coo(rows, cols, vals)
    assert all(csr.row_cs(0) == np.array([1, 2], dtype=np.int32))
    assert all(csr.row_cs(1) == np.array([0], dtype=np.int32))
    assert all(csr.row_cs(2) == np.array([], dtype=np.int32))
    assert all(csr.row_cs(3) == np.array([1], dtype=np.int32))

    assert all(csr.row_vs(0) == np.array([0, 1], dtype=np.float_))
    assert all(csr.row_vs(1) == np.array([2], dtype=np.float_))
    assert all(csr.row_vs(2) == np.array([], dtype=np.float_))
    assert all(csr.row_vs(3) == np.array([3], dtype=np.float_))


@given(csrs())
def test_drop_values(csr):
    csr.drop_values()
    assert csr.values is None


@given(csrs(), st.floats(allow_infinity=False, allow_nan=False))
def test_fill_values(csr, x):
    csr.fill_values(x)
    assert all(csr.values == x)


@given(matrices())
def test_csr_row_nnzs(mat):
    nrows, ncols = mat.shape

    # sparsify the matrix
    mat[mat <= 0] = 0
    smat = sps.csr_matrix(mat)
    # make sure it's sparse
    assume(smat.nnz == np.sum(mat > 0))
    csr = CSR.from_scipy(smat)

    nnzs = csr.row_nnzs()
    assert nnzs.sum() == csr.nnz
    for i in range(nrows):
        row = mat[i, :]
        assert nnzs[i] == np.sum(row > 0)
