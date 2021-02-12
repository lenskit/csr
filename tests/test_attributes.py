import numpy as np
import scipy.sparse as sps

from csr import CSR
from csr.test_utils import csrs, matrices, sparse_matrices

from pytest import raises
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


def test_csr_set_values():
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_)

    csr = CSR.from_coo(rows, cols, vals)

    v2 = np.random.randn(4)
    csr.values = v2

    assert all(csr.values == v2)


def test_csr_set_values_oversize():
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_)

    csr = CSR.from_coo(rows, cols, vals)

    v2 = np.random.randn(6)
    csr.values = v2

    assert csr.values is not None
    assert all(csr.values == v2[:4])


def test_csr_set_values_undersize():
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_)

    csr = CSR.from_coo(rows, cols, vals)

    v2 = np.random.randn(3)

    with raises(ValueError):
        csr.values = v2

    assert all(csr.values == vals)


def test_csr_set_values_none():
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_)

    csr = CSR.from_coo(rows, cols, vals)
    csr.values = None

    assert csr.values is None
    assert all(csr.row(0) == [0, 1, 1])
    assert all(csr.row(1) == [1, 0, 0])
    assert all(csr.row(3) == [0, 1, 0])


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


@given(sparse_matrices())
def test_copy(mat):
    "Test copying a CSR"
    csr = CSR.from_scipy(mat)
    c2 = csr.copy()

    assert c2.nrows == csr.nrows
    assert c2.ncols == csr.ncols
    assert c2.nnz == csr.nnz
    assert c2.rowptrs is not csr.rowptrs
    assert all(c2.rowptrs == csr.rowptrs)
    assert c2.colinds is not csr.colinds
    assert all(c2.colinds == csr.colinds)
    assert c2.values is not csr.values
    assert all(c2.values == csr.values)


@given(sparse_matrices())
def test_copy_share(mat):
    "Test copying a CSR and sharing structure"
    csr = CSR.from_scipy(mat)
    c2 = csr.copy(copy_structure=False)

    assert c2.nrows == csr.nrows
    assert c2.ncols == csr.ncols
    assert c2.nnz == csr.nnz
    assert c2.rowptrs is csr.rowptrs
    assert c2.colinds is csr.colinds
    assert c2.values is not csr.values
    assert all(c2.values == csr.values)


@given(sparse_matrices())
def test_copy_structure_only(mat):
    "Test copying only the structure of a CSV."
    csr = CSR.from_scipy(mat)
    c2 = csr.copy(False)

    assert c2.nrows == csr.nrows
    assert c2.ncols == csr.ncols
    assert c2.nnz == csr.nnz
    assert c2.rowptrs is not csr.rowptrs
    assert all(c2.rowptrs == csr.rowptrs)
    assert c2.colinds is not csr.colinds
    assert all(c2.colinds == csr.colinds)
    assert c2.values is None


@given(csrs(values=False), st.booleans())
def test_copy_csrnv(csr, inc):
    "Test copying a CSR with no values."
    c2 = csr.copy(inc)

    assert c2.nrows == csr.nrows
    assert c2.ncols == csr.ncols
    assert c2.nnz == csr.nnz
    assert c2.rowptrs is not csr.rowptrs
    assert all(c2.rowptrs == csr.rowptrs)
    assert c2.colinds is not csr.colinds
    assert all(c2.colinds == csr.colinds)
    assert c2.values is None
