import numpy as np
import scipy.sparse as sps

from csr import CSR
from csr.test_utils import csrs, csr_slow, sparse_matrices

from pytest import mark, approx, raises
from hypothesis import given, assume, settings, HealthCheck
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph


@mark.parametrize('copy', [True, False])
def test_csr_from_sps(copy):
    "Test creating a CSR from a SciPy matrix"
    # initialize sparse matrix
    mat = np.random.randn(10, 5)
    mat[mat <= 0] = 0
    smat = sps.csr_matrix(mat)
    # make sure it's sparse
    assert smat.nnz == np.sum(mat > 0)

    csr = CSR.from_scipy(smat, copy=copy)
    assert csr.nnz == smat.nnz
    assert csr.nrows == smat.shape[0]
    assert csr.ncols == smat.shape[1]

    assert all(csr.rowptrs == smat.indptr)
    assert all(csr.colinds == smat.indices)
    assert all(csr.values == smat.data)
    assert isinstance(csr.rowptrs, np.ndarray)
    assert isinstance(csr.colinds, np.ndarray)
    assert isinstance(csr.values, np.ndarray)


@given(sparse_matrices(format='csr'), st.booleans())
def test_csr_from_sps_csr(smat, copy):
    "Test creating a CSR from a SciPy CSR matrix"
    csr = CSR.from_scipy(smat, copy=copy)
    assert csr.nnz == smat.nnz
    assert csr.nrows == smat.shape[0]
    assert csr.ncols == smat.shape[1]

    assert all(csr.rowptrs == smat.indptr)
    assert all(csr.colinds == smat.indices)
    assert all(csr.values == smat.data)
    assert isinstance(csr.rowptrs, np.ndarray)
    assert isinstance(csr.colinds, np.ndarray)
    if csr.nnz > 0:
        assert isinstance(csr.values, np.ndarray)


def test_csr_is_numpy_compatible():
    # initialize sparse matrix
    mat = np.random.randn(10, 5)
    mat[mat <= 0] = 0
    smat = sps.csr_matrix(mat)
    # make sure it's sparse
    assert smat.nnz == np.sum(mat > 0)

    csr = CSR.from_scipy(smat)

    d2 = csr.values * 10
    assert d2 == approx(smat.data * 10)


def test_csr_from_coo():
    "Make a CSR from COO data"
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_)

    csr = CSR.from_coo(rows, cols, vals)
    assert csr.nrows == 4
    assert csr.ncols == 3
    assert csr.nnz == 4
    assert csr.values == approx(vals)


def test_csr_from_coo_rand():
    for i in range(100):
        coords = np.random.choice(np.arange(50 * 100, dtype=np.int32), 1000, False)
        rows = np.mod(coords, 100, dtype=np.int32)
        cols = np.floor_divide(coords, 100, dtype=np.int32)
        vals = np.random.randn(1000)

        csr = CSR.from_coo(rows, cols, vals, (100, 50))
        rowinds = csr.rowinds()
        assert csr.nrows == 100
        assert csr.ncols == 50
        assert csr.nnz == 1000

        for i in range(100):
            sp = csr.rowptrs[i]
            ep = csr.rowptrs[i+1]
            assert ep - sp == np.sum(rows == i)
            points, = np.nonzero(rows == i)
            assert len(points) == ep - sp
            po = np.argsort(cols[points])
            points = points[po]
            assert all(np.sort(csr.colinds[sp:ep]) == cols[points])
            assert all(np.sort(csr.row_cs(i)) == cols[points])
            assert all(csr.values[np.argsort(csr.colinds[sp:ep]) + sp] == vals[points])
            assert all(rowinds[sp:ep] == i)

            row = np.zeros(50)
            row[cols[points]] = vals[points]
            assert np.sum(csr.row(i)) == approx(np.sum(vals[points]))
            assert all(csr.row(i) == row)


def test_csr_from_coo_novals():
    for i in range(50):
        coords = np.random.choice(np.arange(50 * 100, dtype=np.int32), 1000, False)
        rows = np.mod(coords, 100, dtype=np.int32)
        cols = np.floor_divide(coords, 100, dtype=np.int32)

        csr = CSR.from_coo(rows, cols, None, (100, 50))
        assert csr.nrows == 100
        assert csr.ncols == 50
        assert csr.nnz == 1000

        for i in range(100):
            sp = csr.rowptrs[i]
            ep = csr.rowptrs[i+1]
            assert ep - sp == np.sum(rows == i)
            points, = np.nonzero(rows == i)
            po = np.argsort(cols[points])
            points = points[po]
            assert all(np.sort(csr.colinds[sp:ep]) == cols[points])
            assert np.sum(csr.row(i)) == len(points)


def test_csr_to_sps():
    # initialize sparse matrix
    mat = np.random.randn(10, 5)
    mat[mat <= 0] = 0
    # get COO
    smat = sps.coo_matrix(mat)
    # make sure it's sparse
    assert smat.nnz == np.sum(mat > 0)

    csr = CSR.from_coo(smat.row, smat.col, smat.data, shape=smat.shape)
    assert csr.nnz == smat.nnz
    assert csr.nrows == smat.shape[0]
    assert csr.ncols == smat.shape[1]

    smat2 = csr.to_scipy()
    assert sps.isspmatrix(smat2)
    assert sps.isspmatrix_csr(smat2)

    for i in range(csr.nrows):
        assert smat2.indptr[i] == csr.rowptrs[i]
        assert smat2.indptr[i+1] == csr.rowptrs[i+1]
        sp = smat2.indptr[i]
        ep = smat2.indptr[i+1]
        assert all(smat2.indices[sp:ep] == csr.colinds[sp:ep])
        assert all(smat2.data[sp:ep] == csr.values[sp:ep])
