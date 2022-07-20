import logging
import numpy as np
import scipy.sparse as sps

from csr import CSR
from csr.test_utils import csrs, csr_slow, sparse_matrices

from pytest import mark, approx, raises
from hypothesis import given, assume, settings, HealthCheck
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph

_log = logging.getLogger(__name__)


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


def test_csr_from_coo_fixed():
    "Make a CSR from COO data"
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_)

    csr = CSR.from_coo(rows, cols, vals)
    assert csr.nrows == 4
    assert csr.ncols == 3
    assert csr.nnz == 4
    assert csr.values == approx(vals)


@given(st.data(), st.integers(0, 100), st.integers(0, 100),
       st.sampled_from(['f4', 'f8']))
def test_csr_from_coo(data, nrows, ncols, dtype):
    dtype = np.dtype(dtype)
    n = nrows * ncols
    nnz = data.draw(st.integers(0, int(n * 0.75)))
    _log.debug('testing %d×%d (%d nnz) of type %s', nrows, ncols, nnz, dtype)

    coords = st.integers(0, max(n - 1, 0))
    coords = data.draw(nph.arrays(np.int32, nnz, elements=coords, unique=True))
    rows = np.mod(coords, nrows, dtype=np.int32)
    cols = np.floor_divide(coords, nrows, dtype=np.int32)

    finite = nph.from_dtype(dtype, allow_infinity=False, allow_nan=False)
    vals = data.draw(nph.arrays(dtype, nnz, elements=finite))

    csr = CSR.from_coo(rows, cols, vals, (nrows, ncols))

    rowinds = csr.rowinds()
    assert csr.nrows == nrows
    assert csr.ncols == ncols
    assert csr.nnz == nnz

    for i in range(nrows):
        sp = csr.rowptrs[i]
        ep = csr.rowptrs[i + 1]
        assert ep - sp == np.sum(rows == i)
        points, = np.nonzero(rows == i)
        assert len(points) == ep - sp
        po = np.argsort(cols[points])
        points = points[po]
        assert all(np.sort(csr.colinds[sp:ep]) == cols[points])
        assert all(np.sort(csr.row_cs(i)) == cols[points])
        assert all(csr.values[np.argsort(csr.colinds[sp:ep]) + sp] == vals[points])
        assert all(rowinds[sp:ep] == i)

        row = np.zeros(ncols, dtype)
        row[cols[points]] = vals[points]
        assert all(csr.row(i) == row)


@given(st.data(), st.integers(0, 100), st.integers(0, 100))
def test_csr_from_coo_novals(data, nrows, ncols):
    n = nrows * ncols
    nnz = data.draw(st.integers(0, int(n * 0.75)))
    _log.debug('testing %d×%d (%d nnz) with no values', nrows, ncols, nnz)

    coords = st.integers(0, max(n - 1, 0))
    coords = data.draw(nph.arrays(np.int32, nnz, elements=coords, unique=True))
    rows = np.mod(coords, nrows, dtype=np.int32)
    cols = np.floor_divide(coords, nrows, dtype=np.int32)

    csr = CSR.from_coo(rows, cols, None, (nrows, ncols))

    rowinds = csr.rowinds()
    assert csr.nrows == nrows
    assert csr.ncols == ncols
    assert csr.nnz == nnz

    for i in range(nrows):
        sp = csr.rowptrs[i]
        ep = csr.rowptrs[i + 1]
        assert ep - sp == np.sum(rows == i)
        points, = np.nonzero(rows == i)
        assert len(points) == ep - sp
        po = np.argsort(cols[points])
        points = points[po]
        assert all(np.sort(csr.colinds[sp:ep]) == cols[points])
        assert all(np.sort(csr.row_cs(i)) == cols[points])
        assert all(rowinds[sp:ep] == i)

        row = csr.row(i)
        assert np.sum(row) == ep - sp


@given(st.data(), st.sampled_from(['csr', 'coo', 'csc']))
def test_sps_to_csr(data, format):
    mat = data.draw(sparse_matrices(format=format))
    nr, nc = mat.shape
    sp_csr: sps.csr_matrix = mat.tocsr()

    csr = CSR.from_scipy(mat)

    assert csr.ncols == nc
    assert csr.nrows == nr
    assert csr.nnz == mat.nnz
    assert np.all(csr.rowptrs == sp_csr.indptr)
    assert np.all(csr.colinds == sp_csr.indices)
    assert np.all(csr.values == sp_csr.data)


@given(csrs(values=True))
def test_csr_to_sps(csr):
    smat = csr.to_scipy()
    assert sps.isspmatrix(smat)
    assert sps.isspmatrix_csr(smat)

    for i in range(csr.nrows):
        assert smat.indptr[i] == csr.rowptrs[i]
        assert smat.indptr[i + 1] == csr.rowptrs[i + 1]
        sp = smat.indptr[i]
        ep = smat.indptr[i + 1]
        assert all(smat.indices[sp:ep] == csr.colinds[sp:ep])
        assert all(smat.data[sp:ep] == csr.values[sp:ep])
