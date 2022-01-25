import numpy as np
import scipy.sparse as sps

from csr import CSR
from csr.test_utils import csrs

from pytest import approx
from hypothesis import given


def test_csr_transpose():
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_)

    csr = CSR.from_coo(rows, cols, vals)
    csc = csr.transpose()
    assert csc.nrows == csr.ncols
    assert csc.ncols == csr.nrows

    assert all(csc.rowptrs == [0, 1, 3, 4])
    assert csc.colinds.max() == 3
    assert csc.values.sum() == approx(vals.sum())

    for r, c, v in zip(rows, cols, vals):
        row = csc.row(c)
        assert row[r] == v


def test_csr_transpose_coords():
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_)

    csr = CSR.from_coo(rows, cols, vals)
    csc = csr.transpose(False)
    assert csc.nrows == csr.ncols
    assert csc.ncols == csr.nrows

    assert all(csc.rowptrs == [0, 1, 3, 4])
    assert csc.colinds.max() == 3
    assert csc.values is None

    for r, c, v in zip(rows, cols, vals):
        row = csc.row(c)
        assert row[r] == 1


def test_csr_transpose_erow():
    "Test transposition with an empty output row"
    nrows = np.random.randint(10, 1000)
    ncols = np.random.randint(10, 500)
    mat = np.random.randn(nrows, ncols)
    mat[mat <= 0] = 0
    mat[:, 0:1] = 0
    smat = sps.csr_matrix(mat)

    csr = CSR.from_scipy(smat)
    csrt = csr.transpose()
    assert csrt.nrows == ncols
    assert csrt.ncols == nrows

    s2 = csrt.to_scipy()
    smat = smat.T.tocsr()
    assert all(smat.indptr == csrt.rowptrs)

    assert np.all(s2.toarray() == smat.toarray())


@given(csrs(values=True))
def test_csr_transpose_many(csr):
    csrt = csr.transpose()
    assert csrt.nrows == csr.ncols
    assert csrt.ncols == csr.nrows
    assert csrt.nnz == csr.nnz

    s2 = csrt.to_scipy()
    smat = csr.to_scipy().T.tocsr()
    assert all(smat.indptr == csrt.rowptrs)

    assert np.all(s2.toarray() == smat.toarray())
