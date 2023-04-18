import logging
import numpy as np
import scipy.sparse as sps

from csr import CSR
from csr.test_utils import csr_slow, csrs, matrices

from pytest import raises
from hypothesis import given, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph

_log = logging.getLogger(__name__)


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


@csr_slow()
@given(csrs(st.integers(1, 100), st.integers(1, 100)))
def test_csr_row_extent(csr):
    for i in range(csr.nrows):
        sp, ep = csr.row_extent(i)
        assert sp == csr.rowptrs[i]
        assert ep == csr.rowptrs[i + 1]


def test_csr_row_fixed():
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_) + 1

    csr = CSR.from_coo(rows, cols, vals)
    assert all(csr.row(0) == np.array([0, 1, 2], dtype=np.float_))
    assert all(csr.row(1) == np.array([3, 0, 0], dtype=np.float_))
    assert all(csr.row(2) == np.array([0, 0, 0], dtype=np.float_))
    assert all(csr.row(3) == np.array([0, 4, 0], dtype=np.float_))


@csr_slow()
@given(csrs(st.integers(1, 100), st.integers(1, 100)))
def test_csr_row(csr):
    smat = csr.to_scipy()

    for i in range(csr.nrows):
        other = smat[i, :].toarray().ravel()
        row = csr.row(i)

        assert row.size == other.size

        assert all(row == other)


@csr_slow()
@given(st.data(), csrs(st.integers(1, 100), st.integers(1, 100)))
def test_csr_multi_rows(data, csr):
    smat = csr.to_scipy()

    rows = data.draw(st.lists(st.integers(0, csr.nrows - 1), max_size=csr.nrows, unique=True))
    row_arrs = csr.row(rows)
    other = smat[rows, :].toarray()
    assert np.all(row_arrs == other)


@csr_slow()
@given(st.data())
def test_csr_rows(data):
    "Test the row() method - row and multi_rows in one test (to demonstrate feasibility)"
    csr = data.draw(csrs(st.integers(1, 100), st.integers(1, 100)))
    smat = csr.to_scipy()

    row_id = st.integers(0, csr.nrows - 1)
    row_list = st.lists(row_id, max_size=csr.nrows, unique=True)
    rows = data.draw(st.one_of(row_id, row_list))

    row_arrs = csr.row(rows)
    other = smat[rows, :].toarray()
    assert np.all(row_arrs == other)


@csr_slow()
@given(st.data())
def test_csr_row_mask(data):
    csr = data.draw(csrs(st.integers(1, 100), st.integers(1, 100)))

    row_id = st.integers(0, csr.nrows - 1)
    row_list = st.lists(row_id, max_size=csr.nrows, unique=True)
    rows = data.draw(st.one_of(row_id, row_list))

    row_arrs = csr.row_mask(rows)
    assert row_arrs.dtype == np.bool_
    if isinstance(rows, list):
        assert row_arrs.shape == (len(rows), csr.ncols)
        for i, row in enumerate(rows):
            sp, ep = csr.row_extent(row)
            assert np.all(row_arrs[i, csr.row_cs(row)])
            assert np.sum(row_arrs[i, :]) == ep - sp
    else:
        assert row_arrs.shape == (csr.ncols,)
        sp, ep = csr.row_extent(rows)
        assert np.all(row_arrs[csr.row_cs(rows)])
        assert np.sum(row_arrs) == ep - sp


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


@csr_slow()
@given(csrs(st.integers(1, 100), st.integers(1, 100), values=True))
def test_drop_values(csr):
    csr.drop_values()
    assert csr.values is None


@csr_slow()
@given(st.data(), csrs())
def test_fill_values(data, csr):
    dtype = np.dtype('f8')
    if csr.values is not None:
        dtype = csr.values.dtype
    x = data.draw(nph.from_dtype(dtype, allow_infinity=False, allow_nan=False))
    csr.fill_values(x)
    assert all(csr.values == x)


def test_csr_set_values():
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_)

    csr = CSR.from_coo(rows, cols, vals)

    v2 = 10 - vals
    csr.values = v2

    assert all(csr.values == v2)


def test_csr_set_values_oversize():
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_)

    csr = CSR.from_coo(rows, cols, vals)

    v2 = np.arange(6, dtype=np.float_) + 10
    csr.values = v2

    assert csr.values is not None
    assert all(csr.values == v2[:4])


def test_csr_set_values_undersize():
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_)

    csr = CSR.from_coo(rows, cols, vals)

    v2 = np.arange(3, dtype=np.float_) + 5

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


@csr_slow()
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


@csr_slow()
@given(csrs(st.integers(1, 100), st.integers(1, 100)))
def test_copy(csr):
    "Test copying a CSR"
    c2 = csr.copy()

    assert c2.nrows == csr.nrows
    assert c2.ncols == csr.ncols
    assert c2.nnz == csr.nnz
    assert c2.rowptrs is not csr.rowptrs
    assert all(c2.rowptrs == csr.rowptrs)
    assert c2.colinds is not csr.colinds
    assert all(c2.colinds == csr.colinds)
    if csr.values is not None:
        assert c2.values is not csr.values
        assert all(c2.values == csr.values)
    else:
        assert c2.values is None


@csr_slow()
@given(csrs(st.integers(1, 100), st.integers(1, 100)))
def test_copy_share(csr):
    "Test copying a CSR and sharing structure"
    c2 = csr.copy(copy_structure=False)

    assert c2.nrows == csr.nrows
    assert c2.ncols == csr.ncols
    assert c2.nnz == csr.nnz
    assert c2.rowptrs is csr.rowptrs
    assert c2.colinds is csr.colinds
    if csr.values is not None:
        assert c2.values is not csr.values
        assert all(c2.values == csr.values)
    else:
        assert c2.values is None


@csr_slow()
@given(csrs(st.integers(1, 100), st.integers(1, 100)))
def test_copy_structure_only(csr):
    "Test copying only the structure of a CSV."
    c2 = csr.copy(False)

    assert c2.nrows == csr.nrows
    assert c2.ncols == csr.ncols
    assert c2.nnz == csr.nnz
    assert c2.rowptrs is not csr.rowptrs
    assert all(c2.rowptrs == csr.rowptrs)
    assert c2.colinds is not csr.colinds
    assert all(c2.colinds == csr.colinds)
    assert c2.values is None


@csr_slow()
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
