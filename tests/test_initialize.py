from csr import CSR
import numpy as np

from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph


@given(st.integers(0, 1000), st.integers(0, 1000))
def test_empty(nrows, ncols):
    csr = CSR.empty(nrows, ncols)
    assert csr.nrows == nrows
    assert csr.ncols == ncols
    assert csr.nnz == 0
    assert all(csr.rowptrs == 0)
    assert len(csr.rowptrs) == nrows + 1
    assert len(csr.colinds) == 0


@given(st.data(), st.integers(0, 1000), st.integers(0, 1000))
def test_uninitialized(data, nrows, ncols):
    sizes = data.draw(nph.arrays(np.int32, nrows, elements=st.integers(0, ncols)))
    csr = CSR.empty(nrows, ncols, sizes)
    assert csr.nrows == nrows
    assert csr.ncols == ncols
    assert csr.nnz == np.sum(sizes)
    assert len(csr.rowptrs) == nrows + 1
    assert all(csr.row_nnzs() == sizes)
    assert len(csr.colinds) == np.sum(sizes)
