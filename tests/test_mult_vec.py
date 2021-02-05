import logging
import numpy as np

from csr import CSR
from csr.test_utils import sparse_matrices

from pytest import approx
from hypothesis import given, settings
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph

_log = logging.getLogger(__name__)


@settings(deadline=None)
@given(st.data())
def test_mult_vec(kernel, data):
    mat = data.draw(sparse_matrices((100, 100)))
    md = mat.toarray()
    csra = CSR.from_scipy(mat)
    # TODO make the test work with larger values
    vals = st.floats(-100, 100)
    v = data.draw(nph.arrays(np.float64, csra.ncols, elements=vals))

    prod = csra.mult_vec(v)
    assert prod.shape == (csra.nrows,)

    v2 = md @ v

    assert prod == approx(v2, nan_ok=True, rel=1.0e-5)


@settings(deadline=None)
@given(st.data())
def test_mult_vec_novalue(kernel, data):
    mat = data.draw(sparse_matrices())
    csra = CSR.from_scipy(mat, True)

    csra.drop_values()
    mat.data[:] = 1.0

    vals = st.floats(-100, 100)
    v = data.draw(nph.arrays(np.float64, csra.ncols, elements=vals))

    prod = csra.mult_vec(v)
    assert prod.shape == (csra.nrows,)

    v2 = mat @ v

    assert prod == approx(v2, nan_ok=True)
