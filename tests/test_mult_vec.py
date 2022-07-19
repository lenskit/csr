import logging

from csr.test_utils import csrs, finite_arrays

from pytest import approx
from hypothesis import given, settings
import hypothesis.strategies as st

_log = logging.getLogger(__name__)


@settings(deadline=None)
@given(st.data(), csrs(values=True))
def test_mult_vec(kernel, data, csra):
    md = csra.to_scipy().toarray()
    # TODO make the test work with larger values
    v = data.draw(finite_arrays(csra.ncols))

    prod = csra.mult_vec(v)
    assert prod.shape == (csra.nrows,)

    v2 = md @ v

    assert prod == approx(v2, nan_ok=True, rel=1.0e-5, abs=1.0e-10)


@settings(deadline=None)
@given(st.data(), csrs(values=False))
def test_mult_vec_novalue(kernel, data, csra):
    mat = csra.to_scipy()

    v = data.draw(finite_arrays(csra.ncols))

    prod = csra.mult_vec(v)
    assert prod.shape == (csra.nrows,)

    v2 = mat @ v

    assert prod == approx(v2, nan_ok=True)
