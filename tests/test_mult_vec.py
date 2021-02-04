import logging
import numpy as np

from csr import CSR
from csr.test_utils import sparse_matrices

from hypothesis import given, settings
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph

_log = logging.getLogger(__name__)


@settings(deadline=None)
@given(st.data())
def test_mult_vec(kernel, data):
    mat = data.draw(sparse_matrices())
    csra = CSR.from_scipy(mat)
    v = data.draw(nph.arrays(np.float64, csra.ncols))

    prod = csra.mult_vec(v)
    assert prod.shape == (csra.nrows,)

    v2 = mat @ v

    np.testing.assert_equal(prod, v2)
