import numpy as np
import pickle

from csr import CSR
from csr.test_utils import csrs, csr_slow

import pytest
from hypothesis import given


@csr_slow()
@given(csrs())
def test_csr_pickle(csr):
    data = pickle.dumps(csr)
    csr2 = pickle.loads(data)

    assert csr2.nrows == csr.nrows
    assert csr2.ncols == csr.ncols
    assert csr2.nnz == csr.nnz
    assert all(csr2.rowptrs == csr.rowptrs)
    assert all(csr2.colinds == csr.colinds)
    if csr.values is not None:
        assert all(csr2.values == csr.values)
    else:
        assert csr2.values is None


@csr_slow()
@given(csrs())
def test_csr64_pickle(csr):
    csr = CSR(csr.nrows, csr.ncols, csr.nnz,
              csr.rowptrs.astype(np.int64), csr.colinds, csr.values,
              _cast=False)

    data = pickle.dumps(csr)
    csr2 = pickle.loads(data)

    assert csr2.nrows == csr.nrows
    assert csr2.ncols == csr.ncols
    assert csr2.nnz == csr.nnz
    assert all(csr2.rowptrs == csr.rowptrs)
    assert csr2.rowptrs.dtype == np.int64
    assert all(csr2.colinds == csr.colinds)
    if csr.values is not None:
        assert all(csr2.values == csr.values)
    else:
        assert csr2.values is None
