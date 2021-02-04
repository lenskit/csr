import numpy as np
import scipy.sparse as sps

from csr import CSR

from pytest import mark


@mark.benchmark(
    group='MultAx'
)
def test_mult_vec(kernel, benchmark):
    A = sps.random(100, 100, 0.1, format='csr')
    A = CSR.from_scipy(A)
    x = np.random.randn(100)

    # make sure it's compiled
    y = A.mult_vec(x)
    assert len(y) == A.nrows

    def op():
        A.mult_vec(x)

    benchmark(op)
