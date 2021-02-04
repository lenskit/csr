import numpy as np
import scipy.sparse as sps

from csr import CSR

from pytest import mark


@mark.benchmark(
    group='MultAx',
    warmup=True,
    warmup_iterations=5
)
def test_mult_vec(kernel, benchmark):
    A = sps.random(100, 100, 0.1, format='csr')
    A = CSR.from_scipy(A)
    x = np.random.randn(100)

    def op():
        A.mult_vec(x)

    benchmark(op)
