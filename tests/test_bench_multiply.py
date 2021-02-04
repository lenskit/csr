import numpy as np
import scipy.sparse as sps

from csr import CSR

from pytest import mark


@mark.benchmark(
    group='MultAB',
    warmup=True,
    warmup_iterations=1
)
def test_mult_ab(kernel, benchmark):
    A = sps.random(100, 500, 0.1, format='csr')
    B = sps.random(500, 200, 0.2, format='csr')
    A = CSR.from_scipy(A)
    B = CSR.from_scipy(B)

    def op():
        A.multiply(B)

    benchmark(op)


@mark.parametrize('density', np.linspace(0, 1, 10))
@mark.benchmark(
    group='MultAB-Density',
    warmup=True,
    warmup_iterations=1
)
def test_mult_ab_by_density(kernel, benchmark, density):
    A = sps.random(100, 100, density, format='csr')
    B = sps.random(100, 100, density, format='csr')
    A = CSR.from_scipy(A)
    B = CSR.from_scipy(B)

    def op():
        A.multiply(B)

    benchmark(op)
