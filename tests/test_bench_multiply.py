import numpy as np
import scipy.sparse as sps

from csr import CSR

from pytest import mark


@mark.benchmark(
    group='MultAB'
)
def test_mult_ab(kernel, benchmark):
    A = sps.random(100, 500, 0.1, format='csr')
    B = sps.random(500, 200, 0.2, format='csr')
    A = CSR.from_scipy(A)
    B = CSR.from_scipy(B)

    # make sure it's compiled
    A.multiply(B)

    def op():
        A.multiply(B)

    benchmark(op)


@mark.parametrize('density', np.linspace(0, 1, 11))
@mark.benchmark(
    group='MultAB-Density'
)
def test_mult_ab_by_density(kernel, benchmark, density):
    A = sps.random(100, 100, density, format='csr')
    B = sps.random(100, 100, density, format='csr')
    A = CSR.from_scipy(A)
    B = CSR.from_scipy(B)

    # make sure it's compiled
    A.multiply(B)

    def op():
        A.multiply(B)

    benchmark(op)


@mark.parametrize('size', [5, 10, 15, 25, 50, 100, 200, 250, 500, 750, 1000])
@mark.benchmark(
    group='MultAB-Size',
    max_time=10
)
def test_mult_ab_by_size(kernel, benchmark, size):
    A = sps.random(size, size, 0.1, format='csr')
    B = sps.random(size, size, 0.1, format='csr')
    A = CSR.from_scipy(A)
    B = CSR.from_scipy(B)

    # make sure it's compiled
    A.multiply(B)

    def op():
        A.multiply(B)

    benchmark(op)


@mark.benchmark(
    group='MultABt'
)
def test_mult_abt(kernel, benchmark):
    A = sps.random(100, 500, 0.1, format='csr')
    B = sps.random(200, 500, 0.2, format='csr')
    A = CSR.from_scipy(A)
    B = CSR.from_scipy(B)

    # make sure it's compiled
    A.multiply(B, transpose=True)

    def op():
        A.multiply(B, transpose=True)

    benchmark(op)


@mark.parametrize('density', np.linspace(0, 1, 11))
@mark.benchmark(
    group='MultABt-Density',
)
def test_mult_abt_by_density(kernel, benchmark, density):
    A = sps.random(100, 100, density, format='csr')
    B = sps.random(100, 100, density, format='csr')
    A = CSR.from_scipy(A)
    B = CSR.from_scipy(B)

    # make sure it's compiled
    A.multiply(B, transpose=True)

    def op():
        A.multiply(B, transpose=True)

    benchmark(op)
