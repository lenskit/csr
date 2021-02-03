import numpy as np
import scipy.sparse as sps

from csr import CSR

from pytest import mark

def op_sps_mult(A, B):
    A @ B
op_sps_mult.prepare = lambda x: x

def op_csr_mult(A, B):
    A.multiply(B)
op_csr_mult.prepare = CSR.from_scipy


@mark.parametrize('impl', ['csr', 'sps'])
def test_matrix_mult(benchmark, impl):
    op = globals()[f'op_{impl}_mult']
    A = sps.random(100, 500, 0.1, format='csr')
    B = sps.random(500, 200, 0.2, format='csr')
    A = op.prepare(A)
    B = op.prepare(B)

    benchmark(op, A, B)
