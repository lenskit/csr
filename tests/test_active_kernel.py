import pytest
from csr.kernels import _default_kernel, get_kernel, use_kernel
default_kernel = _default_kernel()


def test_get_default():
    k = get_kernel()
    assert k is not None
    assert k is default_kernel

    # make sure the default is right
    try:
        import csr.kernels.mkl  # noqa: F401
        assert k.__name__ == 'csr.kernels.mkl'
    except ImportError:
        assert k.__name__ == 'csr.kernels.numba'


def test_get_scipy():
    k = get_kernel('scipy')
    assert k.__name__ == 'csr.kernels.scipy'


def test_get_numba():
    k = get_kernel('numba')
    assert k.__name__ == 'csr.kernels.numba'


def test_get_mkl():
    try:
        import csr.kernels.mkl  # noqa: F401
    except ImportError:
        pytest.skip("kernel MKL is disabled")

    k = get_kernel('mkl')
    assert k.__name__ == 'csr.kernels.mkl'


def test_with_scipy():
    with use_kernel('scipy'):
        k = get_kernel()
        assert k.__name__ == 'csr.kernels.scipy'

    k = get_kernel()
    assert k is default_kernel
