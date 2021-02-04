from csr.kernels import default_kernel, get_kernel, use_kernel


def test_get_default():
    k = get_kernel()
    assert k is not None
    assert k is default_kernel


def test_get_scipy():
    k = get_kernel('scipy')
    assert k.__name__ == 'csr.kernels.scipy'


def test_get_numba():
    k = get_kernel('numba')
    assert k.__name__ == 'csr.kernels.numba'


def test_with_scipy():
    with use_kernel('scipy'):
        k = get_kernel()
        assert k.__name__ == 'csr.kernels.scipy'

    k = get_kernel()
    assert k is default_kernel
