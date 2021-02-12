import logging

from hypothesis import settings
from pytest import fixture
from csr import CSR
from csr.kernels import use_kernel, get_kernel

# turn off Numba logging
logging.getLogger('numba').setLevel(logging.INFO)

KERNELS = ["scipy", "numba"]
try:
    import csr.kernels.mkl  # noqa: F401
    KERNELS.append("mkl")
except ImportError:
    pass  # no MKL available


# set up fixtures
@fixture(scope="module", params=KERNELS)
def kernel(request):
    """
    Fixture for variable CSR kernels.  This fixture is parameterized, so if you
    write a test function with a parameter ``kernel`` as its first parameter, it
    will be called once for each kernel under active test.
    """
    with use_kernel(request.param):
        k = get_kernel()
        # warm-up the kernel
        m = CSR.empty(1, 1)
        h = k.to_handle(m)
        k.release_handle(h)
        del h, m
        yield k


# set up profiles
settings.register_profile('default', deadline=1000)
settings.register_profile('large', max_examples=5000)
settings.register_profile('fast', max_examples=10)
settings.register_profile('nojit', settings.get_profile('fast'),
                          deadline=None)
settings.load_profile('default')
