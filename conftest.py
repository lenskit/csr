import logging

from hypothesis import settings, HealthCheck
from pytest import fixture, skip
from csr import CSR
from csr.kernels import use_kernel, get_kernel

# turn off Numba logging
logging.getLogger('numba').setLevel(logging.INFO)

KERNELS = ["scipy", "numba", "mkl"]
DISABLED_KERNELS = []
try:
    import csr.kernels.mkl  # noqa: F401
except ImportError:
    DISABLED_KERNELS.append("mkl")


# set up fixtures
@fixture(scope="module", params=KERNELS)
def kernel(request):
    """
    Fixture for variable CSR kernels.  This fixture is parameterized, so if you
    write a test function with a parameter ``kernel`` as its first parameter, it
    will be called once for each kernel under active test.
    """
    if request.param in DISABLED_KERNELS:
        skip(f'kernel {request.param} is disabled')

    with use_kernel(request.param):
        k = get_kernel()
        # warm-up the kernel
        m = CSR.empty(1, 1)
        h = k.to_handle(m)
        k.release_handle(h)
        del h, m
        yield k


# set up profiles
settings.register_profile('default', deadline=5000)
settings.register_profile('large', settings.get_profile('default'),
                          max_examples=5000, deadline=None)
settings.register_profile('fast', max_examples=50)
settings.register_profile('nojit', settings.get_profile('fast'),
                          deadline=None, suppress_health_check=HealthCheck.all())
settings.load_profile('default')
