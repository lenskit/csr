import logging

from hypothesis import settings
from pytest import fixture
from csr.kernels import use_kernel, get_kernel

# turn off Numba logging
logging.getLogger('numba').setLevel(logging.INFO)


# set up fixtures
@fixture(scope="module", params=["scipy", "numba"])
def kernel(request):
    """
    Fixture for variable CSR kernels.  This fixture is parameterized, so if you
    write a test function with a parameter ``kernel`` as its first parameter, it
    will be called once for each kernel under active test.
    """
    with use_kernel(request.param):
        yield get_kernel()


# set up profiles
settings.register_profile('large', max_examples=5000)
settings.register_profile('nojit', max_examples=10)
