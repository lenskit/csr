"""
Compressed Sparse Row matrices for Python, with Numba API.
"""

__version__ = "0.5.0"
__all__ = [
    'CSR'
]

from .csr import CSR

# set up the Numba wiring!
from . import _wiring  # noqa: F401

from .constructors import *  # noqa: F401, F403
