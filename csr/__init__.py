"""
Compressed Sparse Row matrices for Python, with Numba API.
"""

__version__ = "0.1.1"
__all__ = [
    'CSR'
]

from .csr import CSR

# set up the Numba wiring!
from . import _wiring  # noqa: F401
