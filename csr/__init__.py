"""
Compressed Sparse Row matrices for Python, with Numba API.
"""

__version__ = "0.1.0"
__all__ = [
    'CSR'
]

from .layout import _CSR  # noqa: F401
from .csr import CSR
