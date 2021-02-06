"""
Default kernel interface for Numba.
"""

from csr.kernels import _default_kernel

_kernel = _default_kernel()

to_handle = _kernel.to_handle
from_handle = _kernel.from_handle
release_handle = _kernel.release_handle
mult_ab = _kernel.mult_ab
mult_abt = _kernel.mult_abt
mult_vec = _kernel.mult_vec
