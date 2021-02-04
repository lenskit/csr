from numba.core.typing.cffi_utils import register_module

from . import _mkl_ops

register_module(_mkl_ops)

from .convert import to_handle, from_handle, release_handle  # noqa: E402

__all__ = [
    'to_handle',
    'from_handle',
    'release_handle',
    'mult_ab',
    'mult_abt',
    'mult_vec',
    'available'
]
