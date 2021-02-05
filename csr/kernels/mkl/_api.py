from types import BuiltinFunctionType
import numba.types as nt
from numba.core.typing.cffi_utils import register_module, register_type
from numba import njit

from . import _mkl_ops

register_type(_mkl_ops.ffi.typeof('intptr_t'), nt.intp)
register_module(_mkl_ops)

ffi = _mkl_ops.ffi

# export all the LK names
__all__ = ['ffi']
for name in dir(_mkl_ops.lib):
    f = getattr(_mkl_ops.lib, name)
    if isinstance(f, BuiltinFunctionType):
        globals()[name] = f
        __all__.append(name)
