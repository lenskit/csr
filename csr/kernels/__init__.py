import os
import warnings
from contextlib import contextmanager
from importlib import import_module
import threading

kernels = {}
__all__ = [
    'releasing',
    'set_kernel',
    'use_kernel',
    'get_kernel',
]


class ActiveKernel(threading.local):
    def __init__(self):
        self.__dict__.update({'active_name': None})

    @property
    def active(self):
        kern = getattr(self, '_active', None)
        if kern is None:
            return _default_kernel()
        else:
            return kern

    def set_active(self, kern):
        self._active = kern


__cached_default = None
__active = ActiveKernel()


@contextmanager
def releasing(h, k):
    try:
        yield h
    finally:
        k.release_handle(h)


def set_kernel(name):
    """
    Set the default kernel.  It is very rare to need to use this — letting CSR select
    its default kernel, or configuring the kernel through the ``CSR_KERNEL`` environment
    variable, is the best option for the vast majority of applications.  This is here
    primarily to enable test code to switch kernels.

    This function does **not** change the kernel exposed by importing ``csr.kernel``,
    which is typically used by compiled Numba functions.  It only changes the kernel
    returned by :py:func:`get_kernel` and used by the pure-Python APIs,

    Args:
        name(str):
            The name of the kernel.
    """

    if name is None:
        __active.set_active(None)
    else:
        __active.set_active(get_kernel(name))


@contextmanager
def use_kernel(name):
    """
    Context manager to run code with a specified (thread-local) kernel.  It calls
    :py:func:`set_kernel`, and restores the previously-active kernel when the context
    exits.
    """
    old = __active.active_name
    try:
        set_kernel(name)
        yield
    finally:
        set_kernel(old)


def get_kernel(name=None):
    """
    Get a kernel.

    Args:
        name(str or None):
            The name of the kernel.  If ``None``, returns the current default kernel.
    """
    if name is None:
        return __active.active

    kern = kernels.get(name, None)
    if not kern:
        mod_name = f'{__name__}.{name}'
        kern = import_module(mod_name)
        kernels[name] = kern
    return kern


def _initialize(name=None):
    global __cached_default
    if __cached_default:
        warnings.warn('default kernel already initialized')

    if name:
        k = import_module(f'csr.kernels.{name}')
    elif 'CSR_KERNEL' in os.environ:
        _env_kernel = os.environ['CSR_KERNEL']
        k = import_module(f'csr.kernels.{_env_kernel}')
    else:
        try:
            import csr.kernels.mkl as k
        except ImportError:
            import csr.kernels.numba as k

    __cached_default = k


def _default_kernel():
    if not __cached_default:
        _initialize()

    return __cached_default
