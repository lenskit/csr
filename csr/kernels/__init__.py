from contextlib import contextmanager
from importlib import import_module
import threading
from . import numba as default_kernel

kernels = {}


class ActiveKernel(threading.local):
    def __init__(self):
        self.__dict__.update({'active_name': None})

    @property
    def active(self):
        kern = getattr(self, '_active', None)
        if kern is None:
            return default_kernel
        else:
            return kern

    def set_active(self, kern):
        self._active = kern


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
