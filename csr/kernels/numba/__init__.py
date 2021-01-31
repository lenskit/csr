"""
Kernel implementing matrix operations in pure Numba.
"""


from .multiply import mult_ab  # noqa: F401


def to_handle(csr):
    """
    Convert a native CSR to a handle.  The caller must arrange for the CSR last at
    least as long as the handle.  The handle must be explicitly released.
    """
    assert csr is not None
    return csr


def from_handle(h):
    """
    Convert a handle to a CSR.  The handle may be released after this is called.
    """

    assert h is not None
    return h


def release_handle(h):
    """
    Release a handle.
    """
    pass
