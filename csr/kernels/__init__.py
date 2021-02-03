from contextlib import contextmanager


@contextmanager
def releasing(h, k):
    try:
        yield h
    finally:
        k.release_handle(h)


def active_kernel():
    from . import numba
    return numba
