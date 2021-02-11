from numba.core import types
from numba.extending import overload


def maybe_swap(a, i, j):
    """
    Swap the elements in two positions in an array.
    In the special case that the array is ``None``, it does nothing.
    """
    if a is not None:
        a[i], a[j] = a[j], a[i]


@overload(maybe_swap)
def _ovl_swap(a, i, j):
    def swap(a, i, j):
        a[i], a[j] = a[j], a[i]

    def nothing(a, i, j):
        pass

    if isinstance(a, types.NoneType):
        return nothing
    else:
        return swap
