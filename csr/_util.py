from numba import njit


@njit
def swap(a, i, j):
    a[i], a[j] = a[j], a[i]
