import numpy as np
from .handle import to_handle, from_handle, release_handle, order_columns  # noqa: F401
from .multiply import mult_ab, mult_abt, mult_vec  # noqa: F401

max_nnz = np.iinfo('i4').max
