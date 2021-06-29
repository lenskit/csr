import numpy as np
from .handle import to_handle, from_handle, release_handle, order_columns
from .multiply import mult_ab, mult_abt, mult_vec

max_nnz = np.iinfo('i4').max
