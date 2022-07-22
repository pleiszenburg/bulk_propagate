import numpy as np
from numba import njit as jit


@jit
def rotation_matrix(angle, axis):
    """
    TODO
    """
    assert axis in (0, 1, 2)
    angle = np.asarray(angle)
    c = np.cos(angle)
    s = np.sin(angle)

    a1 = (axis + 1) % 3
    a2 = (axis + 2) % 3
    R = np.zeros(angle.shape + (3, 3))
    R[..., axis, axis] = 1.0
    R[..., a1, a1] = c
    R[..., a1, a2] = -s
    R[..., a2, a1] = s
    R[..., a2, a2] = c
    return R
