import numpy as np


def upwind_gradient(phi, pos, u, axis):
    d = np.diff(pos)
    diff = np.diff(phi, axis=axis)
    shape = [1] * phi.ndim
    shape[axis] = d.shape[0]
    diff = diff / d.reshape(shape)

    first = [slice(None)] * phi.ndim
    first[axis] = slice(0, 1)
    rest = [slice(None)] * phi.ndim
    rest[axis] = slice(1, None)
    last = [slice(None)] * phi.ndim
    last[axis] = slice(-1, None)
    all_but_last = [slice(None)] * phi.ndim
    all_but_last[axis] = slice(0, -1)

    backward = np.empty_like(phi)
    backward[tuple(first)] = 0.0
    backward[tuple(rest)] = diff

    forward = np.empty_like(phi)
    forward[tuple(last)] = backward[tuple(last)]
    forward[tuple(all_but_last)] = diff

    return np.where(u >= 0.0, backward, forward)
