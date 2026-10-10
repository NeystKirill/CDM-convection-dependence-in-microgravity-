from emulator.reference.reaction import react_step as _react_step_flat


def react_step(T, Yf, Yo, params, dt_half):
    shape = T.shape
    T_new, Yf_new, Yo_new = _react_step_flat(T.ravel(), Yf.ravel(), Yo.ravel(), params, dt_half)
    return T_new.reshape(shape), Yf_new.reshape(shape), Yo_new.reshape(shape)
