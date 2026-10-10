from formulas import minmod


def _slice_x(grid, offset):
    ng, nx = grid.ng, grid.nx
    return slice(ng + offset, ng + offset + nx)


def advection_rhs(grid, params):
    q = grid.q
    jc = grid.interior[0]
    dx, U = grid.dx, params.U

    q_im2 = q[:, jc, _slice_x(grid, -2)]
    q_im1 = q[:, jc, _slice_x(grid, -1)]
    q_i = q[:, jc, _slice_x(grid, 0)]
    q_ip1 = q[:, jc, _slice_x(grid, 1)]

    slope_i = minmod(q_i - q_im1, q_ip1 - q_i)
    slope_im1 = minmod(q_im1 - q_im2, q_i - q_im1)

    face_ip12 = q_i + 0.5 * slope_i
    face_im12 = q_im1 + 0.5 * slope_im1

    return -U * (face_ip12 - face_im12) / dx
