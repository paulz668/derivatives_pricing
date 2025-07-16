import numpy as np
import scipy.linalg as la

from _collections_abc import Callable
from numpy.typing import ArrayLike

from scipy.sparse import diags_array
from scipy.sparse.linalg import spsolve_triangular, splu


def theta_method_pde_solver(
    state_max: np.floating,
    time_max: np.floating,
    M: np.int_,
    N: np.int_,
    a_coeff: Callable[[ArrayLike], ArrayLike],
    b_coeff: Callable[[ArrayLike], ArrayLike],
    c_coeff: Callable[[ArrayLike], ArrayLike],
    t_cond: Callable[[ArrayLike, ArrayLike], ArrayLike],
    state_min: np.floating = 0,
    time_min: np.floating = 0,
    theta: np.floating = 0.5,
):
    """
    Numerically solve a degenerate linear parabolic PDE of the form:
    g_t(S, t) + a(S) g_SS(S, t) + b(S) g_S(S, t) + c(S) g(S, t) = 0
    for S, t in R+, with some boundary condition g(S, T) and a(0, t) = 0
    using the scheme outlined in Chapter 22.4.2 of Wilmott, Dewynne and Howison (1993)

    state_max: maximum value of the state variable in the grid
    time_max: maximum value of the time variable in the grid
    M: number of steps in the state variable
    N: number of steps in the time variable
    a_coeff: function for a(S) coefficient (diffusion term)
    b_coeff: function for b(S) coefficient (convection term)
    c_coeff: function for c(S) coefficient (reaction term)
    t_cond: terminal condition g(S, T)
    theta: theta parameter (0=explicit, 1=implicit, 0.5=Crank-Nicolson)
    """

    # Create state space and time grid
    dstate = (state_max - state_min) / M
    state = np.linspace(state_min, state_max, M + 1)

    dtime = (time_max - time_min) / N
    time = np.linspace(time_min, time_max, N + 1)

    # Initialise solutions array and set terminal condition
    g = np.zeros((M + 1, N + 1))
    g[:, -1] = t_cond(state, time_max)

    # Construct coeffcient matrices for the finite-difference approximation
    A = a_coeff(state[1:]) * theta / dstate**2
    B = (
        -1 / dtime
        - (2 * a_coeff(state) / dstate**2 + 3 * b_coeff(state) / (2 * dstate)) * theta
        + c_coeff(state)
    )
    C = (a_coeff(state[:-1]) / dstate**2 + 2 * b_coeff(state[:-1]) / dstate) * theta
    D = -b_coeff(state[:-2]) / (2 * dstate) * theta

    a = a_coeff(state[1:]) * (1 - theta) / dstate**2
    b = -1 / dtime + (
        2 * a_coeff(state) / dstate**2 + 3 * b_coeff(state) / (2 * dstate)
    ) * (1 - theta)
    c = -(a_coeff(state[:-1]) / dstate**2 + 2 * b_coeff(state[:-1]) / dstate) * (
        1 - theta
    )
    d = b_coeff(state[:-2]) / (2 * dstate) * (1 - theta)

    coeff_matrix_current = diags_array(
        (A, B, C, D), offsets=(-1, 0, 1, 2), format="csc"
    )
    coeff_matrix_next = diags_array((a, b, c, d), offsets=(-1, 0, 1, 2), format="csc")

    for i in range(N, -1, -1):
        target = coeff_matrix_next @ g[: i + 1]
        lu_decomp = splu(coeff_matrix_current)
        g[:i] = lu_decomp.solve(target)

    return g
