import numpy as np
import numba
from near_finder.nufft_interp import periodic_interp1d

################################################################################
# Utilities for finding local coordinates.  These are not public-facing, and
# are intended to be called through the interface in coordinate_routines.py

@numba.njit(fastmath=True, inline='always')
def _guess_ind_finder(cx, cy, x, y, gi, d):
    _min = 1e300
    gi_min = (gi - d) #% cx.size
    gi_max = (gi + d) #% cx.size
    n = cx.size
    for i in range(gi_min, gi_max):
        # ih = i % n
        dx = cx[i%n] - x
        dy = cy[i%n] - y
        d = np.sqrt(dx*dx + dy*dy)
        if d < _min:
            _min = d
            argmin = i%n
    return argmin, _min
@numba.njit(fastmath=True, parallel=True)
def multi_guess_ind_finder(cx, cy, x, y, inds, mins, gis, ds):
    for j in numba.prange(x.size):
        inds[j], mins[j] = _guess_ind_finder(cx, cy, x[j], y[j], gis[j], ds[j])

@numba.njit(fastmath=True, inline='always')
def _expanded_guess_ind_finder(ecx, ecy, x, y, gi, d, md):
    _min = 1e300
    gi = gi + md
    gi_min = (gi - d)
    gi_max = (gi + d)
    for i in range(gi_min, gi_max):
        dx = ecx[i] - x
        dy = ecy[i] - y
        d = np.sqrt(dx*dx + dy*dy)
        if d < _min:
            _min = d
            argmin = i
    return (argmin - md) % (ecx.size-2*md), _min
@numba.njit(fastmath=True, parallel=True)
def multi_expanded_guess_ind_finder(ecx, ecy, x, y, inds, mins, gis, ds, md):
    for j in numba.prange(x.size):
        inds[j], mins[j] = _expanded_guess_ind_finder(ecx, ecy, x[j], y[j], gis[j], ds[j], md)

# new wrappers for general_tree
@numba.njit(fastmath=True, parallel=True)
def _gi_finder_same_d(ecx, ecy, x, y, inds, mins, gis, d, md):
    for j in numba.prange(x.size):
        inds[j], mins[j] = _expanded_guess_ind_finder(ecx, ecy, x[j], y[j], gis[j], d, md)
def gi_finder_same_d(ecx, ecy, x, y, gis, d, md):
    sh = x.shape
    x = x.ravel()
    y = y.ravel()
    gis = gis.ravel()
    inds = np.empty(x.size, dtype=int)
    minr = np.empty(x.size, dtype=float)
    _gi_finder_same_d(ecx, ecy, x, y, inds, minr, gis, d, md)
    return inds.reshape(sh), minr.reshape(sh)
def gi_finder_different_d(ecx, ecy, x, y, gis, ds, md):
    sh = x.shape
    x = x.ravel()
    y = y.ravel()
    gis = gis.ravel()
    inds = np.empty(x.size, dtype=int)
    minr = np.empty(x.size, dtype=float)
    multi_expanded_guess_ind_finder(ecx, ecy, x, y, inds, minr, gis, ds, md)
    return inds.reshape(sh), minr.reshape(sh)

@numba.njit(fastmath=True, inline='always')
def _full_guess_ind_finder(cx, cy, x, y):
    _min = 1e300
    for i in range(cx.size):
        dx = cx[i] - x
        dy = cy[i] - y
        d = np.sqrt(dx*dx + dy*dy)
        if d < _min:
            _min = d
            argmin = i
    return argmin
@numba.njit(fastmath=True, parallel=True)
def multi_full_guess_ind_finder(cx, cy, x, y, inds):
    for j in numba.prange(x.size):
        inds[j] = _full_guess_ind_finder(cx, cy, x[j], y[j])

def compute_local_coordinates(c_i, nc_i, x, y, guess_s, newton_tol, max_iterations, verbose):
    """
    Find using the coordinates:
    x = X(s) + r n_x(s)
    y = Y(s) + r n_y(s)

    c_i: Interpolater for boundary nodes (bdy.c)
    nc_i: Interpolater for normal to boundary (bdy.normal_c)
    x: x coordinate for points at which to interpolate
    y: y coordinates for points at which to interpolate
    guess_s: guess at s (must be provided!)
    newton_tol: newton tolerance
    max_iterations: raise error if convergence not reached after max_iterations
    verbose: how much output to show (True/False)
    """
    xshape = x.shape
    x = x.flatten()
    y = y.flatten()
    g = guess_s.flatten()

    # function for computing (d^2)_s and its derivative
    def f(t, x, y):
        out = c_i(t)
        C = out[0]
        Cp = out[1]
        Cpp = out[2]
        X = C.real
        Y = C.imag
        Xp = Cp.real
        Yp = Cp.imag
        Xpp = Cpp.real
        Ypp = Cpp.imag
        return Xp*(X-x) + Yp*(Y-y), X, Y, Xp, Yp, Xpp, Ypp
    def fp(t, x, y, X, Y, Xp, Yp, Xpp, Ypp):
        return Xpp*(X-x) + Ypp*(Y-y) + Xp*Xp + Yp*Yp

    def mprint(str):
        if verbose:
            print(str)

    # set t to starting guess
    t = g

    # begin Newton iteration
    rem, X, Y, Xp, Yp, Xpp, Ypp = f(t, x, y)
    mrem = np.abs(rem).max()
    mprint('Newton tol: {:0.2e}'.format(mrem))
    iteration = 0
    while mrem > newton_tol:
        J = fp(t, x, y, X, Y, Xp, Yp, Xpp, Ypp)
        delt = -rem/J
        line_factor = 1.0
        while True:
            t_new = t + line_factor*delt
            rem_new, X, Y, Xp, Yp, Xpp, Ypp = f(t_new, x, y)
            mrem_new = np.abs(rem_new).max()
            testit = True
            if testit and ((mrem_new < (1-0.5*line_factor)*mrem) or line_factor < 1e-4):
                t = t_new
                # put theta back in [0, 2 pi]
                t[t < 0] += 2*np.pi
                t[t > 2*np.pi] -= 2*np.pi
                rem = rem_new
                mrem = mrem_new
                break
            line_factor *= 0.5
        mprint('Newton tol: {:0.2e}'.format(mrem))
        iteration += 1
        if iteration > max_iterations:
            raise Exception('Exceeded maximum number of iterations solving for coordinates .')

    # need to determine the sign now
    C = c_i(t)[0]
    X = C.real
    Y = C.imag
    NC = nc_i(t)
    NX = NC.real
    NY = NC.imag
    r = np.hypot(X-x, Y-y)

    xe1 = X + r*NX
    ye1 = Y + r*NY
    err1 = np.hypot(xe1-x, ye1-y)
    xe2 = X - r*NX
    ye2 = Y - r*NY
    err2 = np.hypot(xe2-x, ye2-y)

    sign = (err1 < err2).astype(int)*2 - 1

    return t.reshape(xshape), (r*sign).reshape(xshape)
