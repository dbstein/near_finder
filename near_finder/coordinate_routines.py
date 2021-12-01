import numpy as np
import numba
import scipy
import scipy.interpolate
from near_finder.utilities import fourier_derivative_1d
from near_finder.utilities import interp_fourier as _interp
from near_finder.utilities import have_better_fourier
if have_better_fourier:
    from near_finder.utilities import interp_fourier2 as _interp2
from near_finder.nufft_interp import periodic_interp1d

def compute_local_coordinates(cx, cy, x, y, newton_tol=1e-12,
            interpolation_scheme='nufft', guess_ind=None, verbose=False, max_iterations=30):
    """
    Find (s, r) given (x, y) using the coordinates:
    x = X(s) + r n_x(s)
    y = Y(s) + r n_y(s)

    Where X, Y is given by cx, cy
    Uses a NUFFT based scheme if interpolation_scheme = 'nufft'
    And a polynomail based scheme if interpolation_scheme = 'polyi'

    The NUFFT scheme is more accurate and robust;
    The polynomial based scheme is more accurate
    """
    if interpolation_scheme == 'nufft':
        return compute_local_coordinates_nufft(cx, cy, x, y, newton_tol, guess_ind, verbose, max_iterations)
    elif interpolation_scheme == 'polyi':
        return compute_local_coordinates_polyi(cx, cy, x, y, newton_tol, guess_ind, verbose, max_iterations)
    else:
        raise Exception('interpolation_scheme not recognized')    

@numba.njit(fastmath=True)
def _guess_ind_finder_centering(cx, cy, x, y, gi, d):
    _min = 1e300
    gi_min = (gi - d) % cx.size
    gi_max = (gi + d) % cx.size
    for i in range(gi_min, gi_max):
        dx = cx[i] - x
        dy = cy[i] - y
        d = np.sqrt(dx*dx + dy*dy)
        if d < _min:
            _min = d
            argmin = i
    return argmin
@numba.njit(fastmath=True, parallel=True)
def multi_guess_ind_finder_centering(cx, cy, x, y, inds, gi, d):
    for j in numba.prange(x.size):
        inds[j] = _guess_ind_finder_centering(cx, cy, x[j], y[j], gi[j], d)

def compute_local_coordinates_nufft_centering_with_interpolaters(cx, cy, x, y, gi, nc_i, c_i, newton_tol=1e-12, 
                                                verbose=False, max_iterations=30):
    """
    Find using the coordinates:
    x = X + r n_x
    y = Y + r n_y
    """
    xshape = x.shape
    yshape = y.shape
    x = x.flatten()
    y = y.flatten()
    gi = gi.flatten()

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

    guess_ind = np.empty(x.size, dtype=int)
    multi_guess_ind_finder_centering(cx, cy, x, y, guess_ind, gi, 10)

    # get starting guess
    t = 2*np.pi/cx.size * guess_ind

    # begin Newton iteration
    rem, X, Y, Xp, Yp, Xpp, Ypp = f(t, x, y)
    mrem = np.abs(rem).max()
    if verbose:
        print('Newton tol: {:0.2e}'.format(mrem))
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
        if verbose:
            print('Newton tol: {:0.2e}'.format(mrem))
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

    return t, r*sign

def compute_local_coordinates_nufft_centering(cx, cy, x, y, gi, newton_tol=1e-12, 
                                                verbose=False, max_iterations=30):
    """
    Find using the coordinates:
    x = X + r n_x
    y = Y + r n_y
    """
    xshape = x.shape
    yshape = y.shape
    x = x.flatten()
    y = y.flatten()
    gi = gi.flatten()

    n = cx.shape[0]
    dt = 2*np.pi/n
    ts = np.arange(n)*dt
    ik = 1j*np.fft.fftfreq(n, dt/(2*np.pi))
    # tangent vectors
    xp = fourier_derivative_1d(f=cx, d=1, ik=ik, out='f')
    yp = fourier_derivative_1d(f=cy, d=1, ik=ik, out='f')
    xpp = fourier_derivative_1d(f=cx, d=2, ik=ik, out='f')
    ypp = fourier_derivative_1d(f=cy, d=2, ik=ik, out='f')
    # speed
    sp = np.sqrt(xp*xp + yp*yp)
    isp = 1.0/sp
    # unit tangent vectors
    tx = xp*isp
    ty = yp*isp
    # unit normal vectors
    nx = ty
    ny = -tx
    # interpolation routines for the necessary objects
    if have_better_fourier:
        def interp(f):
            return _interp2(f)
    else:
        def interp(f):
            return _interp(f, x.size)
    nc_i = interp(nx + 1j*ny)
    c_i = interp(cx + 1j*cy)
    cp_i = interp(xp + 1j*yp)
    cpp_i = interp(xpp + 1j*ypp)

    # function for computing (d^2)_s and its derivative
    def f(t, x, y):
        C = c_i(t)
        X = C.real
        Y = C.imag
        Cp = cp_i(t)
        Xp = Cp.real
        Yp = Cp.imag
        return Xp*(X-x) + Yp*(Y-y), X, Y, Xp, Yp
    def fp(t, x, y, X, Y, Xp, Yp):
        Cpp = cpp_i(t)
        Xpp = Cpp.real
        Ypp = Cpp.imag
        return Xpp*(X-x) + Ypp*(Y-y) + Xp*Xp + Yp*Yp

    guess_ind = np.empty(x.size, dtype=int)
    multi_guess_ind_finder_centering(cx, cy, x, y, guess_ind, gi, 10)

    # get starting guess
    t = ts[guess_ind]

    # begin Newton iteration
    rem, X, Y, Xp, Yp = f(t, x, y)
    mrem = np.abs(rem).max()
    if verbose:
        print('Newton tol: {:0.2e}'.format(mrem))
    iteration = 0
    while mrem > newton_tol:
        J = fp(t, x, y, X, Y, Xp, Yp)
        delt = -rem/J
        line_factor = 1.0
        while True:
            t_new = t + line_factor*delt
            rem_new, X, Y, Xp, Yp = f(t_new, x, y)
            mrem_new = np.abs(rem_new).max()
            # try:
            #     rem_new, X, Y, Xp, Yp = f(t_new, x, y)
            #     mrem_new = np.abs(rem_new).max()
            #     testit = True
            # except:
            #     testit = False
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
        if verbose:
            print('Newton tol: {:0.2e}'.format(mrem))
        iteration += 1
        if iteration > max_iterations:
            raise Exception('Exceeded maximum number of iterations solving for coordinates .')

    # need to determine the sign now
    C = c_i(t)
    X = C.real
    Y = C.imag
    if True: # use nx, ny from guess_inds to determine sign
        NX = nx[guess_ind]
        NY = ny[guess_ind]
    else: # interpolate to get these
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

    return t, r*sign


def compute_local_coordinates_nufft(cx, cy, x, y, newton_tol=1e-12, 
                                            guess_ind=None, verbose=False, max_iterations=30):
    """
    Find using the coordinates:
    x = X + r n_x
    y = Y + r n_y
    """
    xshape = x.shape
    yshape = y.shape
    x = x.flatten()
    y = y.flatten()

    n = cx.shape[0]
    dt = 2*np.pi/n
    ts = np.arange(n)*dt
    ik = 1j*np.fft.fftfreq(n, dt/(2*np.pi))
    # tangent vectors
    xp = fourier_derivative_1d(f=cx, d=1, ik=ik, out='f')
    yp = fourier_derivative_1d(f=cy, d=1, ik=ik, out='f')
    xpp = fourier_derivative_1d(f=cx, d=2, ik=ik, out='f')
    ypp = fourier_derivative_1d(f=cy, d=2, ik=ik, out='f')
    # speed
    sp = np.sqrt(xp*xp + yp*yp)
    isp = 1.0/sp
    # unit tangent vectors
    tx = xp*isp
    ty = yp*isp
    # unit normal vectors
    nx = ty
    ny = -tx
    # interpolation routines for the necessary objects
    if have_better_fourier:
        def interp(f):
            return _interp2(f)
    else:
        def interp(f):
            return _interp(f, x.size)
    nc_i = interp(nx + 1j*ny)
    c_i = interp(cx + 1j*cy)
    cp_i = interp(xp + 1j*yp)
    cpp_i = interp(xpp + 1j*ypp)

    # function for computing (d^2)_s and its derivative
    def f(t, x, y):
        C = c_i(t)
        X = C.real
        Y = C.imag
        Cp = cp_i(t)
        Xp = Cp.real
        Yp = Cp.imag
        return Xp*(X-x) + Yp*(Y-y), X, Y, Xp, Yp
    def fp(t, x, y, X, Y, Xp, Yp):
        Cpp = cpp_i(t)
        Xpp = Cpp.real
        Ypp = Cpp.imag
        return Xpp*(X-x) + Ypp*(Y-y) + Xp*Xp + Yp*Yp

    # brute force find of guess_inds if not provided (slow!)
    if guess_ind is None:
        guess_ind = np.empty(x.size, dtype=int)
        multi_guess_ind_finder(cx, cy, x, y, guess_ind)

    # get starting guess
    t = ts[guess_ind]

    # begin Newton iteration
    rem, X, Y, Xp, Yp = f(t, x, y)
    mrem = np.abs(rem).max()
    if verbose:
        print('Newton tol: {:0.2e}'.format(mrem))
    iteration = 0
    while mrem > newton_tol:
        J = fp(t, x, y, X, Y, Xp, Yp)
        delt = -rem/J
        line_factor = 1.0
        while True:
            t_new = t + line_factor*delt
            rem_new, X, Y, Xp, Yp = f(t_new, x, y)
            mrem_new = np.abs(rem_new).max()
            # try:
            #     rem_new, X, Y, Xp, Yp = f(t_new, x, y)
            #     mrem_new = np.abs(rem_new).max()
            #     testit = True
            # except:
            #     testit = False
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
        if verbose:
            print('Newton tol: {:0.2e}'.format(mrem))
        iteration += 1
        if iteration > max_iterations:
            raise Exception('Exceeded maximum number of iterations solving for coordinates .')

    # need to determine the sign now
    C = c_i(t)
    X = C.real
    Y = C.imag
    if True: # use nx, ny from guess_inds to determine sign
        NX = nx[guess_ind]
        NY = ny[guess_ind]
    else: # interpolate to get these
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

    return t, r*sign

snodes = np.linspace(-1, 1, 6)
MAT = np.empty([6,6], dtype=float)
for i in range(6):
    MAT[:,i] = np.power(snodes,i)
IMAT = np.linalg.inv(MAT)

@numba.njit(fastmath=True)
def _polyi(f, xx):
    n = f.size
    h = 2*np.pi/n
    ix = int(xx//h)
    lb = (ix-2)*h
    ub = (ix+3)*h
    scalex = 2*((xx-lb)/(ub-lb) - 0.5)
    p2 = scalex*scalex
    p3 = p2*scalex
    p4 = p3*scalex
    p5 = p4*scalex

    fout = 0.0
    for j in range(6):
        fout += IMAT[0, j]*f[(ix+j-2) % n]
    coef = 0.0
    for j in range(6):
        coef += IMAT[1, j]*f[(ix+j-2) % n]
    fout += coef*scalex
    coef = 0.0
    for j in range(6):
        coef += IMAT[2, j]*f[(ix+j-2) % n]
    fout += coef*p2
    coef = 0.0
    for j in range(6):
        coef += IMAT[3, j]*f[(ix+j-2) % n]
    fout += coef*p3
    coef = 0.0
    for j in range(6):
        coef += IMAT[4, j]*f[(ix+j-2) % n]
    fout += coef*p4
    coef = 0.0
    for j in range(6):
        coef += IMAT[5, j]*f[(ix+j-2) % n]
    fout += coef*p5

    return fout

@numba.njit(fastmath=True)
def _polyi(f, xx):
    n = f.size
    h = 2*np.pi/n
    ix = int(xx//h)
    lb = (ix-2)*h
    ub = (ix+3)*h
    scalex = 2*((xx-lb)/(ub-lb) - 0.5)

    F = np.empty(6, np.float64)
    for j in range(6):
        F[j] = f[(ix+j-2) % n]

    A = IMAT.dot(F)
    return A[0] + A[1]*scalex + A[2]*scalex**2 + A[3]*scalex**3 + A[4]*scalex**4 + A[5]*scalex**5

@numba.njit(fastmath=True)
def _polyi_p(f, xx):
    n = f.size
    h = 2*np.pi/n
    ix = int(xx//h)
    lb = (ix-2)*h
    ub = (ix+3)*h
    scalex = 2*((xx-lb)/(ub-lb) - 0.5)
    dscale = 2.0/(ub-lb)

    F = np.empty(6, np.float64)
    for j in range(6):
        F[j] = f[(ix+j-2) % n]

    A = IMAT.dot(F)
    # calculate value
    O0 = A[0] + A[1]*scalex + A[2]*scalex**2 + A[3]*scalex**3 + A[4]*scalex**4 + A[5]*scalex**5
    # calculate derivative
    O1 = A[1] + 2*A[2]*scalex + 3*A[3]*scalex**2 + 4*A[4]*scalex**3 + 5*A[5]*scalex**4
    return O0, O1*dscale

@numba.njit(fastmath=True)
def _polyi_pp(f, xx):
    n = f.size
    h = 2*np.pi/n
    ix = int(xx//h)
    lb = (ix-2)*h
    ub = (ix+3)*h
    scalex = 2*((xx-lb)/(ub-lb) - 0.5)
    dscale = 2.0/(ub-lb)

    F = np.empty(6, np.float64)
    for j in range(6):
        F[j] = f[(ix+j-2) % n]

    A = IMAT.dot(F)
    # calculate value
    O0 = A[0] + A[1]*scalex + A[2]*scalex**2 + A[3]*scalex**3 + A[4]*scalex**4 + A[5]*scalex**5
    # calculate derivative
    O1 = A[1] + 2*A[2]*scalex + 3*A[3]*scalex**2 + 4*A[4]*scalex**3 + 5*A[5]*scalex**4
    # calculate second derivative
    O2 = 2*A[2] + 6*A[3]*scalex + 12*A[4]*scalex**2 + 20*A[5]*scalex**3
    return O0, O1*dscale, O2*dscale*dscale

# @numba.njit(fastmath=True)
# def _polyi_(f, xx):
#     n = f.size
#     h = 2*np.pi/n
#     ix = int(xx//h)
#     ratx = xx/h - (ix+0.5)

#     fout  = f[(ix - 2) % n] * (3/256 + ratx*(   -9/1920 + ratx*( -5/48/2 + ratx*(  1/8/6 + ratx*( 1/2/24 -  1/8/120*ratx)))))
#     fout += f[(ix - 1) % n] * (-25/256 + ratx*(  125/1920 + ratx*( 39/48/2 + ratx*(-13/8/6 + ratx*(-3/2/24 +  5/8/120*ratx)))))
#     fout += f[(ix    ) % n] * (150/256 + ratx*(-2250/1920 + ratx*(-34/48/2 + ratx*( 34/8/6 + ratx*( 2/2/24 - 10/8/120*ratx)))))
#     fout += f[(ix + 1) % n] * (150/256 + ratx*( 2250/1920 + ratx*(-34/48/2 + ratx*(-34/8/6 + ratx*( 2/2/24 + 10/8/120*ratx)))))
#     fout += f[(ix + 2) % n] * (-25/256 + ratx*( -125/1920 + ratx*( 39/48/2 + ratx*( 13/8/6 + ratx*(-3/2/24 -  5/8/120*ratx)))))
#     fout += f[(ix + 3) % n] * (3/256 + ratx*(    9/1920 + ratx*( -5/48/2 + ratx*( -1/8/6 + ratx*( 1/2/24 +  1/8/120*ratx)))))
#     return fout
@numba.njit(parallel=True, fastmath=True)
def multi_polyi(f, xx, out):
    for i in numba.prange(xx.size):
        out[i] = _polyi(f, xx[i])
@numba.njit(parallel=True, fastmath=True)
def multi_polyi_p(f, xx, out, outp):
    for i in numba.prange(xx.size):
        out[i], outp[i] = _polyi_p(f, xx[i])
@numba.njit(fastmath=True)
def _f(t, _cx, _cy, x, y):
    cx, cxp = _polyi_p(_cx, t)
    cy, cyp = _polyi_p(_cy, t)
    return cxp*(cx-x) + cyp*(cy-y)
@numba.njit(fastmath=True)
def _f_fp(t, _cx, _cy, x, y):
    cx, cxp, cxpp = _polyi_pp(_cx, t)
    cy, cyp, cypp = _polyi_pp(_cy, t)
    f = cxp*(cx-x) + cyp*(cy-y)
    fp = cxpp*(cx-x) + cypp*(cy-y) + cxp*cxp + cyp*cyp
    return f, fp
@numba.njit(fastmath=True)
def _newton(t, xi, yi, newton_tol, _cx, _cy, verbose, maxi):
    # get initial residual
    rem, fp = _f_fp(t, _cx, _cy, xi, yi)

    # Newton iteration
    iteration = 0
    while np.abs(rem) > newton_tol:
        # compute step
        t = t -rem/fp
        rem, fp = _f_fp(t, _cx, _cy, xi, yi)
        iteration += 1
        if iteration > maxi:
            raise Exception('Exceeded maximum number of iterations solving for coordinates.')
    if t < 0:        t += 2*np.pi
    if t >= 2*np.pi: t -= 2*np.pi
    return t
@numba.njit(parallel=False, fastmath=True)
def _multi_newton(its, x, y, newton_tol, _cx, _cy, verbose, maxi):
    N = x.size
    all_t = np.empty(N)
    for i in numba.prange(N):
        all_t[i] = _newton(its[i], x[i], y[i], newton_tol, _cx, _cy, verbose, maxi)
    return all_t
@numba.njit(fastmath=True)
def _guess_ind_finder(cx, cy, x, y):
    _min = 1e300
    argmin = -1
    for i in range(cx.size):
        dx = cx[i] - x
        dy = cy[i] - y
        d = np.sqrt(dx*dx + dy*dy)
        if d < _min:
            _min = d
            argmin = i
    return argmin
@numba.njit(fastmath=True, parallel=True)
def multi_guess_ind_finder(cx, cy, x, y, inds):
    for j in numba.prange(x.size):
        inds[j] = _guess_ind_finder(cx, cy, x[j], y[j])
def compute_local_coordinates_polyi(cx, cy, x, y, newton_tol=1e-12,
                                            guess_ind=None, verbose=False, max_iterations=30):
    """
    Find using the coordinates:
    x = X + r n_x
    y = Y + r n_y
    """
    xshape = x.shape
    x = x.flatten()
    y = y.flatten()
    xsize = x.size

    n = cx.shape[0]
    dt = 2*np.pi/n
    ts = np.arange(n)*dt
    ik = 1j*np.fft.fftfreq(n, dt/(2*np.pi))
    # tangent vectors
    xp = fourier_derivative_1d(f=cx, d=1, ik=ik, out='f')
    yp = fourier_derivative_1d(f=cy, d=1, ik=ik, out='f')
    # speed
    sp = np.sqrt(xp*xp + yp*yp)
    isp = 1.0/sp
    # unit tangent vectors
    tx = xp*isp
    ty = yp*isp
    # unit normal vectors
    nx = ty
    ny = -tx

    # brute force find of guess_inds if not provided (slow!)
    if guess_ind is None:
        guess_ind = np.empty(x.size, dtype=int)
        multi_guess_ind_finder(cx, cy, x, y, guess_ind)

    # get starting points (initial guess for t and r)
    initial_ts = ts[guess_ind]

    # run the multi-newton solver
    t = _multi_newton(initial_ts, x, y, newton_tol, cx, cy, verbose, max_iterations)

    # need to determine the sign now
    X, Y = np.zeros_like(x), np.zeros_like(x)
    XP, YP = np.zeros_like(x), np.zeros_like(x)
    multi_polyi_p(cx, t, X, XP)
    multi_polyi_p(cy, t, Y, YP)
    ISP = 1.0/np.sqrt(XP*XP + YP*YP)
    NX = YP/ISP
    NY = -XP/ISP
    r = np.hypot(X-x, Y-y)
    xe1 = X + r*NX
    ye1 = Y + r*NY
    err1 = np.hypot(xe1-x, ye1-y)
    xe2 = X - r*NX
    ye2 = Y - r*NY
    err2 = np.hypot(xe2-x, ye2-y)
    sign = (err1 < err2).astype(int)*2 - 1

    return t, r*sign


def compute_local_coordinates_polyi_centering(cx, cy, x, y, gi, newton_tol=1e-12, verbose=None, max_iterations=30):
    """
    Find using the coordinates:
    x = X + r n_x
    y = Y + r n_y
    """
    xshape = x.shape
    x = x.flatten()
    y = y.flatten()
    gi = gi.flatten()
    xsize = x.size

    n = cx.shape[0]
    dt = 2*np.pi/n
    ts = np.arange(n)*dt
    ik = 1j*np.fft.fftfreq(n, dt/(2*np.pi))
    # tangent vectors
    xp = fourier_derivative_1d(f=cx, d=1, ik=ik, out='f')
    yp = fourier_derivative_1d(f=cy, d=1, ik=ik, out='f')
    # speed
    sp = np.sqrt(xp*xp + yp*yp)
    isp = 1.0/sp
    # unit tangent vectors
    tx = xp*isp
    ty = yp*isp
    # unit normal vectors
    nx = ty
    ny = -tx

    # brute force find of guess_inds if not provided (slow!)
    guess_ind = np.empty(x.size, dtype=int)
    multi_guess_ind_finder_centering(cx, cy, x, y, guess_ind, gi, 10)

    # get starting points (initial guess for t and r)
    initial_ts = ts[guess_ind]

    # run the multi-newton solver
    t = _multi_newton(initial_ts, x, y, newton_tol, cx, cy, verbose, max_iterations)

    # need to determine the sign now
    X, Y = np.zeros_like(x), np.zeros_like(x)
    XP, YP = np.zeros_like(x), np.zeros_like(x)
    multi_polyi_p(cx, t, X, XP)
    multi_polyi_p(cy, t, Y, YP)
    ISP = 1.0/np.sqrt(XP*XP + YP*YP)
    NX = YP/ISP
    NY = -XP/ISP
    r = np.hypot(X-x, Y-y)
    xe1 = X + r*NX
    ye1 = Y + r*NY
    err1 = np.hypot(xe1-x, ye1-y)
    xe2 = X - r*NX
    ye2 = Y - r*NY
    err2 = np.hypot(xe2-x, ye2-y)
    sign = (err1 < err2).astype(int)*2 - 1

    return t, r*sign





































def compute_local_coordinates_old(cx, cy, x, y, newton_tol=1e-12,
            interpolation_scheme='nufft', guess_ind=None, verbose=False, max_iterations=30):
    """
    Find (s, r) given (x, y) using the coordinates:
    x = X(s) + r n_x(s)
    y = Y(s) + r n_y(s)

    Where X, Y is given by cx, cy
    Uses a NUFFT based scheme if interpolation_scheme = 'nufft'
    And a polynomail based scheme if interpolation_scheme = 'polyi'

    The NUFFT scheme is more accurate and robust;
    The polynomial based scheme is more accurate
    """
    if interpolation_scheme == 'nufft':
        return compute_local_coordinates_nufft_old(cx, cy, x, y, newton_tol, guess_ind, verbose, max_iterations)
    elif interpolation_scheme == 'polyi':
        return compute_local_coordinates_polyi_old(cx, cy, x, y, newton_tol, guess_ind, verbose, max_iterations)
    else:
        raise Exception('interpolation_scheme not recognized')    

def compute_local_coordinates_nufft_old(cx, cy, x, y, newton_tol=1e-12, 
                                            guess_ind=None, verbose=False, max_iterations=30):
    """
    Find using the coordinates:
    x = X + r n_x
    y = Y + r n_y
    """
    xshape = x.shape
    yshape = y.shape
    x = x.flatten()
    y = y.flatten()

    n = cx.shape[0]
    dt = 2*np.pi/n
    ts = np.arange(n)*dt
    ik = 1j*np.fft.fftfreq(n, dt/(2*np.pi))
    # tangent vectors
    xp = fourier_derivative_1d(f=cx, d=1, ik=ik, out='f')
    yp = fourier_derivative_1d(f=cy, d=1, ik=ik, out='f')
    # speed
    sp = np.sqrt(xp*xp + yp*yp)
    isp = 1.0/sp
    # unit tangent vectors
    tx = xp*isp
    ty = yp*isp
    # unit normal vectors
    nx = ty
    ny = -tx
    # derivatives of the normal vector
    nxp = fourier_derivative_1d(f=nx, d=1, ik=ik, out='f')
    nyp = fourier_derivative_1d(f=ny, d=1, ik=ik, out='f')

    # interpolation routines for the necessary objects
    if have_better_fourier:
        def interp(f):
            return _interp2(f)
    else:
        def interp(f):
            return _interp(np.fft.fft(f), x.size)
    nx_i =  interp(nx)
    ny_i =  interp(ny)
    nxp_i = interp(nxp)
    nyp_i = interp(nyp)
    x_i =   interp(cx)
    y_i =   interp(cy)
    xp_i =  interp(xp)
    yp_i =  interp(yp)

    # functions for coordinate transform and its jacobian
    def f(t, r):
        x = x_i(t) + r*nx_i(t)
        y = y_i(t) + r*ny_i(t)
        return x, y
    def Jac(t, r):
        dxdt = xp_i(t) + r*nxp_i(t)
        dydt = yp_i(t) + r*nyp_i(t)
        dxdr = nx_i(t)
        dydr = ny_i(t)
        J = np.zeros((t.shape[0],2,2),dtype=float)
        J[:,0,0] = dxdt
        J[:,0,1] = dydt
        J[:,1,0] = dxdr
        J[:,1,1] = dydr
        return J.transpose((0,2,1))

    # brute force find of guess_inds if not provided (slow!)
    if guess_ind is None:
        xd = x - cx[:,None]
        yd = y - cy[:,None]
        dd = xd**2 + yd**2
        guess_ind = dd.argmin(axis=0)

    # get starting points (initial guess for t and r)
    t = ts[guess_ind]
    cxg = cx[guess_ind]
    cyg = cy[guess_ind]
    xdg = x - cxg
    ydg = y - cyg
    r = np.sqrt(xdg**2 + ydg**2)
    # how about we re-sign r?
    nxg = nx[guess_ind]
    nyg = ny[guess_ind]
    xcr1 = cxg + r*nxg
    ycr1 = cyg + r*nyg
    d1 = np.hypot(xcr1-x, ycr1-y)
    xcr2 = cxg - r*nxg
    ycr2 = cyg - r*nyg
    d2 = np.hypot(xcr2-x, ycr2-y)
    better2 = d1 > d2
    r[better2] *= -1

    # begin Newton iteration
    xo, yo = f(t, r)
    remx = xo - x
    remy = yo - y
    rem = np.abs(np.sqrt(remx**2 + remy**2)).max()
    if verbose:
        print('Newton tol: {:0.2e}'.format(rem))
    iteration = 0
    while rem > newton_tol:
        J = Jac(t, r)
        delt = -np.linalg.solve(J, np.column_stack([remx, remy]))
        line_factor = 1.0
        while True:
            t_new, r_new = t + line_factor*delt[:,0], r + line_factor*delt[:,1]
            try:
                xo, yo = f(t_new, r_new)
                remx = xo - x
                remy = yo - y
                rem_new = np.sqrt(remx**2 + remy**2).max()
                testit = True
            except:
                testit = False
            if testit and ((rem_new < (1-0.5*line_factor)*rem) or line_factor < 1e-4):
                t = t_new
                # put theta back in [0, 2 pi]
                t[t < 0] += 2*np.pi
                t[t > 2*np.pi] -= 2*np.pi
                r = r_new
                rem = rem_new
                break
            line_factor *= 0.5
        if verbose:
            print('Newton tol: {:0.2e}'.format(rem))
        iteration += 1
        if iteration > max_iterations:
            raise Exception('Exceeded maximum number of iterations solving for coordinates .')

    return t, r


#### Faster version based on poly interpolation + numba
# LAGRANGE VERSION IS SLOW RIGHT NOW
# Lagrange interpolation matrix
"""
snodes = np.linspace(-1, 1, 6)
MAT = np.empty([6,6], dtype=float)
for i in range(6):
    MAT[:,i] = np.power(snodes,i)
IMAT = np.linalg.inv(MAT)
@numba.njit
def _polyi(f, xx):
    n = f.size
    h = 2*np.pi/n
    ix = int(xx//h)
    lb = (ix-2)*h
    ub = (ix+3)*h
    scalex = 2*((xx-lb)/(ub-lb) - 0.5)
    p2 = scalex*scalex
    p3 = p2*scalex
    p4 = p3*scalex
    p5 = p4*scalex
    ix0 = (ix-2)%n
    ix1 = (ix-1)%n
    ix2 = (ix  )%n
    ix3 = (ix+1)%n
    ix4 = (ix+2)%n
    ix5 = (ix+3)%n

    fout = 0.0
    for j in range(6):
        fout += IMAT[0, j]*f[(ix+j-2) % n]
    coef = 0.0
    for j in range(6):
        coef += IMAT[1, j]*f[(ix+j-2) % n]
    fout += coef*scalex
    coef = 0.0
    for j in range(6):
        coef += IMAT[2, j]*f[(ix+j-2) % n]
    fout += coef*p2
    coef = 0.0
    for j in range(6):
        coef += IMAT[3, j]*f[(ix+j-2) % n]
    fout += coef*p3
    coef = 0.0
    for j in range(6):
        coef += IMAT[4, j]*f[(ix+j-2) % n]
    fout += coef*p4
    coef = 0.0
    for j in range(6):
        coef += IMAT[5, j]*f[(ix+j-2) % n]
    fout += coef*p5

    return fout
#"""

# using this for now, much faster, not as stable...
@numba.njit
def _f_old(t, r, cx, cy, nx, ny):
    x = _polyi(cx, t) + r*_polyi(nx, t)
    y = _polyi(cy, t) + r*_polyi(ny, t)
    return x, y
@numba.njit
def _Jac_old(t, r, xp, yp, nx, ny, nxp, nyp):
    J00 = _polyi(xp, t) + r*_polyi(nxp, t)
    J01 = _polyi(nx, t)
    J10 = _polyi(yp, t) + r*_polyi(nyp, t)
    J11 = _polyi(ny, t)
    return J00, J01, J10, J11
@numba.njit
def _newton_old(t, r, xi, yi, newton_tol, cx, cy, nx, ny, xp, yp, nxp, nyp, verbose, maxi):
    # get initial residual
    xo, yo = _f_old(t, r, cx, cy, nx, ny)
    remx = xo - xi
    remy = yo - yi
    rem = np.sqrt(remx**2 + remy**2)

    # Newton iteration
    iteration = 0
    while rem > newton_tol:
        # get Jacobian
        J00, J01, J10, J11 = _Jac_old(t, r, xp, yp, nx, ny, nxp, nyp)
        idetJ = 1.0/(J00*J11-J01*J10)
        # compute step
        delt = (J01*remy-J11*remx)*idetJ
        delr = (J10*remx-J00*remy)*idetJ

        # take step with simple line search
        line_factor = 1.0
        while True:
            t_new, r_new = t + line_factor*delt, r + line_factor*delr
            xo, yo = _f_old(t_new, r_new, cx, cy, nx, ny)
            remx = xo - xi
            remy = yo - yi
            rem_new = np.sqrt(remx**2 + remy**2)
            if (rem_new < (1-0.25*line_factor)*rem) or line_factor < 1e-4:
                t = t_new
                r = r_new
                rem = rem_new
                break
            line_factor *= 0.5
        iteration += 1
        if iteration > maxi:
            raise Exception('Exceeded maximum number of iterations solving for coordinates.')
    if t < 0:        t += 2*np.pi
    if t >= 2*np.pi: t -= 2*np.pi
    return t, r
@numba.njit(parallel=True)
def _multi_newton_old(its, irs, x, y, newton_tol, cx, cy, nx, ny, xp, yp, nxp, nyp, verbose, maxi):
    N = x.size
    all_r = np.empty(N)
    all_t = np.empty(N)
    for i in numba.prange(N):
        all_t[i], all_r[i] = _newton_old(its[i], irs[i], x[i], y[i], newton_tol, cx, cy, nx, ny, xp, yp, nxp, nyp, verbose, maxi)
    return all_t, all_r
def compute_local_coordinates_polyi_old(cx, cy, x, y, newton_tol=1e-12,
                                            guess_ind=None, verbose=False, max_iterations=30):
    """
    Find using the coordinates:
    x = X + r n_x
    y = Y + r n_y
    """
    xshape = x.shape
    x = x.flatten()
    y = y.flatten()
    xsize = x.size

    n = cx.shape[0]
    dt = 2*np.pi/n
    ts = np.arange(n)*dt
    ik = 1j*np.fft.fftfreq(n, dt/(2*np.pi))
    # tangent vectors
    xp = fourier_derivative_1d(f=cx, d=1, ik=ik, out='f')
    yp = fourier_derivative_1d(f=cy, d=1, ik=ik, out='f')
    # speed
    sp = np.sqrt(xp*xp + yp*yp)
    isp = 1.0/sp
    # unit tangent vectors
    tx = xp*isp
    ty = yp*isp
    # unit normal vectors
    nx = ty
    ny = -tx
    # derivatives of the normal vector
    nxp = fourier_derivative_1d(f=nx, d=1, ik=ik, out='f')
    nyp = fourier_derivative_1d(f=ny, d=1, ik=ik, out='f')

    # brute force find of guess_inds if not provided (slow!)
    if guess_ind is None:
        xd = x - cx[:,None]
        yd = y - cy[:,None]
        dd = xd**2 + yd**2
        guess_ind = dd.argmin(axis=0)

    # get starting points (initial guess for t and r)
    initial_ts = ts[guess_ind]
    cxg = cx[guess_ind]
    cyg = cy[guess_ind]
    xdg = x - cxg
    ydg = y - cyg
    initial_rs = np.sqrt(xdg**2 + ydg**2)
    # how about we re-sign r?
    nxg = nx[guess_ind]
    nyg = ny[guess_ind]
    xcr1 = cxg + initial_rs*nxg
    ycr1 = cyg + initial_rs*nyg
    d1 = np.hypot(xcr1-x, ycr1-y)
    xcr2 = cxg - initial_rs*nxg
    ycr2 = cyg - initial_rs*nyg
    d2 = np.hypot(xcr2-x, ycr2-y)
    better2 = d1 > d2
    initial_rs[better2] *= -1

    # run the multi-newton solver
    all_t, all_r = _multi_newton_old(initial_ts, initial_rs, x, y, newton_tol, cx, cy, nx, ny, xp, yp, nxp, nyp, verbose, max_iterations)
    return all_t, all_r
































