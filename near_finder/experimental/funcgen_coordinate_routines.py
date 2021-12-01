import numpy as np
import numba
import scipy
import scipy.interpolate
from near_finder.utilities import fourier_derivative_1d
from function_generator import FunctionGenerator
from near_finder.utilities import interp_fourier
from near_finder.nufft_interp import periodic_interp1d

@numba.njit(fastmath=True)
def dirft1(x, fh, ik):
    c = np.empty(x.size, dtype=numba.complex128)
    for j in range(x.size):
        ch = 0.0 + 0.0j
        for i in range(fh.size):
            ch += np.exp(ik[i]*x[j])*fh[i]
        c[j] = ch
    return c
@numba.njit(fastmath=True)
def dirft2(x, fh, ik):
    c = np.empty(x.size, dtype=numba.complex128)
    for j in range(x.size):
        ch = 0.0 + 0.0j
        xx = np.exp(x[j]*1j)
        ixx = 1.0/xx
        xh1 = 1.0
        xh2 = ixx
        for i in range(fh.size//2):
            ch += xh1*fh[i] + xh2*fh[fh.size-1-i]
            xh1 *= xx
            xh2 *= ixx 
        c[j] = ch
    return c
@numba.njit(fastmath=True)
def dirft2_scalar(x, fh, ik):
    ch = 0.0 + 0.0j
    xx = np.exp(x*1j)
    ixx = 1.0/xx
    xh1 = 1.0
    xh2 = ixx
    for i in range(fh.size//2):
        ch += xh1*fh[i] + xh2*fh[fh.size-1-i]
        xh1 *= xx
        xh2 *= ixx 
    return ch
class direct_interp_fourier(object):
    def __init__(self, f):
        self.fh = np.fft.fft(f)/f.size
        self.ik = 1j*np.fft.fftfreq(self.fh.size, 1.0/self.fh.size)
    def __call__(self, x):
        if type(x) == np.ndarray:
            return dirft2(x, self.fh, self.ik)
        else:
            return dirft2_scalar(x, self.fh, self.ik)

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

@numba.njit(fastmath=True)
def _f_fg(t, _cx, _cy, _cxp, _cyp, x, y):
    cx = _cx(t)
    cy = _cy(t)
    cxp = _cxp(t)
    cyp = _cyp(t)
    return cxp*(cx-x) + cyp*(cy-y)
@numba.njit(fastmath=True)
def _f_fp_fg(t, _cx, _cy, _cxp, _cyp, _cxpp, _cypp, x, y):
    cx = _cx(t)
    cy = _cy(t)
    cxp = _cxp(t)
    cyp = _cyp(t)
    cxpp = _cxpp(t)
    cypp = _cypp(t)
    f = cxp*(cx-x) + cyp*(cy-y)
    fp = cxpp*(cx-x) + cypp*(cy-y) + cxp*cxp + cyp*cyp
    return f, fp
@numba.njit(fastmath=True)
def _newton_fg(t, xi, yi, newton_tol, _cx, _cy, _cxp, _cyp, _cxpp, _cypp, verbose, maxi):
    # get initial residual
    rem, fp = _f_fp(t, _cx, _cy, _cxp, _cyp, _cxpp, _cypp, xi, yi)

    # Newton iteration
    iteration = 0
    while np.abs(rem) > newton_tol:
        # compute step
        t = t -rem/fp
        rem, fp = _f_fp(t, _cx, _cy, _cxp, _cyp, _cxpp, _cypp, xi, yi)
        iteration += 1
        if iteration > maxi:
            raise Exception('Exceeded maximum number of iterations solving for coordinates.')
    if t < 0:        t += 2*np.pi
    if t >= 2*np.pi: t -= 2*np.pi
    return t
@numba.njit(parallel=True, fastmath=True)
def _multi_newton_fg(its, x, y, newton_tol, _cx, _cy, _cxp, _cyp, _cxpp, _cypp, verbose, maxi):
    N = x.size
    all_t = np.empty(N)
    for i in numba.prange(N):
        all_t[i] = _newton(its[i], x[i], y[i], newton_tol, _cx, _cy, _cxp, _cyp, _cxpp, _cypp, verbose, maxi)
    return all_t

class coordinate_finder:
    def __init__(self, c, funcgen_tol, funcgen_order):
        # c:  boundary coordinates, complex
        self.c = c
        self.cx = self.c.real.copy()
        self.cy = self.c.imag.copy()
        # get some useful boundary-related information
        self.n = self.c.size
        self.dt = 2*np.pi/self.n
        self.ts = np.arange(self.n)*self.dt
        self.ik = np.fft.fftfreq(self.n, 1.0/self.n)
        # get derivatives of this
        self.cp = fourier_derivative_1d(f=self.c, d=1, ik=self.ik)
        self.cpp = fourier_derivative_1d(f=self.c, d=2, ik=self.ik)
        # get direct fourier evaluator for this
        self.dirf_c = periodic_interp1d(self.c)
        self.dirf_cp = periodic_interp1d(self.cp)
        self.dirf_cpp = periodic_interp1d(self.cpp)
        # build function generator representations of these functions
        self.funcgen_c = FunctionGenerator(self.dirf_c, 0, 2*np.pi, funcgen_tol, n=funcgen_order)
        self.funcgen_cp = FunctionGenerator(self.dirf_cp, 0, 2*np.pi, funcgen_tol, n=funcgen_order)
        self.funcgen_cpp = FunctionGenerator(self.dirf_cpp, 0, 2*np.pi, funcgen_tol, n=funcgen_order)
    def __call__(self, x, y, gi, newton_tol, max_iterations=50):
        xshape = x.shape
        x = x.flatten()
        y = y.flatten()
        gi = gi.flatten()
        xsize = x.size
        # brute force find of guess_inds given centering location
        guess_ind = np.empty(x.size, dtype=int)
        multi_guess_ind_finder_centering(self.cx, self.cy, x, y, guess_ind, gi, 10)
        # get starting guess for t
        guess_t = self.ts[guess_ind]
        # run the multi-newton solver
        t = _multi_newton(initial_ts, x, y, newton_tol, cx, cy, verbose, max_iterations)
        # compute r [PICK UP HERE!]
        # X, Y = np.zeros_like(x), np.zeros_like(x)
        # XP, YP = np.zeros_like(x), np.zeros_like(x)
        # multi_polyi_p(cx, t, X, XP)
        # multi_polyi_p(cy, t, Y, YP)
        # ISP = 1.0/np.sqrt(XP*XP + YP*YP)
        # NX = YP/ISP
        # NY = -XP/ISP
        # r = np.hypot(X-x, Y-y)
        # xe1 = X + r*NX
        # ye1 = Y + r*NY
        # err1 = np.hypot(xe1-x, ye1-y)
        # xe2 = X - r*NX
        # ye2 = Y - r*NY
        # err2 = np.hypot(xe2-x, ye2-y)
        # sign = (err1 < err2).astype(int)*2 - 1
        return t

def compute_local_coordinates_funcgen_centering(cx, cy, x, y, gi, newton_tol=1e-12, verbose=None, max_iterations=30):
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


