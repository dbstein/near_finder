import numpy as np
import numba
import scipy
import scipy.interpolate
from .utilities import fourier_derivative_1d
from .utilities import interp_fourier as _interp
from .utilities import interp_poly as _poly_interp

def compute_local_coordinates(cx, cy, x, y, newton_tol=1e-14, 
                                            guess_ind=None, verbose=False):
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
    def interp(f):
        return _interp(np.fft.fft(f), x.size)
        # return _interp(f)
    def interp(f):
        return _poly_interp(f, 5)
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
    xdg = x - cx[guess_ind]
    ydg = y - cy[guess_ind]
    r = np.sqrt(xdg**2 + ydg**2)

    # begin Newton iteration
    xo, yo = f(t, r)
    remx = xo - x
    remy = yo - y
    rem = np.abs(np.sqrt(remx**2 + remy**2)).max()
    if verbose:
        print('Newton tol: {:0.2e}'.format(rem))
    while rem > newton_tol:
        J = Jac(t, r)
        delt = -np.linalg.solve(J, np.column_stack([remx, remy]))
        line_factor = 1.0
        while True:
            t_new, r_new = t + line_factor*delt[:,0], r + line_factor*delt[:,1]
            xo, yo = f(t_new, r_new)
            remx = xo - x
            remy = yo - y
            rem_new = np.sqrt(remx**2 + remy**2).max()
            if (rem_new < (1-0.5*line_factor)*rem) or line_factor < 1e-4:
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

    return t, r


#### Faster version based on poly interpolation + numba
@numba.njit
def _polyi(f, xx):
    n = f.size
    h = 2*np.pi/n
    ix = int(xx//h)
    ratx = xx/h - (ix+0.5)

    fout  = f[(ix + 0) % n] * (3/256 + ratx*(   -9/1920 + ratx*( -5/48/2 + ratx*(  1/8/6 + ratx*( 1/2/24 -  1/8/120*ratx)))))
    fout += f[(ix + 1) % n] * (-25/256 + ratx*(  125/1920 + ratx*( 39/48/2 + ratx*(-13/8/6 + ratx*(-3/2/24 +  5/8/120*ratx)))))
    fout += f[(ix + 2) % n] * (150/256 + ratx*(-2250/1920 + ratx*(-34/48/2 + ratx*( 34/8/6 + ratx*( 2/2/24 - 10/8/120*ratx)))))
    fout += f[(ix + 3) % n] * (150/256 + ratx*( 2250/1920 + ratx*(-34/48/2 + ratx*(-34/8/6 + ratx*( 2/2/24 + 10/8/120*ratx)))))
    fout += f[(ix + 4) % n] * (-25/256 + ratx*( -125/1920 + ratx*( 39/48/2 + ratx*( 13/8/6 + ratx*(-3/2/24 -  5/8/120*ratx)))))
    fout += f[(ix + 5) % n] * (3/256 + ratx*(    9/1920 + ratx*( -5/48/2 + ratx*( -1/8/6 + ratx*( 1/2/24 +  1/8/120*ratx)))))
    return fout
@numba.njit
def _f(t, r, cx, cy, nx, ny):

    x = _polyi(cx, t) + r*_polyi(nx, t)
    y = _polyi(cy, t) + r*_polyi(ny, t)
    return x, y
@numba.njit
def _Jac(t, r, xp, yp, nx, ny, nxp, nyp):
    J00 = _polyi(xp, t) + r*_polyi(nxp, t)
    J01 = _polyi(nx, t)
    J10 = _polyi(yp, t) + r*_polyi(nyp, t)
    J11 = _polyi(ny, t)
    return J00, J01, J10, J11
@numba.njit
def _newton(t, r, xi, yi, newton_tol, _f, cx, cy, nx, ny, _Jac, xp, yp, nxp, nyp):
    # get initial residual
    xo, yo = _f(t, r, cx, cy, nx, ny)
    remx = xo - xi
    remy = yo - yi
    rem = np.sqrt(remx**2 + remy**2)

    # Newton iteration
    while rem > newton_tol:
        # get Jacobian
        J00, J01, J10, J11 = _Jac(t, r, xp, yp, nx, ny, nxp, nyp)
        idetJ = 1.0/(J00*J11-J01*J10)
        # compute step
        delt = (J01*remy-J11*remx)*idetJ
        delr = (J10*remx-J00*remy)*idetJ

        # take step with simple line search
        line_factor = 1.0
        while True:
            t_new, r_new = t + line_factor*delt, r + line_factor*delr
            xo, yo = _f(t_new, r_new, cx, cy, nx, ny)
            remx = xo - xi
            remy = yo - yi
            rem_new = np.sqrt(remx**2 + remy**2)
            if (rem_new < (1-0.5*line_factor)*rem) or line_factor < 1e-4:
                t = t_new
                r = r_new
                rem = rem_new
                break
            line_factor *= 0.5
    # put theta back in [0, 2 pi]
    if t < 0:       t += 2*np.pi
    if t > 2*np.pi: t -= 2*np.pi
    return t, r
@numba.njit(parallel=True)
def _multi_newton(its, irs, x, y, newton_tol, _f, cx, cy, nx, ny, _Jac, xp, yp, nxp, nyp):
    N = x.size
    all_r = np.empty(N)
    all_t = np.empty(N)
    for i in numba.prange(N):
        all_t[i], all_r[i] = _newton(its[i], irs[i], x[i], y[i], newton_tol, _f, cx, cy, nx, ny, _Jac, xp, yp, nxp, nyp)
    return all_t, all_r

def compute_local_coordinates(cx, cy, x, y, newton_tol=1e-14,
                                            guess_ind=None, verbose=False):
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
    xdg = x - cx[guess_ind]
    ydg = y - cy[guess_ind]
    initial_rs = np.sqrt(xdg**2 + ydg**2)

    # run the multi-newton solver
    all_t, all_r = _multi_newton(initial_ts, initial_rs, x, y, newton_tol, _f, cx, cy, nx, ny, _Jac, xp, yp, nxp, nyp)
    return all_t, all_r

