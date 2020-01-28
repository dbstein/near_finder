import numpy as np
import scipy as sp
import scipy.signal
import warnings
import finufftpy
import fast_interp
import numba

@numba.njit
def inarray(val, arr):
    init = False
    i = 0
    arr = arr.ravel()
    while not init and i < arr.size:
        init = arr[i] == val
        i += 1
    if not init:
        i = 0
    return init, i-1

@numba.njit
def extend_array(a, new_size, fill_zero=False):
    new = np.empty(new_size, a.dtype)
    new[:a.size] = a
    if fill_zero:
        new[a.size:] = 0
    return new

def fourier_derivative_1d(f=None, fh=None, d=1, ik=None, h=None, out='f'):
    """
    1 dimensional fourier derivative

    Inputs:
        f,   float/complex(n), physical function to be differentiated
        fh,  complex(n), fourier transform of function to be differentiated
             (only one of f or fh need be provided; if both are fh is used)
        d,   int, derivative to be computed
        ik,  complex(n), wavenumbers, if not provided computed as:
                ik = np.fft.fftfreq(n, h/(2*np.pi))
        h,   float, sample spacing
             (one of ik or h must be provided.  if both are, ik is used)
        out, str, descibes the output type.  options are:
             'f': returns the physical function only, assuming it is real
             'c': returns the physical function only, assuming it is complex
             'h': returns the fourier transform only
             'fh': returns a tuple of ('f', 'h') output
             'ch': returns a tuple of ('c', 'h') output
    """
    if fh is None: fh = np.fft.fft(f)
    if ik is None: ik = np.fft.fftfreq(n, h/(2*np.pi))
    if d == 0:
        mult = ik*0.0 + 1.0
    elif d == 1:
        mult = ik
    elif d == 2:
        mult = ik*ik
    else:
        mult = ik**d
    dh = fh*mult
    if out == 'f':
        return np.fft.ifft(dh).real
    elif out == 'c':
        return np.fft.ifft(dh)
    elif out == 'h':
        return dh
    elif out == 'fh':
        return np.fft.ifft(dh).real, dh
    elif out == 'ch':
        return np.fft.ifft(dh), dh
    else:
        raise Exception("Variable 'out' not recognized.")

def compute_curvature(x, y):
    """
    Computes the curvature of a closed plane curve (x(s), y(s))
    given points x(s_i), y(s_i)
    the final point is assumed to be not repeated
    """
    n = x.shape[0]
    ik = 1j*np.fft.fftfreq(n, 1.0/n)
    xh = np.fft.fft(x)
    yh = np.fft.fft(y)
    xp = fourier_derivative_1d(fh=xh, d=1, ik=ik, out='f')
    yp = fourier_derivative_1d(fh=yh, d=1, ik=ik, out='f')
    xpp = fourier_derivative_1d(fh=xh, d=2, ik=ik, out='f')
    ypp = fourier_derivative_1d(fh=yh, d=2, ik=ik, out='f')
    sp = np.sqrt(xp*xp + yp*yp)
    return (xp*ypp-yp*xpp)/sp**3

def compute_speed(x, y):
    """
    Computes the speed of a closed plane curve (x(s), y(s))
    given points x(s_i), y(s_i)
    the final point is assumed to be not repeated
    """
    n = x.shape[0]
    ik = 1j*np.fft.fftfreq(n, 1.0/n)
    xh = np.fft.fft(x)
    yh = np.fft.fft(y)
    xp = fourier_derivative_1d(fh=xh, d=1, ik=ik, out='f')
    yp = fourier_derivative_1d(fh=yh, d=1, ik=ik, out='f')
    return np.sqrt(xp*xp + yp*yp)

def compute_normals(x, y):
    n = x.shape[0]
    ik = 1j*np.fft.fftfreq(n, 1.0/n)
    xh = np.fft.fft(x)
    yh = np.fft.fft(y)
    xp = fourier_derivative_1d(fh=xh, d=1, ik=ik, out='f')
    yp = fourier_derivative_1d(fh=yh, d=1, ik=ik, out='f')
    sp = np.sqrt(xp*xp + yp*yp)
    sp = np.sqrt(xp*xp + yp*yp)
    isp = 1.0/sp
    tx = xp*isp
    ty = yp*isp
    nx = ty
    ny = -tx
    return nx, ny

def star(N, x=0.0, y=0.0, r=1.0, a=0.5, f=3, rot=0.0):
    """
    Function defining a star shaped object
    Parameters:
        N:   number of points
        x:   x coordinate of center
        y:   y coordinate of center
        r:   nominal radius
        a:   amplitude of wobble, 0<a<1, smaller a is less wobbly
        f:   frequency - how many lobes are in the star
        rot: angle of rotation
    """
    t = np.linspace(0.0, 2.0*np.pi, N, endpoint=False)
    c = (x+1j*y) + (r + r*a*np.cos(f*(t-rot)))*np.exp(1j*t)
    return c.real, c.imag

def upsample(f, N):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = sp.signal.resample(f, N)
    return out

class interp_poly(object):
    def __init__(self, f, order):
        self.func = fast_interp.interp1d(0, 2*np.pi, 2*np.pi/f.size, f, k=order, p=True)
    def __call__(self, x_out):
        return self.func(x_out)

class interp_fourier(object):
    def __init__(self, in_hat, out_size):
        self.in_hat = in_hat
        self.out = np.empty(out_size, dtype=complex)
        self.adj = 1.0/self.in_hat.shape[0]
    def __call__(self, x_out):
        finufftpy.nufft1d2(x_out, self.out, 1, 1e-15, self.in_hat, modeord=1)
        return self.out.real*self.adj
