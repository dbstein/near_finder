import numpy as np
import numba
from function_generator import FunctionGenerator
try:
    get_thread_id = numba.np.ufunc.parallel._get_thread_id
except:
    get_thread_id = numba.get_thread_id

"""
Highly optimized periodic interpolation for stacks of functions
calls either serial or parallel direct or nufft or finufft functions
based on heuristics regarding the size of the transforms
"""

################################################################################
# direct evaluation function

def dirft2d_1(x, y, fh):
    szx = fh.shape[0]
    szy = fh.shape[1]
    szx2 = szx // 2
    szy2 = szy // 2
    out = np.zeros(x.size, dtype=fh.dtype)
    for j in numba.prange(x.size):
        xx = np.exp(x[j]*1j)
        yy = np.exp(y[j]*1j)
        ixx = 1.0/xx
        iyy = 1.0/yy
        xh1 = 1.0
        xh2 = ixx
        for i in range(szx2):
            yh1 = 1.0
            yh2 = iyy
            for k in range(szy2):
                out[j] += xh1*yh1*fh[i, k] + xh2*yh1*fh[szx-1-i, k] + xh1*yh2*fh[i, szy-1-k] + xh2*yh2*fh[szx-1-i, szy-1-k]
                yh1 *= yy
                yh2 *= iyy
            xh1 *= xx
            xh2 *= ixx
    return out
def dirft2d_batch(x, y, fh):
    # note: other storage order is better (with stacking of fh over last index over last index)
    # however, this is most convenient and this function is not optimized for speed, but for checking...
    bn = fh.shape[0]
    szx = fh.shape[1]
    szy = fh.shape[2]
    szx2 = szx // 2
    szy2 = szy // 2
    out = np.zeros((bn, x.size), dtype=fh.dtype)
    for j in numba.prange(x.size):
        xx = np.exp(x[j]*1j)
        yy = np.exp(y[j]*1j)
        ixx = 1.0/xx
        iyy = 1.0/yy
        xh1 = 1.0
        xh2 = ixx
        for i in range(szx2):
            yh1 = 1.0
            yh2 = iyy
            for k in range(szy2):
                for l in range(bn):
                    out[l, j] += xh1*yh1*fh[l, i, k] + xh2*yh1*fh[l, szx-1-i, k] + xh1*yh2*fh[l, i, szy-1-k] + xh2*yh2*fh[l, szx-1-i, szy-1-k]
                yh1 *= yy
                yh2 *= iyy
            xh1 *= xx
            xh2 *= ixx
    return out

dirft2d_1_serial = numba.njit(dirft2d_1, fastmath=True, parallel=False)
dirft2d_1_parallel = numba.njit(dirft2d_1, fastmath=True, parallel=True)
dirft2d_batch_serial = numba.njit(dirft2d_batch, fastmath=True, parallel=False)
dirft2d_batch_parallel = numba.njit(dirft2d_batch, fastmath=True, parallel=True)

class interp2d_numba_direct:
    def __init__(self, fh):
        """
        fh:  (n_trans, nx_modes, ny_modes), stack of fourier modes for functions to interp
        """
        if len(fh.shape) == 2:
            self.serial_function = dirft2d_1_serial
            self.parallel_function = dirft2d_1_parallel
            self.n = fh.shape[0]*fh.shape[1]
        else:
            self.serial_function = dirft2d_batch_serial
            self.parallel_function = dirft2d_batch_parallel
            self.n = fh.shape[1]*fh.shape[2]
        self.scale = 1.0/self.n
        self.fh = fh * self.scale
    def __call__(self, x, y):
        m = x.size
        if m < 10 or m*self.n < 50000:
            return self.serial_function(x, y, self.fh)
        else:
            return self.parallel_function(x, y, self.fh)

################################################################################
# numba implementation of finufft

@numba.njit(fastmath=True)
def nphi(x, beta):
    x2 = x*x
    ok = int(x2<1)
    x2 = x2 * ok
    return np.exp(beta*(np.sqrt(1.0-x2)-1.0)) * ok
@numba.vectorize
def mphi(x, beta):
    return nphi(x, beta)

def convolve2d_1(xs, ys, big_y, ialphax, ialphay, w2, beta):
    nx = big_y.shape[0]
    ny = big_y.shape[1]
    hx = 2*np.pi/nx
    hy = 2*np.pi/ny
    out = np.zeros(xs.size, dtype=big_y.dtype)
    nzy = np.zeros((numba.get_num_threads(), 2*w2+1), dtype=np.float64)
    for i in numba.prange(xs.size):
        tid = get_thread_id()
        x = xs[i]
        indx = int(x // hx)
        min_indx = indx - w2
        max_indx = indx + w2 + 1
        y = ys[i]
        indy = int(y // hy)
        min_indy = indy - w2
        max_indy = indy + w2 + 1
        for kind, k in enumerate(range(min_indy, max_indy)):
            nzy[tid, kind] = nphi(ialphay*(y - k*hy), beta)
        for jind, j in enumerate(range(min_indx, max_indx)):
            nzx = nphi(ialphax*(x - j*hx), beta)
            for kind, k in enumerate(range(min_indy, max_indy)):
                out[i] += nzx*nzy[tid, kind]*big_y[j%nx, k%ny]
    return out
def convolve2d_batch(xs, ys, big_y, ialphax, ialphay, w2, beta):
    bn = big_y.shape[0]
    nx = big_y.shape[1]
    ny = big_y.shape[2]
    hx = 2*np.pi/nx
    hy = 2*np.pi/ny
    out = np.zeros((bn, xs.size), dtype=big_y.dtype)
    nzy = np.zeros((numba.get_num_threads(), 2*w2+1), dtype=np.float64)
    for i in numba.prange(xs.size):
        tid = get_thread_id()
        x = xs[i]
        indx = int(x // hx)
        min_indx = indx - w2
        max_indx = indx + w2 + 1
        y = ys[i]
        indy = int(y // hy)
        min_indy = indy - w2
        max_indy = indy + w2 + 1
        for kind, k in enumerate(range(min_indy, max_indy)):
            nzy[tid, kind] = nphi(ialphay*(y - k*hy), beta)
        for jind, j in enumerate(range(min_indx, max_indx)):
            nzx = nphi(ialphax*(x - j*hx), beta)
            for kind, k in enumerate(range(min_indy, max_indy)):
                w = nzx*nzy[tid, kind]
                for l in range(bn):
                    out[l, i] += w*big_y[l, j%nx, k%ny]
    return out

convolve2d_1_serial = numba.njit(convolve2d_1, fastmath=True, parallel=False)
convolve2d_1_parallel = numba.njit(convolve2d_1, fastmath=True, parallel=True)
convolve2d_batch_serial = numba.njit(convolve2d_batch, fastmath=True, parallel=False)
convolve2d_batch_parallel = numba.njit(convolve2d_batch, fastmath=True, parallel=True)

class interp2d_numba_nufft:
    def __init__(self, fh, eps):
        """
        fh:  (n_trans, nx_modes, ny_modes), stack of fourier modes for functions to interp
        """
        self.eps = eps
        self.w = 1 + int(np.ceil(np.log10(1/self.eps)))
        self.beta = 2.30 * self.w
        self.w2 = int(np.ceil(self.w/2))
        if len(fh.shape) == 2:
            self.serial_function = convolve2d_1_serial
            self.parallel_function = convolve2d_1_parallel
            self.bn = None
            self.nx = fh.shape[0]
            self.ny = fh.shape[1]
        else:
            self.serial_function = convolve2d_batch_serial
            self.parallel_function = convolve2d_batch_parallel
            self.bn = fh.shape[0]
            self.nx = fh.shape[1]
            self.ny = fh.shape[2]
        assert self.nx % 2 == 0, 'fh must have even # of modes in x-direction'
        assert self.ny % 2 == 0, 'fh must have even # of modes in y-direction'
        self.nx2 = int(self.nx//2)
        self.ny2 = int(self.ny//2)
        self.fh = fh * self.nx * self.ny / (2*np.pi)**2
        self.x = np.linspace(0, 2*np.pi, 2*self.nx, endpoint=False)
        self.alphax = np.pi*self.w/(2*self.nx)
        self.ialphax = 1.0/self.alphax
        self.y = np.linspace(0, 2*np.pi, 2*self.ny, endpoint=False)
        self.alphay = np.pi*self.w/(2*self.ny)
        self.ialphay = 1.0/self.alphay
        xpsi = np.fft.fftshift(mphi(self.ialphax*(self.x-np.pi), self.beta))
        xpsi_hat = np.fft.fft(xpsi)
        xpsi_hat = np.concatenate([xpsi_hat[:self.nx2], xpsi_hat[-self.nx2:]])
        self.pkx = 4*np.pi/self.nx/xpsi_hat
        ypsi = np.fft.fftshift(mphi(self.ialphay*(self.y-np.pi), self.beta))
        ypsi_hat = np.fft.fft(ypsi)
        ypsi_hat = np.concatenate([ypsi_hat[:self.ny2], ypsi_hat[-self.ny2:]])
        self.pky = 4*np.pi/self.ny/ypsi_hat
        self.pk = self.pkx[:,None]*self.pky
        self.fh_adj = self.pk*self.fh
        if self.bn is None:
            self.pad_fh_adj = np.zeros((2*self.nx, 2*self.ny), dtype=complex)
            self.pad_fh_adj[:self.nx//2,  :self.ny//2]  =  self.fh_adj[:self.nx//2,  :self.ny//2]
            self.pad_fh_adj[-self.nx//2:, :self.ny//2]  =  self.fh_adj[-self.nx//2:, :self.ny//2]
            self.pad_fh_adj[:self.nx//2,  -self.ny//2:] =  self.fh_adj[:self.nx//2,  -self.ny//2:]
            self.pad_fh_adj[-self.nx//2:, -self.ny//2:] =  self.fh_adj[-self.nx//2:, -self.ny//2:]
        else:
            self.pad_fh_adj = np.zeros((self.bn, 2*self.nx, 2*self.ny), dtype=complex)
            self.pad_fh_adj[:, :self.nx//2,  :self.ny//2]  =  self.fh_adj[:, :self.nx//2,  :self.ny//2]
            self.pad_fh_adj[:, -self.nx//2:, :self.ny//2]  =  self.fh_adj[:, -self.nx//2:, :self.ny//2]
            self.pad_fh_adj[:, :self.nx//2,  -self.ny//2:] =  self.fh_adj[:, :self.nx//2,  -self.ny//2:]
            self.pad_fh_adj[:, -self.nx//2:, -self.ny//2:] =  self.fh_adj[:, -self.nx//2:, -self.ny//2:]
        self.pad_f = np.fft.ifft2(self.pad_fh_adj)
    def __call__(self, x, y):
        if x.size < 1000:
            return self.serial_function(x, y, self.pad_f, self.ialphax, self.ialphay, self.w2, self.beta)
        else:
            return self.parallel_function(x, y, self.pad_f, self.ialphax, self.ialphay, self.w2, self.beta)

################################################################################
# method using my hacked version of finufft

class interp2d_finufft:
    def __init__(self, fh, eps, **kwargs):
        """
        fh:  (n_trans, nx_modes, ny_modes), stack of fourier modes for functions to interp
        """
        if len(fh.shape) == 2:
            self.bn = 1
            self.nx = fh.shape[0]
            self.ny = fh.shape[1]
        else:
            self.bn = fh.shape[0]
            self.nx = fh.shape[1]
            self.ny = fh.shape[2]
        self.plan = Plan(2, (self.nx, self.ny), n_trans=self.bn, eps=eps, isign=1, modeord=1, chkbnds=0, **kwargs)
        self.scale = 1.0/(self.nx*self.ny)
        self.fh = fh * self.scale
        self.plan.execute_type2_part1(self.fh)
    def __call__(self, x, y):
        self.plan.setpts(x, y, check=False)
        return self.plan.execute_type2_part2()

################################################################################
# one method to rule them all (at least unless i can reduce the overhead with finufft)

class periodic_interp2d:
    def __init__(self, f=None, fh=None, eps=1e-14, xbounds=None, ybounds=None, **kwargs):
        """
        f / fh: (n_func, nx, ny) or (nx, ny) stack of functions to interpolate
        """
        if fh is None:
            if f is None: raise Exception("If fh isn't given, need to give f")
            self.fh = np.fft.fft2(f)
        else:
            self.fh = fh
        if len(self.fh.shape) == 3:
            self.n_func = self.fh.shape[0]
            self.singleton = False
        else:
            self.n_func = 1
            self.singleton = True
        self.eps = eps
        self.evaluator = interp2d_numba_nufft(self.fh, self.eps, **kwargs)
        self.xbounds = xbounds
        self.ybounds = ybounds

    def __call__(self, x, y):
        if type(x) != np.ndarray:
            x = np.array([x])
            y = np.array([y])
            scalar = True
        else:
            sh = list(x.shape)
            if not self.singleton:
                sh = [self.n_func,] + sh
            x = x.ravel()
            y = y.ravel()
            scalar = False
        if self.xbounds is not None:
            x = 2*np.pi*(x - self.xbounds[0])/(self.xbounds[1]-self.xbounds[0])
        if self.ybounds is not None:
            y = 2*np.pi*(y - self.ybounds[0])/(self.ybounds[1]-self.ybounds[0])
        out = self.evaluator(x, y)
        if scalar:
            return out[0] if self.singleton else out[:,0]
        else:
            return out.reshape(sh)
