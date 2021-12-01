import numpy as np
import numba
from finufft import Plan
from function_generator import FunctionGenerator

"""
Highly optimized periodic interpolation for stacks of functions
calls either serial or parallel direct or nufft or finufft functions
based on heuristics regarding the size of the transforms
"""

################################################################################
# direct evaluation function

def dirft_1(x, fh):
    sz = fh.size
    sz2 = sz // 2
    out = np.zeros(x.size, dtype=fh.dtype)
    for j in numba.prange(x.size):
        xx = np.exp(x[j]*1j)
        ixx = 1.0/xx
        xh1 = 1.0
        xh2 = ixx
        for i in range(sz2):
            out[j] += xh1*fh[i] + xh2*fh[sz-1-i]
            xh1 *= xx
            xh2 *= ixx 
    return out
def dirft_batch(x, fh):
    bn = fh.shape[0]
    sz = fh.shape[1]
    sz2 = sz // 2
    out = np.zeros((bn, x.size), dtype=fh.dtype)
    for j in numba.prange(x.size):
        xx = np.exp(x[j]*1j)
        ixx = 1.0/xx
        xh1 = 1.0
        xh2 = ixx
        for i in range(sz2):
            for k in range(bn):
                out[k, j] += xh1*fh[k,i] + xh2*fh[k,sz-1-i]
            xh1 *= xx
            xh2 *= ixx 
    return out

dirft_1_serial = numba.njit(dirft_1, fastmath=True, parallel=False)
dirft_1_parallel = numba.njit(dirft_1, fastmath=True, parallel=True)
dirft_batch_serial = numba.njit(dirft_batch, fastmath=True, parallel=False)
dirft_batch_parallel = numba.njit(dirft_batch, fastmath=True, parallel=True)

class interp_numba_direct:
    def __init__(self, fh):
        """
        fh:  (n_trans, n_modes), stack of fourier modes for functions to interp
             or just (n_modes)
        """
        if len(fh.shape) == 1:
            self.serial_function = dirft_1_serial
            self.parallel_function = dirft_1_parallel
            self.n = fh.size
        else:
            self.serial_function = dirft_batch_serial
            self.parallel_function = dirft_batch_parallel
            self.n = fh.shape[1]
        self.fh = fh / self.n
    def __call__(self, x):
        m = x.size
        if m < 10 or m*self.n < 50000:
            return self.serial_function(x, self.fh)
        else:
            return self.parallel_function(x, self.fh)

################################################################################
# numba implementation of finufft (with not great computation of pk)
# and function generator evaluation of phi

@numba.njit(fastmath=True, inline='always')
def nphi(x, beta):
    x2 = x*x
    ok = int(x2<1)
    x2 = x2 * ok
    return np.exp(beta*(np.sqrt(1.0-x2)-1.0)) * ok
@numba.vectorize
def mphi(x, beta):
    return nphi(x, beta)

def convolve_1(xs, big_y, ialpha, w2, beta):
    n = big_y.size
    h = 2*np.pi/n
    out = np.zeros(xs.size, dtype=big_y.dtype)
    for i in numba.prange(xs.size):
        x = xs[i]
        ind = int(x // h)
        min_ind = ind - w2
        max_ind = ind + w2 + 1
        for j in range(min_ind, max_ind):
            z = ialpha*(x - j*h)
            out[i] += nphi(z, beta)*big_y[j%n]
    return out
def convolve_batch(xs, big_y, ialpha, w2, beta):
    bn = big_y.shape[0]
    n = big_y.shape[1]
    h = 2*np.pi/n
    out = np.zeros((bn, xs.size), dtype=big_y.dtype)
    for i in numba.prange(xs.size):
        x = xs[i]
        ind = int(x // h)
        min_ind = ind - w2
        max_ind = ind + w2 + 1
        for j in range(min_ind, max_ind):
            jj = j % n
            z = ialpha*(x - j*h)
            ker = nphi(z, beta)
            for k in range(bn):
                out[k, i] += ker*big_y[k, jj]
    return out

convolve_1_serial = numba.njit(convolve_1, fastmath=True, parallel=False)
convolve_1_parallel = numba.njit(convolve_1, fastmath=True, parallel=True)
convolve_batch_serial = numba.njit(convolve_batch, fastmath=True, parallel=False)
convolve_batch_parallel = numba.njit(convolve_batch, fastmath=True, parallel=True)

class interp_numba_nufft:
    def __init__(self, fh, eps):
        """
        fh:  (n_trans, n_modes), stack of fourier modes for functions to interp
        """
        self.eps = eps
        self.w = 1 + int(np.ceil(np.log10(1/self.eps)))
        self.beta = 2.30 * self.w
        self.w2 = np.ceil(self.w/2)
        if len(fh.shape) == 1:
            self.serial_function = convolve_1_serial
            self.parallel_function = convolve_1_parallel
            self.bn = None
            self.n = fh.size
        else:
            self.serial_function = convolve_batch_serial
            self.parallel_function = convolve_batch_parallel
            self.bn = fh.shape[0]
            self.n = fh.shape[1]
        assert self.n % 2 == 0, 'fh must have even # of modes'
        self.n2 = int(self.n//2)
        self.fh = fh * self.n / (2*np.pi)
        self.x = np.linspace(0, 2*np.pi, 2*self.n, endpoint=False)
        self.alpha = np.pi*self.w/(2*self.n)
        self.ialpha = 1.0/self.alpha
        xpsi = np.fft.fftshift(mphi(self.ialpha*(self.x-np.pi), self.beta))
        xpsi_hat = np.fft.fft(xpsi)
        xpsi_hat = np.concatenate([xpsi_hat[:self.n2], xpsi_hat[-self.n2:]])
        self.pk = 4*np.pi/self.n/xpsi_hat
        self.fh_adj = self.pk*self.fh
        if self.bn is None:
            self.pad_fh_adj = np.zeros(2*self.n, dtype=complex)
            self.pad_fh_adj[:self.n//2] =  self.fh_adj[:self.n//2]
            self.pad_fh_adj[-self.n//2:] = self.fh_adj[-self.n//2:]
        else:
            self.pad_fh_adj = np.zeros((self.bn, 2*self.n), dtype=complex)
            self.pad_fh_adj[:, :self.n//2] =  self.fh_adj[:, :self.n//2]
            self.pad_fh_adj[:, -self.n//2:] = self.fh_adj[:, -self.n//2:]
        self.pad_f = np.fft.ifft(self.pad_fh_adj)
    def __call__(self, x):
        if x.size < 1000:
            return self.serial_function(x, self.pad_f, self.ialpha, self.w2, self.beta)
        else:
            return self.parallel_function(x, self.pad_f, self.ialpha, self.w2, self.beta)

################################################################################
# method using my hacked version of finufft

class interp_finufft:
    def __init__(self, fh, eps):
        """
        fh:  (n_trans, n_modes), stack of fourier modes for functions to interp
        """
        if len(fh.shape) == 1:
            self.n = fh.size
            self.plan = Plan(2, (self.n,), eps=eps, isign=1, modeord=1, chkbnds=0)
        else:
            self.n = fh.shape[1]
            self.plan = Plan(2, (self.n,), n_trans=fh.shape[0], eps=eps, isign=1, modeord=1, chkbnds=0)
        self.fh = fh / self.n
        self.plan.execute_type2_part1(self.fh)
    def __call__(self, x):
        self.plan.setpts(x)
        return self.plan.execute_type2_part2()

################################################################################
# one method to rule them all (at least unless i can reduce the overhead with finufft)

class periodic_interp1d:
    def __init__(self, f, eps=1e-14):
        """
        f: (n_func, n), stack of functions to interpolate
        """
        self.f = f
        self.n = self.f.shape[-1]
        self.nb = 1 if len(self.f.shape) == 1 else self.f.shape[0]
        self.dtype = self.f.dtype
        self.fh = np.fft.fft(self.f)
        self.finufft = interp_finufft(self.fh, eps)
        if self.n < 100:
            self.non_finufft = interp_numba_direct(self.fh)
        else:
            self.non_finufft = interp_numba_nufft(self.fh, eps)
    def __call__(self, x):
        if type(x) != np.ndarray:
            x = np.array([x])
            scalar = True
        else:
            scalar = False
        if x.size / self.nb > 50000:
            out = self.finufft(x).astype(self.dtype)
        else:
            out = self.non_finufft(x).astype(self.dtype)
        if scalar:
            return out[0]
        else:
            return out


"""

def allclose(x, y):
    return np.abs(x-y).max()

def test(n, m, eps=1e-14):

    print('\n\n--- Testing with', n, 'source points and', m, 'target points.')
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    y = np.exp(np.sin(x)) + 1j*np.exp(np.sin(2*x)-np.cos(3*x))
    yh = np.fft.fft(y)
    yhr = np.row_stack([yh, 2*yh, 3*yh])
    xc = np.random.rand(m) * 2*np.pi

    # direct eval
    interp = interp_numba_direct(yh)
    out1_dir = interp(xc)
    print('Timing direct, 1 density')
    %timeit -n 3 -r 5 out = interp(xc)
    interp = interp_numba_direct(yhr)
    out2_dir = interp(xc)
    print('Timing direct, 3 densities')
    %timeit -n 3 -r 5 out = interp(xc)

    # my nufft
    interp = interp_numba_nufft(yh, eps)
    out1_me = interp(xc)
    print('Timing mine, 1 density')
    %timeit -n 3 -r 5 out = interp(xc)
    interp = interp_numba_nufft(yhr, eps)
    out2_me = interp(xc)
    print('Timing mine, 3 densities')
    %timeit -n 3 -r 5 out = interp(xc)

    # finufft
    interp = interp_finufft(yh, eps)
    out1_fi = interp(xc)
    print('Timing FI, 1 density')
    %timeit -n 3 -r 5 out = interp(xc)
    interp = interp_finufft(yhr, eps)
    out2_fi = interp(xc)
    print('Timing FI, 3 densities')
    %timeit -n 3 -r 5 out = interp(xc)

    # finufft classic
    interp = interp_finufft_classic(yh, eps)
    out1_fi = interp(xc)
    print('Timing FI Classic, 1 density')
    %timeit -n 3 -r 5 out = interp(xc)
    interp = interp_finufft_classic(yhr, eps)
    out2_fi = interp(xc)
    print('Timing FI Classic, 3 densities')
    %timeit -n 3 -r 5 out = interp(xc)

    # finufft old
    interp = interp_finufft_old(yh, eps)
    out1_fi = interp(xc)
    print('Timing FI Old, 1 density')
    %timeit -n 3 -r 5 out = interp(xc)
    interp = interp_finufft_old(yhr, eps)
    out2_fi = interp(xc)
    print('Timing FI Old, 3 densities')
    %timeit -n 3 -r 5 out = interp(xc)

    print('')
    print('Difference, direct/nufft_me, 1 density:   {:0.2e}'.format(allclose(out1_dir, out1_me)))
    print('Difference, direct/nufft_me, 3 densities: {:0.2e}'.format(allclose(out2_dir, out2_me)))

    print('Difference, direct/finnufft, 1 density:   {:0.2e}'.format(allclose(out1_dir, out1_fi)))
    print('Difference, direct/finnufft, 3 densities: {:0.2e}'.format(allclose(out2_dir, out2_fi)))

test(50, 16)
test(50, 1000)
test(50, 100000)
test(50, 1000000)

test(200, 16)
test(200, 1000)
test(200, 20000)

test(1000, 16)
test(1000, 1000)
test(1000, 100000)

test(10000, 16)
test(10000, 1000)
#"""
