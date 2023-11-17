import numpy as np
import numba
# from finufft import Plan
from function_generator import FunctionGenerator
from near_finder.nufft_interp import interp_numba_direct, interp_numba_nufft, interp_finufft, periodic_interp1d

def allclose(x, y):
    return np.abs(x-y).max()

n = 100
m = 100
eps = 1e-14

print('\n\n--- Testing with', n, 'source points and', m, 'target points.')
x = np.linspace(0, 2*np.pi, n, endpoint=False)
y = np.exp(np.sin(x)) + 1j*np.exp(np.sin(2*x)-np.cos(3*x))
yh = np.fft.fft(y)
yhr = np.row_stack([yh, 2*yh, 3*yh])
xc = np.random.rand(m) * 2*np.pi

# interp = interp_finufft(yh, eps, nthreads=1, maxbatchsize=1, spread_thread=1)#, debug=2, spread_debug=2)
# out1_fi = interp(xc)



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
    # interp = interp_finufft(yh, eps)
    # out1_fi = interp(xc)
    # print('Timing FI, 1 density')
    # %timeit -n 3 -r 5 out = interp(xc)
    # interp = interp_finufft(yhr, eps)
    # out2_fi = interp(xc)
    # print('Timing FI, 3 densities')
    # %timeit -n 3 -r 5 out = interp(xc)

    # finufft (via nice interface)
    # interp = periodic_interp1d(f=y, eps=eps)
    # out1_pi = interp(xc)
    # print('Timing PI, 1 density')
    # %timeit -n 3 -r 5 out = interp(xc)
    # interp = periodic_interp1d(fh=yhr, eps=eps)
    # out2_pi = interp(xc)
    # print('Timing PI, 3 densities')
    # %timeit -n 3 -r 5 out = interp(xc)


    print('')
    print('Difference, direct/nufft_me,        1 density:   {:0.2e}'.format(allclose(out1_dir, out1_me)))
    print('Difference, direct/nufft_me,        3 densities: {:0.2e}'.format(allclose(out2_dir, out2_me)))

test(50, 2)
test(50, 10)
test(50, 20)
test(50, 50)
test(50, 200)
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
