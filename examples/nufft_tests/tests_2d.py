import numpy as np
import numba
# from finufft import Plan
from function_generator import FunctionGenerator
from near_finder.nufft_interp2d import dirft2d_1_serial, dirft2d_1_parallel, dirft2d_batch_serial, dirft2d_batch_parallel
from near_finder.nufft_interp2d import interp2d_numba_direct, interp2d_numba_nufft, interp2d_finufft, periodic_interp2d
import time

def allclose(x, y):
    return np.abs(x-y).max()

def test(nx, ny, m, eps=1e-14):

    print('\n\n--- Testing with (', nx, ',', ny, ') source points and', m, 'target points.')
    vx = np.linspace(0, 2*np.pi, nx, endpoint=False)
    vy = np.linspace(0, 2*np.pi, ny, endpoint=False)
    x, y = np.meshgrid(vx, vy, indexing='ij')
    xt, yt = np.random.rand(m)*2*np.pi, np.random.rand(m)*2*np.pi
    f1 = lambda x, y: np.exp(np.sin(x))*np.cos(2*y)
    f2 = lambda x, y: np.sin(x)*np.log(2+np.cos(3*y))
    f3 = lambda x, y: np.cos(x)*np.exp(2+np.sin(3*y))
    fg1 = f1(x, y)
    fg2 = f2(x, y)
    fg3 = f3(x, y)
    fg = np.stack([fg1, fg2, fg3])
    ft1 = f1(xt, yt)
    ft2 = f2(xt, yt)
    ft3 = f3(xt, yt)
    ft = np.stack([ft1, ft2, ft3])
    fh1 = np.fft.fft2(fg1)
    fh = np.fft.fft2(fg)

    # direct eval
    if nx*ny*m < 100**3:
        did_direct = True
        interp = interp2d_numba_direct(fh[0])
        out1_dir = interp(xt, yt)
        print('Timing direct, 1 density')
        %timeit -n 3 -r 5 out = interp(xt, yt)
        interp = interp2d_numba_direct(fh)
        out2_dir = interp(xt, yt)
        print('Timing direct, 3 densities')
        %timeit -n 3 -r 5 out = interp(xt, yt)
    else:
        did_direct = False

    # my nufft
    st = time.time()
    interp = interp2d_numba_nufft(fh[0], eps)
    my_form1_time = time.time() - st
    out1_me = interp(xt, yt)
    print('Timing mine, 1 density')
    %timeit -n 3 -r 5 out = interp(xt, yt)
    st = time.time()
    interp = interp2d_numba_nufft(fh, eps)
    my_form_batch_time = time.time() - st
    out2_me = interp(xt, yt)
    print('Timing mine, 3 densities')
    %timeit -n 3 -r 5 out = interp(xt, yt)

    print('My NUFFT1 form time: {:0.1f}'.format(my_form1_time*1000))
    print('My NUFFTb form time: {:0.1f}'.format(my_form_batch_time*1000))

    print('')

    if did_direct:
        print('Difference, direct/nufft_me, 1 density:   {:0.2e}'.format(allclose(out1_dir, out1_me)))
        print('Difference, direct/nufft_me, 3 densities: {:0.2e}'.format(allclose(out2_dir, out2_me)))

    print('Error,      nufft_me/truth,    1 density:   {:0.2e}'.format(allclose(out1_me, ft[0])))
    print('Error,      nufft_me/truth,    3 densities: {:0.2e}'.format(allclose(out2_me, ft)))


test(50, 50, 16)
test(50, 100, 1000)
test(50, 50, 100000)

test(200, 200, 16)
test(200, 200, 1000)
test(200, 200, 20000)

test(1000, 1000, 16)
test(1000, 1000, 1000)
test(1000, 1000, 10000)

print('\n--- Testing on domain with different bounds ---')

vx = np.linspace(-1.0, 3.0, 100, endpoint=False)
vy = np.linspace(-1.0, 3.0, 200, endpoint=False)
x, y = np.meshgrid(vx, vy, indexing='ij')
xt, yt = np.random.rand(10,10)*4.0-1.0, np.random.rand(10,10)*4.0-1.0
f1 = lambda x, y: np.exp(np.sin(np.pi*x/2))*np.cos(np.pi*y)
f2 = lambda x, y: np.sin(2*np.pi)*np.log(2+np.cos(np.pi*y))
f3 = lambda x, y: np.cos(np.pi*x/2)*np.exp(2+np.sin(3*np.pi*y))
fg1 = f1(x, y)
fg2 = f2(x, y)
fg3 = f3(x, y)
fg = np.stack([fg1, fg2, fg3])
ft1 = f1(xt, yt)
ft2 = f2(xt, yt)
ft3 = f3(xt, yt)
ft = np.stack([ft1, ft2, ft3])
fh1 = np.fft.fft2(fg1)
fh = np.fft.fft2(fg)

interp = periodic_interp2d(fh=fh, eps=eps, xbounds=[-1.0, 3.0], ybounds=[-1.0, 3.0])
fe = interp(xt, yt)
print('...Error is: {:0.2e}'.format(allclose(fe, ft)))
