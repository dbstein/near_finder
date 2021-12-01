import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from near_finder.points_near_points import gridpoints_near_points, gridpoints_near_points_update, gridpoints_near_points_sparse, points_near_points_sparse
from near_finder.points_near_curve import gridpoints_near_curve, gridpoints_near_curve_sparse, gridpoints_near_curve_update
from near_finder.phys_routines import points_inside_curve, points_inside_curve_sparse, points_inside_curve_update
from near_finder.utilities import star

"""
Demonstration of near_finder utility
"""

################################################################################
# Setup

# number of points in grid (in each direction)
ng = 2000
# number of points in boundary (set to small to see accuracy, large to see speed)
nb = 20
# nb = 1000
verbose=False

# coordinates for grid
if nb == 100:
	xv = np.linspace(-1*int(ng/nb), 1*int(ng/nb), ng, endpoint=True)
	yv = np.linspace(-1*int(ng/nb), 1*int(ng/nb), ng, endpoint=True)
else:
	xv = np.linspace(-2, 2, ng, endpoint=True)
	yv = np.linspace(-2, 2, ng, endpoint=True)
x, y = np.meshgrid(xv, yv, indexing='ij')

# get a star boundary
bx, by = star(nb, a=0.3, f=5)
bxr = np.pad(bx, (0,1), mode='wrap')
byr = np.pad(by, (0,1), mode='wrap')

print('\n\nNear-Finder demonstration, on', ng, 'by', ng, 'grid, boundary has', nb, 'points.')
print('All times given in ms.')

################################################################################
# Test finding points near points

d = 0.05
n_close, x_ind, y_ind, ci, distance = gridpoints_near_points_sparse(bx, by, xv, yv, d)
st = time.time()
_ = gridpoints_near_points_sparse(bx, by, xv, yv, d)
time_near_points_sparse = time.time() - st
distance_sparse = distance.copy()
close_sparse = np.zeros(x.shape, dtype=bool)
close_sparse[x_ind, y_ind] = True

# output
print('Time for gridpoints near points finder (sparse):         {:0.1f}'.format(time_near_points_sparse*1000))

close, close_ind, distance = gridpoints_near_points(bx, by, xv, yv, d)
st = time.time()
close, close_ind, distance = gridpoints_near_points(bx, by, xv, yv, d)
time_near_points_dense = time.time() - st

# output
print('Time for gridpoints near points finder (dense):          {:0.1f}'.format(time_near_points_dense*1000))
print('Allclose, gridpoints near points finder:                ', np.allclose(close_sparse, close))

close = np.zeros(x.shape, dtype=bool)
int_helper1  = np.zeros(x.shape, dtype=int)
int_helper2  = np.zeros(x.shape, dtype=int)
float_helper = np.full(x.shape, np.Inf, dtype=float)
bool_helper  = np.zeros(x.shape, dtype=bool)
_ = gridpoints_near_points_update(bx, by, xv, yv, d, 1, close, int_helper1, int_helper2, float_helper, bool_helper)
st = time.time()
nc, idx, idy, gi = gridpoints_near_points_update(bx, by, xv, yv, d, 1, close, int_helper1, int_helper2, float_helper, bool_helper)
time_near_points_update = time.time() - st

# output
print('Time for gridpoints near points finder (update):         {:0.1f}'.format(time_near_points_update*1000))

if ng <= 500:
	st = time.time()
	n_close, ind, guess_ind, dists = points_near_points_sparse(d, bx, by, x, y)
	time_points_near_points_sparse = time.time() - st
	general_close = np.zeros(x.shape, dtype=bool)
	general_close.ravel()[ind] = True

	print('Time for points near general points finder (sparse): {:0.1f}'.format(time_points_near_points_sparse*1000))
	print('Allclose, points near points finder:                ', np.allclose(general_close, close))

################################################################################
# Test finding points near curve

spresult = gridpoints_near_curve_sparse(bx, by, xv, yv, d, interpolation_scheme='polyi', verbose=False)
st = time.time()
n_close, x_ind, y_ind, spr, spt, (d, cx, cy) = gridpoints_near_curve_sparse(bx, by, xv, yv, d, interpolation_scheme='polyi', verbose=False)
time_near_curve_sparse = time.time() - st
print('Time for gridpoints near curve finder (sparse/polyi):    {:0.1f}'.format(time_near_curve_sparse*1000))

spresult = gridpoints_near_curve_sparse(bx, by, xv, yv, d, verbose=False)
st = time.time()
n_close, x_ind, y_ind, spr, spt, (d, cx, cy) = gridpoints_near_curve_sparse(bx, by, xv, yv, d, verbose=verbose)
time_near_curve_sparse_nufft = time.time() - st
print('Time for gridpoints near curve finder (sparse/nufft):    {:0.1f}'.format(time_near_curve_sparse_nufft*1000))

in_annulus_sparse = np.zeros(x.shape, dtype=bool)
in_annulus_sparse[x_ind, y_ind] = True
r_sparse = np.zeros(x.shape)
r_sparse[x_ind, y_ind] = spr

result = gridpoints_near_curve(bx, by, xv, yv, d, verbose=False)
st = time.time()
in_annulus, r, t, _ = gridpoints_near_curve(bx, by, xv, yv, d, verbose=verbose)
time_near_curve_dense = time.time() - st
print('Time for gridpoints near curve finder (dense/nufft):     {:0.1f}'.format(time_near_curve_dense*1000))
print('Allclose, gridpoints near curve finder:                 ', np.allclose(in_annulus_sparse, in_annulus))

################################################################################
# Test finding points inside curve

spinside, xm, xM, ym, yM = points_inside_curve_sparse(xv, yv, spresult)
st = time.time()
_ = points_inside_curve_sparse(xv, yv, spresult)
time_inside_curve_sparse = time.time() - st
print('Time for points inside curve finder (sparse):            {:0.1f}'.format(time_inside_curve_sparse*1000))

inside2 = np.zeros(x.shape, dtype=bool)
points_inside_curve_update(xv, yv, spresult, inside2)
inside2[:] = False
st = time.time()
points_inside_curve_update(xv, yv, spresult, inside2)
time_inside_curve_update = time.time() - st
print('Time for points inside curve finder (udpate):            {:0.1f}'.format(time_inside_curve_update*1000))
print('Allclose, gridpoints inside curve finder:               ', np.allclose(spinside, inside2[xm:xM,ym:yM]))

inside = points_inside_curve(x, y, result)
st = time.time()
_ = points_inside_curve(x, y, result)
time_inside_curve_dense = time.time() - st
print('Time for points inside curve finder (dense):             {:0.1f}'.format(time_inside_curve_dense*1000))
print('Allclose, gridpoints inside curve finder:               ', np.allclose(spinside, inside[xm:xM,ym:yM]))

if True:
	fig, ax = plt.subplots()
	ax.pcolormesh(x[xm:xM,ym:yM], y[xm:xM,ym:yM], inside[xm:xM,ym:yM])
	ax.plot(bxr, byr, color='white')
	ax.set_title('Inside Curve (dense)')

	fig, ax = plt.subplots()
	ax.pcolormesh(x[xm:xM,ym:yM], y[xm:xM,ym:yM], spinside)
	ax.plot(bxr, byr, color='white')
	ax.set_title('Inside Curve (sparse)')

	fig, ax = plt.subplots()
	ax.pcolormesh(x[xm:xM,ym:yM], y[xm:xM,ym:yM], inside2[xm:xM,ym:yM])
	ax.plot(bxr, byr, color='white')
	ax.set_title('Inside Curve (update)')



