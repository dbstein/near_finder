import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from near_finder.points_near_points import gridpoints_near_points_update
from near_finder.points_near_curve import gridpoints_near_curve_update
from near_finder.phys_routines import points_inside_curve_update
from near_finder.utilities import star
from near_finder.points_near_curve import _upsample_curve

"""
Demonstration of near_finder utility
"""

################################################################################
# Setup

ng                   = 300     # number of points in grid (in each direction)
nb1                  = 200       # number of points in boundary 1
nb2                  = 200       # number of points in boundary 2
nb3                  = 300       # number of points in boundary 3
nb4                  = 200       # number of points in boundary 4
verbose              = True    # provide verbose output in coordinate solving
interpolation_scheme = 'polyi'  # 'polyi' or 'nufft'
                                # note that verbose=True does nothing for 'polyi'
tol                  = 1e-12
max_iterations       = 10

# coordinates for grid
xv = np.linspace(-10, 10, ng, endpoint=True)
yv = np.linspace(-10, 10, ng, endpoint=True)
x, y = np.meshgrid(xv, yv, indexing='ij')

# outer boundary
bx1, by1 = star(nb1, r=9, a=0.05, f=3)
# inner boundaries
bx2, by2 = star(nb2, x=1.55,  y=3,  r=4.1, a=0.2,  f=5)
bx3, by3 = star(nb3, x=-5,    y=4,  r=2.2, a=0.25, f=7)
bx4, by4 = star(nb4, x=-5.44, y=-4, r=2,   a=0.3,  f=3)

print('\n\nNear-Finder demonstration, on', ng, 'by', ng, 'grid.')
print('All times given in ms.')

################################################################################
# Test finding points near points

d = 0.05
close        = np.zeros(x.shape, dtype=bool)
int_helper1  = np.zeros(x.shape, dtype=int)
int_helper2  = np.zeros(x.shape, dtype=int)
float_helper = np.full(x.shape, np.Inf, dtype=float)
bool_helper  = np.zeros(x.shape, dtype=bool)
_ = gridpoints_near_points_update(bx1, by1, xv, yv, d, 1, close, int_helper1, int_helper2, float_helper, bool_helper)
st = time.time()
nc1, idx1, idy1, gi1 = gridpoints_near_points_update(bx1, by1, xv, yv, d, 1, close, int_helper1, int_helper2, float_helper, bool_helper)
nc2, idx2, idy2, gi2 = gridpoints_near_points_update(bx2, by2, xv, yv, d, 2, close, int_helper1, int_helper2, float_helper, bool_helper)
nc3, idx3, idy3, gi3 = gridpoints_near_points_update(bx3, by3, xv, yv, d, 3, close, int_helper1, int_helper2, float_helper, bool_helper)
nc4, idx4, idy4, gi4 = gridpoints_near_points_update(bx4, by4, xv, yv, d, 4, close, int_helper1, int_helper2, float_helper, bool_helper)
time_near_points = time.time() - st
print('Time for points near points finder:             {:0.1f}'.format(time_near_points*1000))

################################################################################
# Test finding points near curve

close        = np.zeros(x.shape, dtype=bool)
int_helper1  = np.zeros(x.shape, dtype=int)
int_helper2  = np.zeros(x.shape, dtype=int)
float_helper = np.full(x.shape, np.Inf, dtype=float)
bool_helper  = np.zeros(x.shape, dtype=bool)
_ = gridpoints_near_curve_update(bx1, by1, xv, yv, d, 1, close, int_helper1, int_helper2, float_helper, bool_helper, interpolation_scheme=interpolation_scheme, verbose=verbose)
st = time.time()
res1 = gridpoints_near_curve_update(bx1, by1, xv, yv, d, 1, close, int_helper1, int_helper2, float_helper, bool_helper, interpolation_scheme=interpolation_scheme, verbose=verbose, tol=tol, max_iterations=max_iterations)
res2 = gridpoints_near_curve_update(bx2, by2, xv, yv, d, 2, close, int_helper1, int_helper2, float_helper, bool_helper, interpolation_scheme=interpolation_scheme, verbose=verbose, tol=tol, max_iterations=max_iterations)
res3 = gridpoints_near_curve_update(bx3, by3, xv, yv, d, 3, close, int_helper1, int_helper2, float_helper, bool_helper, interpolation_scheme=interpolation_scheme, verbose=verbose, tol=tol, max_iterations=max_iterations)
res4 = gridpoints_near_curve_update(bx4, by4, xv, yv, d, 4, close, int_helper1, int_helper2, float_helper, bool_helper, interpolation_scheme=interpolation_scheme, verbose=verbose, tol=tol, max_iterations=max_iterations)
time_near_points = time.time() - st
print('Time for points near curve finder:              {:0.1f}'.format(time_near_points*1000))

################################################################################
# Test finding points near points finder on upsampled boundary

_close        = np.zeros(x.shape, dtype=bool)
_int_helper1  = np.zeros(x.shape, dtype=int)
_int_helper2  = np.zeros(x.shape, dtype=int)
_float_helper = np.full(x.shape, np.Inf, dtype=float)
_bool_helper  = np.zeros(x.shape, dtype=bool)
st = time.time()
nc1, idx1, idy1, gi1 = gridpoints_near_points_update(res1[5][1], res1[5][2], xv, yv, d, 1, _close, _int_helper1, _int_helper2, _float_helper, _bool_helper)
nc2, idx2, idy2, gi2 = gridpoints_near_points_update(res2[5][1], res2[5][2], xv, yv, d, 2, _close, _int_helper1, _int_helper2, _float_helper, _bool_helper)
nc3, idx3, idy3, gi3 = gridpoints_near_points_update(res3[5][1], res3[5][2], xv, yv, d, 3, _close, _int_helper1, _int_helper2, _float_helper, _bool_helper)
nc4, idx4, idy4, gi4 = gridpoints_near_points_update(res4[5][1], res4[5][2], xv, yv, d, 4, _close, _int_helper1, _int_helper2, _float_helper, _bool_helper)
time_near_points = time.time() - st
print('Time for points near points finder (upsampled): {:0.1f}'.format(time_near_points*1000))

################################################################################
# Test finding physical region (inside b1, outside b2, b3)

phys = np.zeros(x.shape, dtype=bool)
points_inside_curve_update(xv, yv, res3, phys, inside=True)
points_inside_curve_update(xv, yv, res3, phys, inside=False)
phys = np.zeros(x.shape, dtype=bool)
st = time.time()
points_inside_curve_update(xv, yv, res1, phys, inside=True)
points_inside_curve_update(xv, yv, res2, phys, inside=False)
points_inside_curve_update(xv, yv, res3, phys, inside=False)
points_inside_curve_update(xv, yv, res4, phys, inside=False)
time_phys = time.time() - st
print('Time for phys finder:                           {:0.1f}'.format(time_phys*1000))

################################################################################
# Make graph

bx1r, by1r = np.pad(res1[5][1], (0,1), mode='wrap'), np.pad(res1[5][2], (0,1), mode='wrap')
bx2r, by2r = np.pad(res2[5][1], (0,1), mode='wrap'), np.pad(res2[5][2], (0,1), mode='wrap')
bx3r, by3r = np.pad(res3[5][1], (0,1), mode='wrap'), np.pad(res3[5][2], (0,1), mode='wrap')
bx4r, by4r = np.pad(res4[5][1], (0,1), mode='wrap'), np.pad(res4[5][2], (0,1), mode='wrap')

fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2,2)
ax1.imshow(phys.T[::-1],  extent=[-10,10,-10,10])
ax2.imshow(close.T[::-1], extent=[-10,10,-10,10])
ax3.imshow(phys.T[::-1],  extent=[-10,10,-10,10])
ax4.imshow(close.T[::-1], extent=[-10,10,-10,10])
ss = [10, 10, 20, 20]
for ax in [ax3, ax4]:
    ax.set(xlim=(-7.5,-5.5), ylim=(-6.5,-4.5))
for ax, s in zip([ax1, ax2, ax3, ax4], ss):
    ax.plot(bx1r, by1r, color='white')
    ax.plot(bx2r, by2r, color='white')
    ax.plot(bx3r, by3r, color='white')
    ax.plot(bx4r, by4r, color='white')
    ax.scatter(bx1, by1, color='white', s=s)
    ax.scatter(bx2, by2, color='white', s=s)
    ax.scatter(bx3, by3, color='white', s=s)
    ax.scatter(bx4, by4, color='white', s=s)
    ax.set(xticks=[], xticklabels=[], yticks=[], yticklabels=[])
fig.tight_layout()
