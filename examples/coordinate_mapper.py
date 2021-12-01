import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import time
from near_finder.general_tree import LightweightCoordinateTree, FullCoordinateTree

from near_finder.utilities import star
from near_finder.points_near_curve import gridpoints_near_curve, points_near_curve
from pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary import Global_Smooth_Boundary as GSB

"""
Demonstration of coordinate_mapper utility
"""

################################################################################
# Setup

# number of points in boundary (set to small to see accuracy, large to see speed)
nb = 800

# where we need to find coordinates to (in distance of h from boundary)
coordinate_distance = 10

# tolerance for newton solves
tol = 1.0e-12

# get a star boundary
bx, by = star(nb, a=0.2, f=5)
bdy = GSB(bx, by)
max_h = bdy.speed.max()*bdy.dt
ng = 2 * int(2 / max_h)

# coordinates for grid
xv = np.linspace(-2, 2, ng, endpoint=False)
yv = np.linspace(-2, 2, ng, endpoint=False)
x, y = np.meshgrid(xv, yv, indexing='ij')

print('\n\nCoordinate mapper demonstration, on', ng, 'by', ng, 'grid, boundary has', nb, 'points.')
print('All times given in ms.')

h = xv[1] - xv[0]
coord = coordinate_distance*h

################################################################################
# Get jittered test points
cx = x + np.random.rand(*x.shape)*0.5*h
cy = y + np.random.rand(*x.shape)*0.5*h

################################################################################
# Function to get comparable output

def _modularize(in_coordinates, rs, ts, inner_dist, outer_dist):
    inc = in_coordinates
    good_r = np.logical_and(rs[inc] >= -inner_dist, rs[inc] <= outer_dist)
    outc = np.zeros(rs.shape, dtype=bool)
    outr = np.zeros_like(rs)
    outt = np.zeros_like(rs)
    outc[inc] = good_r
    outr[outc] = rs[outc]
    outt[outc] = ts[outc]
    outt[outt < 0.0] += 2*np.pi
    outt[outt >= 2*np.pi] -= 2*np.pi
    return outc, outr, outt
def modularize(in_coordinates, rs, ts):
    return _modularize(in_coordinates, rs, ts, coord, 0.0)

def tree_modularize(out):
    outc = out[0] == 1
    outr = np.zeros(outc.shape, dtype=float)
    outt = np.zeros(outc.shape, dtype=float)
    outr[outc] = out[3][outc]
    outt[outc] = out[4][outc]
    return outc, outr, outt

################################################################################
# Compute the coordinates for the coor_nodes directly

do_direct = nb <= 1000
if do_direct:
    st = time.time()
    in_coor, r_dir, t_dir, _t = points_near_curve(bx, by, cx, cy, coord, tol=tol)
    direct_time = time.time() - st
    in_coorg = in_coor.reshape(cx.shape)
    r_dirg = r_dir.reshape(cx.shape)
    t_dirg = t_dir.reshape(cx.shape)

    # make output comparable
    c_dir, r_dir, t_dir = modularize(in_coor, r_dir, t_dir)
    a_dir = (c_dir, r_dir, t_dir)

################################################################################
# Using gridpoints near curve

in_grid, r_grid, t_grid, _ = gridpoints_near_curve(bx, by, xv, yv, coord, tol=tol)
st = time.time()
in_grid, r_grid, t_grid, _ = gridpoints_near_curve(bx, by, xv, yv, coord, tol=tol)
gridpoints_time = time.time() - st
# make output comparable
c_gnpc, r_gnpc, t_gnpc = modularize(in_grid, r_grid, t_grid)
a_gnpc = (c_gnpc, r_gnpc, t_gnpc)

################################################################################
# Build Full Coordinate Tree

FTree = FullCoordinateTree(bdy, coord, 0.0, parameters={'order' : 12, 'tol' : tol})
st=time.time()
FTree = FullCoordinateTree(bdy, coord, 0.0, parameters={'order' : 12, 'tol' : tol})
ftree_form_time = time.time() - st

# plot
fig, ax = plt.subplots()
FTree.plot(ax, mpl)

# classify
out = FTree(cx, cy)
st = time.time()
out = FTree(cx, cy)
ftree_coordinate_time = time.time() - st

# make output comparable
cd_ftree, rd_ftree, td_ftree = tree_modularize(out)
ad_ftree = (cd_ftree, rd_ftree, td_ftree)

# grid-classify
out = FTree(xv, yv, grid=True)
st = time.time()
out = FTree(xv, yv, grid=True)
ftree_grid_coordinate_time = time.time() - st

# make output comparable
cg_ftree, rg_ftree, tg_ftree = tree_modularize(out)
ag_ftree = (cg_ftree, rg_ftree, tg_ftree)

################################################################################
# Build Lightweight Coordinate Tree

LTree = LightweightCoordinateTree(bdy, coord, 0.0)
st=time.time()
LTree = LightweightCoordinateTree(bdy, coord, 0.0)
ltree_form_time = time.time() - st

# plot
fig, ax = plt.subplots()
LTree.plot(ax, mpl)

# classify
out = LTree(cx, cy, newton_tol=tol)
st = time.time()
out = LTree(cx, cy, newton_tol=tol)
ltree_coordinate_time = time.time() - st

# make output comparable
cd_ltree, rd_ltree, td_ltree = tree_modularize(out)
ad_ltree = (cd_ltree, rd_ltree, td_ltree)

# grid-classify
out = LTree(xv, yv, grid=True, newton_tol=tol)
st = time.time()
out = LTree(xv, yv, grid=True, newton_tol=tol)
ltree_grid_coordinate_time = time.time() - st

# make output comparable
cg_ltree, rg_ltree, tg_ltree = tree_modularize(out)
ag_ltree = (cg_ltree, rg_ltree, tg_ltree)

################################################################################
# Analyze Error

def get_errors(c1, r1, t1, c2, r2, t2):
    # where they disagree
    disagreements = c1 != c2
    n_disagreements = np.sum(disagreements)
    print(".... Number of disagreements is:", n_disagreements)
    # where they both agree the point is within coordinates
    agree_incoor = np.logical_and(c1, c2)
    r_err = np.abs(r1[agree_incoor]-r2[agree_incoor]).max()
    t_err = np.abs(t1[agree_incoor]-t2[agree_incoor]).max()
    print(".... r error, where agree:       {:0.2e}".format(r_err))
    print(".... t error, where agree:       {:0.2e}".format(t_err))
    if t_err > np.pi:
        adjusted_t_err = np.abs(t_err - 2*np.pi)
        print(".... adj. t error, where agree:  {:0.2e}".format(adjusted_t_err))
    if n_disagreements > 0:
        r_disagree = np.abs(r1[disagreements] - r2[disagreements]).max()
        print('.... r error, where disagree:    {:0.2e}'.format(r_disagree))

################################################################################
# Get the difference between these things

print('\n--- Errors on general points ---')
if do_direct:
    print('.. direct vs. ftree:')
    get_errors(*a_dir, *ad_ftree)
    print('.. direct vs. ltree')
    get_errors(*a_dir, *ad_ltree)
print('.. ftree vs. ltree')
get_errors(*ad_ftree, *ad_ltree)

print('\n--- Errors on gridpoints ---')
print('.. direct vs. ftree:')
get_errors(*a_gnpc, *ag_ftree)
print('.. direct vs. ltree')
get_errors(*a_gnpc, *ag_ltree)
print('.. ftree vs. ltree')
get_errors(*ag_ftree, *ag_ltree)

print('\n--- Timings, general points ---')
if do_direct:
    print('.. direct:                     {:0.2f}'.format(direct_time*1000))
print('.. full tree:                  {:0.2f}'.format(ftree_coordinate_time*1000))
print('.. lightweight tree:           {:0.2f}'.format(ltree_coordinate_time*1000))

print('\n--- Timings, gridpoints ---')
print('.. gridpoints near curve:      {:0.2f}'.format(gridpoints_time*1000))
print('.. full tree, grid:            {:0.2f}'.format(ftree_grid_coordinate_time*1000))
print('.. lightweight tree, grid:     {:0.2f}'.format(ltree_grid_coordinate_time*1000))

print('\n Timings, tree formation')
print('Time, full tree form:          {:0.2f}'.format(ftree_form_time*1000))
print('Time, lightweight tree form:   {:0.2f}'.format(ltree_form_time*1000))
