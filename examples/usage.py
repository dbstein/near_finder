import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from near_finder.near_routines import gridpoints_near_points, gridpoints_near_curve
from near_finder.phys_routines import gridpoints_inside_curve
from near_finder.utilities import star

"""
Demonstration of near_finder utility
"""

################################################################################
# Setup

# number of points in grid (in each direction)
ng = 200
# number of points in boundary (set to small to see accuracy, large to see speed)
nb = 16
# nb = 1000

# coordinates for grid
xv = np.linspace(-1.3, 1.3, ng, endpoint=True)
yv = np.linspace(-1.3, 1.3, ng, endpoint=True)
x, y = np.meshgrid(xv, yv, indexing='ij')

# get a star boundary
bx, by = star(nb, a=0.2, f=5)
bxr = np.pad(bx, (0,1), mode='wrap')
byr = np.pad(by, (0,1), mode='wrap')

print('\n\nNear-Finder demonstration, on', ng, 'by', ng, 'grid, boundary has', nb, 'points.')
print('All times given in ms.')

################################################################################
# Test finding points near points

d = 0.05
# first run to compile numba functions
close, guess, closest = gridpoints_near_points(bx, by, xv, yv, d)
# run again for timing
st = time.time()
close, guess, closest = gridpoints_near_points(bx, by, xv, yv, d)
time_near_points = time.time() - st

# output
print('Time for points near points finder: {:0.1f}'.format(time_near_points*1000))
fig, ax = plt.subplots()
ax.pcolormesh(x, y, close)
ax.scatter(bx, by, color='white')
ax.set_title('Near Points')

################################################################################
# Test finding points near curve

result = gridpoints_near_curve(bx, by, xv, yv, d, verbose=True)
st = time.time()
in_annulus, r, t, _ = gridpoints_near_curve(bx, by, xv, yv, d, verbose=False)
time_near_curve = time.time() - st

# output
print('Time for points near curve finder: {:0.1f}'.format(time_near_curve*1000))
fig, ax = plt.subplots()
ax.pcolormesh(x, y, in_annulus)
ax.plot(bxr, byr, color='white')
ax.set_title('Near Curve')

fig, ax = plt.subplots()
clf = ax.pcolormesh(x, y, r)
ax.plot(bxr, byr, color='white')
ax.set_title('r-coordinate')
plt.colorbar(clf)

fig, ax = plt.subplots()
clf = ax.pcolormesh(x, y, t)
ax.plot(bxr, byr, color='white')
ax.set_title('t-coordinate')
plt.colorbar(clf)

################################################################################
# Test finding points inside curve

inside = gridpoints_inside_curve(x, y, result)
st = time.time()
inside = gridpoints_inside_curve(x, y, result)
time_inside_curve = time.time() - st

print('Time for points inside curve finder: {:0.1f}'.format(time_inside_curve*1000))
fig, ax = plt.subplots()
ax.pcolormesh(x, y, inside)
ax.plot(bxr, byr, color='white')
ax.set_title('Inside Curve')
