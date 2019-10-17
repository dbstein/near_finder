import numpy as np
import numba
from .utilities import compute_curvature, compute_speed, upsample
from .coordinate_routines import compute_local_coordinates

def gridpoints_near_curve(cx, cy, xv, yv, d, tol=1e-14, verbose=False):
    """
    Computes, for all gridpoints, whether the gridpoints
    1) are within d of the (closed) curve
    2) for those that are within d of the curve,
        how far the gridpoints are from the curve
        (that is, closest approach in the normal direction)
    3) local coordinates, in (r, t) coords, for the curve
        that is, we assume the curve is given by X(t) = (cx(t_i), cy(t_i))
        we define local coordinates X(t, r) = X(t) + n(t) r, where n is
        the normal to the curve at t,
        and return (r, t) for each gridpoint in some search region

    Inputs:
        cx,  float(nb): x-coordinates of boundary
        cy,  float(nb): y-coordinates of boundary
        xv,  float(nx): x-values for grid coordinates
        yv,  float(ny): y-values for grid coordinates
        d,   float:     distance to search within  
        tol, float:     tolerance to be passed to Newton solver for coords
        verbose, bool:  flag passed to coord solver for verbose output
    Outputs: (tuple of)
        in_annulus, bool (nx, ny), whether points are in annulus of radius d
        r,          float(nx, ny), r-coordinate for points in_annulus
        t,          float(nx, ny), t-coordinate for points in_annulus
        (d, cx, cy) float,         search distance, upsampled cx, cy
    """

    # compute the speed of the approximation
    n = cx.size
    dt = 2*np.pi/n
    speed = compute_speed(cx, cy)
    max_h = np.max(speed)*dt
    # if the curve is too poorly resolved to compute things accurate, upsample
    if max_h > d:
        n *= int(np.ceil(max_h/d))
        cx, cy = upsample(cx, n), upsample(cy, n)
    # find all candidate points
    D = 1.5*d # extra large fudge factor because its a curve
    near, guess_ind, close = gridpoints_near_points(cx, cy, xv, yv, d)
    # for all candidate points, perform brute force checking
    wh = np.where(near)
    x_test = xv[wh[0]]
    y_test = yv[wh[1]]
    gi = guess_ind[wh[0], wh[1]]
    t_test, r_test = compute_local_coordinates(cx, cy, x_test, y_test,
                                newton_tol=tol, guess_ind=gi, verbose=verbose)
    # initialize output matrices
    sh = (xv.size, yv.size)
    in_annulus = np.zeros(sh, dtype=bool)
    r = np.zeros(sh, dtype=float)
    t = np.zeros(sh, dtype=float)
    # for those found by the coarse search, update based on values
    init = np.abs(r_test) <= d
    in_annulus[near] = init
    r[in_annulus] = r_test[init]
    t[in_annulus] = t_test[init]
    return in_annulus, r, t, (d, cx, cy)

def gridpoints_near_points(bx, by, xv, yv, d):
    """
    Fast near-points finder for a grid and set of points. 

    Returns a boolean array with size [xv.size, yv.size]
    The elements of the boolean array give whether that gridpoint is within
    d of any of the points bx/by

    When bx/by describe a polygon, one may use this function to find all points
    within a distance D of the polygon, by setting:
    d = sqrt(D^2 + (l/2)^2), where l is the length of the longest polygonal
    segment.  If l < D, then d need only be 1.12D to guarantee all near-points
    are found.  Note that points that are not within D of the polygon will aslo
    be marked as "near", however

    Inputs:
        bx, float(nb): x-coordinates of boundary
        by, float(nb): y-coordinates of boundary
        xv, float(nx): x-values for grid coordinates
        yv: float(ny): y-values for grid coordinates
        d:  distance to find near points
    Outputs:
        close,     bool(nx, ny),  is this point within d of any boundary point?
        guess_ind, int(nx, ny),   index of closest boundary point to this point
        closest,   float(nx, ny), closest distance to a boundary point
    """
    sh = (xv.shape[0], yv.shape[0])
    close     = np.zeros(sh, dtype=int)
    guess_ind = np.zeros(sh, dtype=int)   - 1
    closest   = np.zeros(sh, dtype=float) + 1e15

    _grid_near_points(bx, by, xv, yv, d, close, guess_ind, closest)
    return close > 0, guess_ind, closest

@numba.njit(parallel=True)
def _grid_near_points(x, y, xv, yv, d, close, gi, closest):
    N = x.shape[0]
    Nx = xv.shape[0]
    Ny = yv.shape[0]
    xh = xv[1] - xv[0]
    yh = yv[1] - yv[0]
    xsd = d//xh + 1
    ysd = d//yh + 1
    d2 = d*d
    xlb = xv[0]
    ylb = yv[0]
    for i in numba.prange(N):
        x_loc = (x[i] - xlb) // xh
        y_loc = (y[i] - ylb) // yh
        x_lower = max(x_loc - xsd, 0)
        x_upper = min(x_loc + xsd + 1, Nx)
        y_lower = max(y_loc - ysd, 0)
        y_upper = min(y_loc + ysd + 1, Ny)
        for j in range(x_lower, x_upper):
            for k in range(y_lower, y_upper):
                xd = xv[j] - x[i]
                yd = yv[k] - y[i]
                dist2 = xd**2 + yd**2
                close[j, k] += int(dist2 < d2)
                if dist2 < closest[j, k]:
                    closest[j, k] = dist2
                    gi[j, k] = i

