import numpy as np
import numba
import numexpr as ne
import scipy as sp
import scipy.spatial
from .utilities import compute_curvature, compute_speed, upsample, extend_array, inarray
from .coordinate_routines import compute_local_coordinates
try:
    from array_list.array_list import create_selector
    have_array_list = True
except:
    have_array_list = False
    import warnings
    warnings.warn('array_list package is not found; points_near_points and points_near_curve will be slower.')

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
    near, guess_ind, close = gridpoints_near_points(cx, cy, xv, yv, D)
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

    close = np.zeros(sh, dtype=int)
    guess_ind = np.full(sh, -1, dtype=int)
    closest = np.full(sh, 1e15, dtype=float)

    _grid_near_points(bx, by, xv, yv, d, close, guess_ind, closest)
    return ne.evaluate('close > 0'), guess_ind, closest

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

def points_near_curve(cx, cy, x, y, d, tol=1e-14, verbose=False):
    """
    Computes, for all points, whether the points
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
        x,   float(*):  x-values for test points
        y,   float(*):  y-values for test points
        d,   float:     distance to search within  
        tol, float:     tolerance to be passed to Newton solver for coords
        verbose, bool:  flag passed to coord solver for verbose output
    Outputs: (tuple of)
        in_annulus, bool (*), whether points are in annulus of radius d
        r,          float(*), r-coordinate for points in_annulus
        t,          float(*), t-coordinate for points in_annulus
        (d, cx, cy) float,         search distance, upsampled cx, cy
    """
    sh = x.shape
    x = x.ravel()
    y = y.ravel()
    sz = x.size

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
    near, guess_ind, close = points_near_points(d, bx=cx, by=cy, tx=x, ty=y)
    # initialize output matrices
    in_annulus = np.zeros(sz, dtype=bool)
    r = np.zeros(sz, dtype=float)
    t = np.zeros(sz, dtype=float)
    if np.sum(near) > 0:
        # for all candidate points, perform brute force checking
        x_test = x[near]
        y_test = y[near]
        gi = guess_ind[near]
        t_test, r_test = compute_local_coordinates(cx, cy, x_test, y_test,
                                    newton_tol=tol, guess_ind=gi, verbose=verbose)
        # for those found by the coarse search, update based on values
        init = np.abs(r_test) <= d
        in_annulus[near] = init
        r[in_annulus] = r_test[init]
        t[in_annulus] = t_test[init]
        in_annulus = in_annulus.reshape(sh)
        r = r.reshape(sh)
        t = t.reshape(sh)
    not_in_annlus = np.logical_not(in_annulus)
    r[not_in_annlus] = np.nan
    t[not_in_annlus] = np.nan
    return in_annulus, r, t, (d, cx, cy)

@numba.njit()
def _wrangle_groups(bx, by, tx, ty, Groups, close, guess_ind, dists):
    N = tx.size
    for i in range(N):
        g = Groups.get(i)
        close[i] = g.size > 0
        if close[i]:
            txi = tx[i]
            tyi = ty[i]
            min_ind = -1
            min_d2 = 1e15
            for j in range(g.size):
                gj = g[j]
                dx = txi - bx[gj]
                dy = tyi - by[gj]
                d2 = dx**2 + dy**2
                if d2 < min_d2:
                    min_ind = j
                    min_d2 = d2
            if min_ind >= 0:
                guess_ind[i] = g[min_ind]
                dists[i] = np.sqrt(min_d2)

def points_near_points(d, bx=None, by=None, btree=None, tx=None, ty=None, ttree=None):
    """
    Fast tree based near-points finder for a set of test points and
        set of boundary points

    Returns a boolean array with size x.shape
    The elements of the boolean array give whether that gridpoint is within
    d of any of the points bx/by

    When bx/by describe a polygon, one may use this function to find all points
    within a distance D of the polygon, by setting:
    d = sqrt(D^2 + (l/2)^2), where l is the length of the longest polygonal
    segment.  If l < D, then d need only be 1.12D to guarantee all near-points
    are found.  Note that points that are not within D of the polygon will aslo
    be marked as "near", however

    Inputs:
        d:  distance to find near points
        bx, float(nb): x-coordinates of boundary
        by, float(nb): y-coordinates of boundary
        btree, cKDTree for bx, by
        tx,  float(*):  x-values for test points
        ty:  float(*):  y-values for test points
        ttree, cKDTree for tx, ty
    Outputs:
        close,     bool(*),  is this point within d of any boundary point?
        guess_ind, int(*),   index of closest boundary point to this point
        closest,   float(*), closest distance to a boundary point

    For the inputs, for (*x, *y, *tree), at least
        *x and *y --or-- *tree must be given
    if *tree is given, it will be used
    """
    sh = tx.shape
    tx = tx.ravel()
    ty = ty.ravel()
    sz = tx.size

    # construct tree for boundary / test
    if btree is None:
        btree = sp.spatial.cKDTree(np.column_stack([bx, by]))
    if ttree is None:
        ttree = sp.spatial.cKDTree(np.column_stack([tx, ty]))
    # query close points
    groups = ttree.query_ball_tree(btree, d)
    groups = [np.array(group) for group in groups]
    # wrangle output
    close = np.zeros(sz, dtype=bool)
    guess_ind = np.zeros(sz, dtype=int) - 1
    dists = np.zeros(sz, dtype=float) + 1e15
    if False: # this is slower right now...
        Groups = create_selector(groups)
        _wrangle_groups(bx, by, tx, ty, Groups, close, guess_ind, dists)
    else:
        for gi, group in enumerate(groups):
            close[gi] = len(group) > 0
            if close[gi]:
                dx = tx[gi] - bx[group]
                dy = ty[gi] - by[group]
                d2 = dx**2 + dy**2
                min_ind = np.argmin(d2)
                guess_ind[gi] = group[min_ind]
                dists[gi] = np.sqrt(d2[min_ind])
    close = close.reshape(sh)
    guess_ind = guess_ind.reshape(sh)
    dists = dists.reshape(sh)
    return close, guess_ind, dists



















### Sparse versions of these same routines...

def gridpoints_near_points_sparse(bx, by, xv, yv, d):
    """
    Fast near-points finder for a grid and set of points. 

    Where:
        n_close is an int, giving the number of gridpoints within
            a distance d of any of the boundary points
        x_ind, int(n_close), giving x indeces of close points
        y_ind, int(n_close), giving y indeces of close points
        that is, the jth close points location is given by:
            (x_ind[j], y_ind[j])

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
    Outputs: (tuple of...)
        n_close:   int:            number of close points
        ind_x,     int(n_close):   index into xv for close point
        ind_y,     int(n_close):   index into yv for close point
        guess_ind, int(n_close):   index of closest boundary point to
            the corresponding close point 
        closest,   float(n_close): distance between close point and
            corresponding closest boundary point
    """
    return _grid_near_points_sparse(bx, by, xv, yv, d)

@numba.njit(parallel=True)
def _grid_near_points_sparse(x, y, xv, yv, d):
    N = x.size
    Nx = xv.size
    Ny = yv.size
    xh = xv[1] - xv[0]
    yh = yv[1] - yv[0]
    xsd = d//xh + 1
    ysd = d//yh + 1
    d2 = d*d
    xlb = xv[0]
    ylb = yv[0]

    n_close = 0
    arr_size = 10
    ind_x   = np.empty(arr_size, dtype=np.int64)
    ind_y   = np.empty(arr_size, dtype=np.int64)
    gi      = np.empty(arr_size, dtype=np.int64)
    closest = np.empty(arr_size, dtype=np.float64)

    # sparse storage for duplicate checking
    x_size    = [0,]*Nx
    x_located = [np.empty(0, dtype=np.int64),]*Nx
    x_ind     = [np.empty(0, dtype=np.int64),]*Nx

    for i in range(N):
        x_loc = (x[i] - xlb) // xh
        y_loc = (y[i] - ylb) // yh
        x_lower = max(x_loc - xsd, 0)
        x_upper = min(x_loc + xsd + 1, Nx)
        y_lower = max(y_loc - ysd, 0)
        y_upper = min(y_loc + ysd + 1, Ny)
        for j in range(x_lower, x_upper):
            for k in range(y_lower, y_upper):
                # get the distances
                xd = xv[j] - x[i]
                yd = yv[k] - y[i]
                dist2 = xd**2 + yd**2
                # if we're close...
                if dist2 < d2:
                    # check if we already know of a close point
                    tester = x_located[j][:x_size[j]]
                    init, ind = inarray(k, tester)
                    # if we already know of a close point...
                    if init:
                        full_ind = x_ind[j][ind]
                        if dist2 < closest[full_ind]:
                            gi     [full_ind] = i
                            closest[full_ind] = dist2
                    else:
                        # if our main arrays are too small, expand!
                        if n_close >= arr_size:
                            arr_size *= 2
                            ind_x   = extend_array(ind_x,   arr_size)
                            ind_y   = extend_array(ind_y,   arr_size)
                            gi      = extend_array(gi,      arr_size)
                            closest = extend_array(closest, arr_size)
                        # if our sparse indexing arrays are too small, expand!
                        if x_size[j] >= x_located[j].size:
                            x_located[j] = extend_array(x_located[j], max(1, 2*x_size[j]))
                            x_ind[j]     = extend_array(x_ind[j],     max(1, 2*x_size[j]))
                        # update sparse indexing information
                        x_located[j][x_size[j]] = k
                        x_ind[j][x_size[j]] = n_close
                        x_size[j] += 1
                        # update main indexing information
                        ind_x  [n_close] = j
                        ind_y  [n_close] = k
                        gi     [n_close] = i
                        closest[n_close] = dist2
                        n_close += 1
    # reduce main arrays to correct size
    ind_x   = ind_x  [:n_close]
    ind_y   = ind_y  [:n_close]
    gi      = gi     [:n_close]
    closest = closest[:n_close]
    return n_close, ind_x, ind_y, gi, closest

def gridpoints_near_curve_sparse(cx, cy, xv, yv, d, tol=1e-14, verbose=False):
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
    n_close, x_ind, y_ind, guess_ind, close = gridpoints_near_points_sparse(cx, cy, xv, yv, D)
    # for all candidate points, perform brute force checking
    x_test = xv[x_ind]
    y_test = yv[y_ind]
    t_test, r_test = compute_local_coordinates(cx, cy, x_test, y_test,
                                newton_tol=tol, guess_ind=guess_ind, verbose=verbose)
    init = np.abs(r_test) <= d
    x_ind = x_ind[init]
    y_ind = y_ind[init]
    r     = r_test[init]
    t     = t_test[init]
    return x_ind, y_ind, r, t, (d, cx, cy)


