import numpy as np
import numba
import numexpr as ne
import scipy as sp
import scipy.spatial
from near_finder.utilities import extend_array, inarray

################################################################################
# Dense Routines

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
    are found.  Note that points that are not within D of the polygon will also
    be marked as "near", however

    Inputs:
        bx, float(nb): x-coordinates of boundary
        by, float(nb): y-coordinates of boundary
        xv, float(nx): x-values for grid coordinates
        yv: float(ny): y-values for grid coordinates
        d:  distance to find near points
    Outputs:
        close,     bool(nx, ny),  is this point within d of any boundary point?
        close_ind, int(nx, ny),   index of closest boundary point to this point
        distance,  float(nx, ny), closest distance to a boundary point
    """
    sh = (xv.shape[0], yv.shape[0])

    close = np.zeros(sh, dtype=bool)
    close_ind = np.full(sh, -1, dtype=int)
    distance = np.full(sh, 1e15, dtype=float)

    _grid_near_points(bx, by, xv, yv, d, close, close_ind, distance)
    return ne.evaluate('close > 0'), close_ind, distance

def gridpoints_near_points_update(bx, by, xv, yv, d, idn, close, int_helper1,
                                int_helper2, float_helper, bool_helper):
    """
    Fast near-points finder for a grid and set of points. 

    Returns a boolean array with size [xv.size, yv.size]
    The elements of the boolean array give whether that gridpoint is within
    d of any of the points bx/by

    When bx/by describe a polygon, one may use this function to find all points
    within a distance D of the polygon, by setting:
    d = sqrt(D^2 + (l/2)^2), where l is the length of the longest polygonal
    segment.  If l < D, then d need only be 1.12D to guarantee all near-points
    are found.  Note that points that are not within D of the polygon will also
    be marked as "near", however

    Inputs:
        bx,  float(nb): x-coordinates of boundary
        by,  float(nb): y-coordinates of boundary
        xv,  float(nx): x-values for grid coordinates
        yv:  float(ny): y-values for grid coordinates
        d:   float:     distance to find near points
        idn, int:       unique identificiation for this update (ONLY use 1, 2, 3, ....)
    InOuts:
        close,        bool(nx, ny):  is this point within d of any boundary point?
        int_helper1,  int(nx, ny):   index of closest boundary point to this point
        int_helper2,  int(nx, ny):   helper grid for keying points to boundaries
        float_helper, float(nx, ny): closest distance to a boundary point
        bool_helper,  bool(nx, ny):  helper grid for identifying if a change was
                                     made in this update
            SPECIFICALLY: After this call, bool_helper[indx, indy] will contain
                what close[indx, indy] was BEFORE the call
                this is useful in the points_near_curve routines
        INITIAL SET VALUES FOR HELPERS:
            int_helper1  --> 0
            int_helper2  --> 0
            float_helper --> np.inf
            bool_helper  --> False
    Outputs:
        nclose, int:           number of close points added in this update
        indx,   int(nclose):   sparse indeces into where close was added in this update
        indy,   int(nclose):   sparse indeces into where close was added in this update
        sci,    float(nclose): close_ind corresponding to indx, indy
    """
    nclose, indx, indy, sci = _grid_near_points_udpate(bx, by, xv, yv, d, close, int_helper1, float_helper, int_helper2, bool_helper, idn)
    return nclose, indx, indy, sci

@numba.njit
def _grid_near_points(x, y, xv, yv, d, close, gi, closest):
    N = x.shape[0]
    Nx = xv.shape[0]
    Ny = yv.shape[0]
    xh = xv[1] - xv[0]
    yh = yv[1] - yv[0]
    xsd = int(d//xh + 1)
    ysd = int(d//yh + 1)
    d2 = d*d
    xlb = xv[0]
    ylb = yv[0]
    for i in range(N):
        x_loc = int((x[i] - xlb) // xh)
        y_loc = int((y[i] - ylb) // yh)
        x_lower = max(x_loc - xsd, 0)
        x_upper = min(x_loc + xsd + 1, Nx)
        y_lower = max(y_loc - ysd, 0)
        y_upper = min(y_loc + ysd + 1, Ny)
        for j in range(x_lower, x_upper):
            for k in range(y_lower, y_upper):
                xd = xv[j] - x[i]
                yd = yv[k] - y[i]
                dist2 = xd**2 + yd**2
                close[j, k] = close[j, k] or dist2 < d2
                if dist2 < closest[j, k]:
                    closest[j, k] = dist2
                    gi[j, k] = i

@numba.njit
def _grid_near_points_udpate(x, y, xv, yv, d, close, gi, closest, helper1, helper2, idn):
    N = x.shape[0]
    Nx = xv.shape[0]
    Ny = yv.shape[0]
    xh = xv[1] - xv[0]
    yh = yv[1] - yv[0]
    xsd = int(d//xh + 1)
    ysd = int(d//yh + 1)
    d2 = d*d
    xlb = xv[0]
    ylb = yv[0]
    # udpate dense grid
    counter = 0
    for i in range(N):
        x_loc = int((x[i] - xlb) // xh)
        y_loc = int((y[i] - ylb) // yh)
        x_lower = max(x_loc - xsd, 0)
        x_upper = min(x_loc + xsd + 1, Nx)
        y_lower = max(y_loc - ysd, 0)
        y_upper = min(y_loc + ysd + 1, Ny)
        for j in range(x_lower, x_upper):
            for k in range(y_lower, y_upper):
                xd = xv[j] - x[i]
                yd = yv[k] - y[i]
                dist2 = xd**2 + yd**2
                close_here = dist2 < d2
                comes_from_other = helper1[j, k] != idn
                if close_here:
                    if comes_from_other:
                        counter += 1
                        helper2[j, k] = close[j, k]
                        closest[j, k] = dist2
                        gi[j, k] = i
                    elif dist2 < closest[j, k]:
                        closest[j, k] = dist2
                        gi[j, k] = i
                    close[j, k] = True
                    helper1[j, k] = idn
    # construct sparse output
    idx  = np.empty(counter, dtype=np.int64)
    idy  = np.empty(counter, dtype=np.int64)
    sgi  = np.empty(counter, dtype=np.int64)
    ind = 0
    for i in range(N):
        x_loc = (x[i] - xlb) // xh
        y_loc = (y[i] - ylb) // yh
        x_lower = max(x_loc - xsd, 0)
        x_upper = min(x_loc + xsd + 1, Nx)
        y_lower = max(y_loc - ysd, 0)
        y_upper = min(y_loc + ysd + 1, Ny)
        for j in range(x_lower, x_upper):
            for k in range(y_lower, y_upper):
                if helper1[j, k] == idn:
                    idx[ind] = j
                    idy[ind] = k
                    sgi[ind] = gi[j, k]
                    helper1[j, k] = -idn
                    ind += 1
    return counter, idx, idy, sgi

def points_near_points(d, bx, by, tx, ty, btree=None, ttree=None):
    """
    Fast tree based near-points finder for a set of test points and
        set of boundary points

    If your points have a regular grid structure, then gridpoints_near_points
    will outperform this function by an enormous amount!

    When bx/by describe a polygon, one may use this function to find all points
    within a distance D of the polygon, by setting:
    d = sqrt(D^2 + (l/2)^2), where l is the length of the longest polygonal
    segment.  If l < D, then d need only be 1.12D to guarantee all near-points
    are found.  Note that points that are not within D of the polygon will also
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
        close_ind, int(*),   index of closest boundary point to this point
        distance,  float(*), closest distance to a boundary point

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
    close_ind = np.zeros(sz, dtype=int) - 1
    dists = np.zeros(sz, dtype=float) + 1e15
    for gi, group in enumerate(groups):
        close[gi] = len(group) > 0
        if close[gi]:
            dx = tx[gi] - bx[group]
            dy = ty[gi] - by[group]
            d2 = dx**2 + dy**2
            min_ind = np.argmin(d2)
            close_ind[gi] = group[min_ind]
            dists[gi] = np.sqrt(d2[min_ind])
    close = close.reshape(sh)
    close_ind = close_ind.reshape(sh)
    dists = dists.reshape(sh)
    return close, close_ind, dists

################################################################################
# Sparse Routines

def gridpoints_near_points_sparse(bx, by, xv, yv, d):
    """
    Fast sparse near-points finder for a grid and set of points. 

    When bx/by describe a polygon, one may use this function to find all points
    within a distance D of the polygon, by setting:
    d = sqrt(D^2 + (l/2)^2), where l is the length of the longest polygonal
    segment.  If l < D, then d need only be 1.12D to guarantee all near-points
    are found.  Note that points that are not within D of the polygon will also
    be marked as "near", however

    NOTE: This function suffers from having to allocate and resize a lot of
    arrays on the fly.  Depending on your needs, gridpoints_near_points_update
    may prove to be a substantially faster solution

    Inputs:
        bx, float(nb): x-coordinates of boundary
        by, float(nb): y-coordinates of boundary
        xv, float(nx): x-values for grid coordinates
        yv: float(ny): y-values for grid coordinates
        d:  distance to find near points
    Outputs: (tuple of...)
        nclose:   int:            number of close points
        ind_x,     int(nclose):   index into xv for close point
        ind_y,     int(nclose):   index into yv for close point
        close_ind, int(nclose):   index of closest boundary point to
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

def points_near_points_sparse(d, bx, by, tx, ty, btree=None, ttree=None):
    """
    Fast tree based near-points finder for a set of test points and
        set of boundary points

    When bx/by describe a polygon, one may use this function to find all points
    within a distance D of the polygon, by setting:
    d = sqrt(D^2 + (l/2)^2), where l is the length of the longest polygonal
    segment.  If l < D, then d need only be 1.12D to guarantee all near-points
    are found.  Note that points that are not within D of the polygon will also
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
        n_close:   int:            number of close points
        ind,       int(n_close):   index into flattened array
        close_ind, int(n_close):   index of closest boundary point to
            the corresponding close point 
        distance,   float(n_close): distance between close point and
            corresponding closest boundary point

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
    arr_size = 10
    ind = np.zeros(arr_size, dtype=int)
    close_ind = np.zeros(arr_size, dtype=int)
    dists = np.zeros(arr_size, dtype=float)
    n_close = 0
    for gi, group in enumerate(groups):
        if len(group) > 0:
            # first check if the arrays need to be expanded
            if n_close >= arr_size:
                arr_size *= 2
                ind       = extend_array(ind,       arr_size)
                close_ind = extend_array(close_ind, arr_size)
                dists     = extend_array(dists,     arr_size)
            # now add in the close point and guess indeces
            ind[n_close] = gi
            dx = tx[gi] - bx[group]
            dy = ty[gi] - by[group]
            d2 = dx**2 + dy**2
            min_ind = np.argmin(d2)
            close_ind[n_close] = group[min_ind]
            dists[n_close] = np.sqrt(d2[min_ind])
            n_close += 1
    return n_close, ind[:n_close], close_ind[:n_close],  dists[:n_close]
