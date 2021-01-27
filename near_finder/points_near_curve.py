import numpy as np
from near_finder.utilities import compute_curvature, compute_speed, upsample
from near_finder.coordinate_routines import compute_local_coordinates
from near_finder.points_near_points import points_near_points, gridpoints_near_points
from near_finder.points_near_points import points_near_points_sparse, gridpoints_near_points_sparse
from near_finder.points_near_points import gridpoints_near_points_update

################################################################################
# Dense Routines

def _upsample_curve(cx, cy, d):
    """
    Upsamples the curve (cx, cy) so that it can be used in nearest curve finding
    In particular, we insist that the largest point-point distance is <= d
    Inputs:
        cx, float(n): x-coordinates of curve
        cy, float(n): y-coordinates of curve
        d,  float:    search distance
    Returns:
        cx, float(N): x-coordinates of upsampled curve
        cy, float(N): y-coordinates of upsampled curve
        D,  float:    search distance to be used in near-points finder
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
    # extra large fudge factor for near search because its a curve
    D = 1.5*d
    return cx.copy(), cy.copy(), D

def _check_near_points(cx, cy, x, y, d, iao, ro, to, near, interpolation_scheme, tol, gi, verbose):
    """
    Check all points in (x[near], y[near] to see if they are close to the curve
    This is done using a Newton solver, and checking the local coordinates
    Inputs:
        cx,   float(n): x-coordinates of upsampled curve
        cy,   float(n): y-coordinates of upsampled curve
        x,    float(*): x-coordinates of points being searched
        y,    float(*): y-coordinates of points being searched
        d,    float:    search distance
        near, bool(*):  which points in (x, y) should be searched
        interpolation_scheme, tol, gi, verbose:
            see documentation to "compute_local_coordinates" 
            these parameters correspond as:
                interpolation_scheme --> interpolation_scheme
                gi                   --> guess_ind
                tol                  --> newton_tol
                verbose              --> verbose
    InOuts:
        iao, bool(*):  whether points are in the annulus or not
        ro,  float(*): r-coordinate for points that are in the annulus
        to,  float(*): t-coordinate for points that are in the annulus
    """
    t, r = compute_local_coordinates(cx, cy, x, y, interpolation_scheme=interpolation_scheme,
                                newton_tol=tol, guess_ind=gi, verbose=verbose)
    ia = np.abs(r) <= d
    nia = np.logical_not(ia)
    r[nia] = np.nan
    t[nia] = np.nan
    iao[near] = ia
    ro[near] = r
    to[near] = t
def _check_near_points_all(cx, cy, x, y, d, interpolation_scheme, tol, gi, verbose, max_iterations):
    """
    Check all points in (x, y to see if they are close to the curve
    This is done using a Newton solver, and checking the local coordinates
    Inputs:
        cx,   float(n): x-coordinates of upsampled curve
        cy,   float(n): y-coordinates of upsampled curve
        x,    float(*): x-coordinates of points being searched
        y,    float(*): y-coordinates of points being searched
        d,    float:    search distance
        interpolation_scheme, tol, gi, verbose:
            see documentation to "compute_local_coordinates" 
            these parameters correspond as:
                interpolation_scheme --> interpolation_scheme
                gi                   --> guess_ind
                tol                  --> newton_tol
                verbose              --> verbose
    Outputs:
        ia, bool(*):  whether points are in the annulus or not
        r,  float(*): r-coordinate for points that are in the annulus
        t,  float(*): t-coordinate for points that are in the annulus
    """
    t, r = compute_local_coordinates(cx, cy, x, y, interpolation_scheme=interpolation_scheme,
                                newton_tol=tol, guess_ind=gi, verbose=verbose, max_iterations=max_iterations)
    ia = np.abs(r) <= d
    nia = np.logical_not(ia)
    r[nia] = np.nan
    t[nia] = np.nan
    return ia, r, t

def gridpoints_near_curve(cx, cy, xv, yv, d, interpolation_scheme='nufft', tol=1e-12, verbose=False):
    """
    Computes, for all gridpoints spanned by (xv, yv), whether the gridpoints
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
        cx,                   float(nb): x-coordinates of boundary
        cy,                   float(nb): y-coordinates of boundary
        xv,                   float(nx): x-values for grid coordinates
        yv,                   float(ny): y-values for grid coordinates
        d,                    float:     distance to search within 
        interpolation_scheme, tol, verbose:
            see documentation to "compute_local_coordinates" 
            these parameters correspond as:
                interpolation_scheme --> interpolation_scheme
                tol                  --> newton_tol
                verbose              --> verbose
    Outputs: (tuple of)
        in_annulus, bool (nx, ny), whether points are in annulus of radius d
        r,          float(nx, ny), r-coordinate for points in_annulus
        t,          float(nx, ny), t-coordinate for points in_annulus
        (d, cx, cy) float,         search distance, upsampled cx, cy
    """
    # upsample if needed and get search distance
    cx, cy, D = _upsample_curve(cx, cy, d)
    # get points near points
    near, guess_ind, _ = gridpoints_near_points(cx, cy, xv, yv, D)
    # allocate output arrays
    sh = (xv.size, yv.size)
    in_annulus = np.zeros(sh, dtype=bool)
    r = np.full(sh, np.nan, dtype=float)
    t = np.full(sh, np.nan, dtype=float)
    # if there are any near points, compute local coordinates
    if near.any():
        # extract test values
        wh = np.where(near)
        x_test = xv[wh[0]]
        y_test = yv[wh[1]]
        gi = guess_ind[near]
        # get local coordinates, find points in_annulus, get coords
        _check_near_points( cx, cy, x_test, y_test, d, in_annulus, r, t, near,
                                        interpolation_scheme, tol, gi, verbose )
    return in_annulus, r, t, (d, cx, cy)

def gridpoints_near_curve_update(cx, cy, xv, yv, d, idn, close, int_helper1, int_helper2, float_helper, bool_helper, interpolation_scheme='nufft', tol=1e-12, verbose=False, max_iterations=30):
    """
    Computes, for all gridpoints spanned by (xv, yv), whether the gridpoints
    1) are within d of the (closed) curve
    2) for those that are within d of the curve,
        how far the gridpoints are from the curve
        (that is, closest approach in the normal direction)
    3) local coordinates, in (r, t) coords, for the curve
        that is, we assume the curve is given by X(t) = (cx(t_i), cy(t_i))
        we define local coordinates X(t, r) = X(t) + n(t) r, where n is
        the normal to the curve at t,
        and return (r, t) for each gridpoint in some search region

    THIS DIFFERS SIGNIFICANTLY FROM gridpoints_near_curve. In particular,
        this is more appropriate for cases where you are dealing with multiple
        body problems.  It requires the pre-allocation of the output arrays,
        as well as some extra helper arrays. It updates, rather than providing
        output, and only touches those points of the ouptut arrays that are
        actually close to this particular curve. Output is provided in a sparse
        format, rather than dense format (although the output arrays that are
        preallocated are still dense)

    Inputs:
        cx,  float(nb): x-coordinates of boundary
        cy,  float(nb): y-coordinates of boundary
        xv,  float(nx): x-values for grid coordinates
        yv,  float(ny): y-values for grid coordinates
        d,   float:     distance to search within 
        idn, int:       unqiue ID for this boundary (use ONLY positive numbers 1, 2, ...)
        interpolation_scheme, tol, verbose:
            see documentation to "compute_local_coordinates" 
            these parameters correspond as:
                interpolation_scheme --> interpolation_scheme
                tol                  --> newton_tol
                verbose              --> verbose
    InOuts:
        close,        bool(nx, ny): whether points are in annulus of radius d
        int_helper1,  int(nx, ny)
        int_helper2,  int(nx, ny)
        bool_helper,  bool(nx, ny)
        float_helper, float(nx, ny)
        guess_ind, 

    Outputs: (tuple of)
        nclose, int: number of points close to this curve
        indx,   int(nclose):   x-index into grid of close point
        indy,   int(nclose):   y-index into grid of close point
        r,      float(nclose): r-coordinate for corresponding close point
        t,      float(nclose): t-coordinate for corresponding close point
        (d, cx, cy) float,         search distance, upsampled cx, cy
    """
    # upsample if needed and get search distance
    cx, cy, D = _upsample_curve(cx, cy, d)
    # get points near points
    nclose, indx, indy, sgi = gridpoints_near_points_update(cx, cy, xv, yv, D, idn, close, int_helper1, int_helper2, float_helper, bool_helper)
    # if there are any near points, compute local coordinates
    if nclose > 0:
        # extract test values
        x_test = xv[indx]
        y_test = yv[indy]
        # get local coordinates, find points in_annulus, get coords
        ia, r, t = _check_near_points_all( cx, cy, x_test, y_test, d, interpolation_scheme, tol, sgi, verbose, max_iterations=max_iterations )
    close[indx, indy] = np.logical_or(bool_helper[indx, indy], ia)
    indx = indx[ia]
    indy = indy[ia]
    r    = r   [ia]
    t    = t   [ia]
    nclose = indx.size

    return nclose, indx, indy, r, t, (d, cx, cy)

def points_near_curve_coarse(cx, cy, x, y, d):
    # upsample if needed and get search distance
    cx, cy, D = _upsample_curve(cx, cy, d)    
    near, guess_ind, close = points_near_points(d, bx=cx, by=cy, tx=x, ty=y)
    return near, guess_ind

def points_near_curve(cx, cy, x, y, d, near=None, guess_ind=None, interpolation_scheme='nufft', tol=1e-12, verbose=False):
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
        cx,                   float(nb): x-coordinates of boundary
        cy,                   float(nb): y-coordinates of boundary
        x,                    float(*):  x-values to search
        y,                    float(*):  y-values to search
        d,                    float:     distance to search within  
        near,                 bool(*):   result giving near-points to curve
                                (if this wasn't found appropriately, then
                                the results of this function may not be correct!)
        guess_ind,            int(*):    guess indeces for Newton solver
        interpolation_scheme, str:       type of interpolation used in Newton solver
        tol,                  float:     tolerance to be passed to Newton solver for coords
        verbose,              bool:  flag passed to coord solver for verbose output
    Outputs: (tuple of)
        in_annulus, bool (*), whether points are in annulus of radius d
        r,          float(*), r-coordinate for points in_annulus
        t,          float(*), t-coordinate for points in_annulus
        (d, cx, cy) float,     search distance, upsampled cx, cy
    """
    sh = x.shape
    x = x.ravel()
    y = y.ravel()
    sz = x.size

    # upsample if needed and get search distance
    cx, cy, D = _upsample_curve(cx, cy, d)    
    # get points near points
    if near is None or guess_ind is None:
        near, guess_ind, close = points_near_points(d, bx=cx, by=cy, tx=x, ty=y)
    # initialize output matrices
    in_annulus = np.zeros(sz, dtype=bool)
    r = np.full(sz, np.nan, dtype=float)
    t = np.full(sz, np.nan, dtype=float)
    # if there are any near points, compute local coordinates
    if near.any():
        # extract test values
        x_test = x[near]
        y_test = y[near]
        gi = guess_ind[near]
        # get local coordinates, find points in_annulus, get coords
        _check_near_points( cx, cy, x_test, y_test, d, in_annulus, r, t, near,
                                        interpolation_scheme, tol, gi, verbose )
    in_annulus = in_annulus.reshape(sh)
    r = r.reshape(sh)
    t = t.reshape(sh)
    return in_annulus, r, t, (d, cx, cy)

################################################################################
# Sparse Routines

def gridpoints_near_curve_sparse(cx, cy, xv, yv, d, near_xind=None, near_yind=None, guess_ind=None, interpolation_scheme='nufft', tol=1e-12, verbose=False):
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
        cx,                   float(nb): x-coordinates of boundary
        cy,                   float(nb): y-coordinates of boundary
        xv,                   float(nx): x-values for grid coordinates
        yv,                   float(ny): y-values for grid coordinates
        d,                    float:     distance to search within                  
        near_xind,            int(ncheck): points to compute coords of
        near_yind,            int(ncheck): points to compute coords of
                                (if this wasn't found appropriately, then
                                the results of this function may not be correct!)
        guess_ind,            int(ncheck): guess indeces for Newton solver
        interpolation_scheme, str:       type of interpolation used in Newton solver
        tol,                  float:     tolerance to be passed to Newton solver for coords
        verbose,              bool:  flag passed to coord solver for verbose output
    Outputs: (tuple of)
        x_ind, int(nclose):   indeces into xv for close points
        y_ind, int(nclose):   indeces into yv for close points
        r,     float(nclose): r-coordinate for close points
        t,     float(nclose): t-coordinate for close points
        (d, cx, cy) float,         search distance, upsampled cx, cy
    """
    cx, cy, D = _upsample_curve(cx, cy, d)
    if near_xind is None or near_yind is None or guess_ind is None:
        n_close, near_xind, near_yind, guess_ind, close = gridpoints_near_points_sparse(cx, cy, xv, yv, D)
    else:
        n_close = near_xind.size
    ia    = np.zeros(n_close, dtype=bool)
    near  = np.ones(n_close, dtype=bool)
    r     = np.full(n_close, np.nan, dtype=float)
    t     = np.full(n_close, np.nan, dtype=float)
    if n_close > 0:
        # for all candidate points, perform brute force checking
        x_test = xv[near_xind]
        y_test = yv[near_yind]
        # get local coordinates, find points in_annulus, get coords
        _check_near_points( cx, cy, x_test, y_test, d, ia, r, t, near,
                                interpolation_scheme, tol, guess_ind, verbose )
    return n_close, near_xind[ia], near_yind[ia], r[ia], t[ia], (d, cx, cy)

def points_near_curve_sparse(cx, cy, x, y, d, ind, guess_ind, interpolation_scheme='nufft', tol=1e-12, verbose=False):
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
        cx,                   float(nb): x-coordinates of boundary
        cy,                   float(nb): y-coordinates of boundary
        x,                    float(*):  x-values to search
        y,                    float(*):  y-values to search
        d,                    float:     distance to search within  
        near_ind,             int(*):    if given, indeces into flattened x array
                                         giving points to find coords for
                                (if this wasn't found appropriately, then
                                the results of this function may not be correct!)
        guess_ind,            int(*):    guess indeces for Newton solver
        interpolation_scheme, str:       type of interpolation used in Newton solver
        tol,                  float:     tolerance to be passed to Newton solver for coords
        verbose,              bool:  flag passed to coord solver for verbose output
    Outputs: (tuple of)
        x_ind, int(nclose):   indeces into xv for close points
        y_ind, int(nclose):   indeces into yv for close points
        r,     float(nclose): r-coordinate for close points
        t,     float(nclose): t-coordinate for close points
        (d, cx, cy) float,         search distance, upsampled cx, cy
    """
    sh = x.shape
    x = x.ravel()
    y = y.ravel()
    sz = x.size

    # upsample if needed and get search distance
    cx, cy, D = _upsample_curve(cx, cy, d)    
    # get points near points
    if near is None or guess_ind is None:
        n_close, near_ind, guess_ind, close = points_near_points_sparse(d, bx=cx, by=cy, tx=x, ty=y)
    else:
        n_close = near_ind.size
    # initialize output matrices
    ia    = np.zeros(n_close, dtype=bool)
    near  = np.ones(n_close, dtype=bool)
    r     = np.full(n_close, np.nan, dtype=float)
    t     = np.full(n_close, np.nan, dtype=float)
    # if there are any near points, compute local coordinates
    if n_close > 0:
        # extract test values
        x_test = x[near_ind]
        y_test = y[near_ind]
        gi = guess_ind[near_ind]
        # get local coordinates, find points in_annulus, get coords
        _check_near_points( cx, cy, x_test, y_test, d, in_annulus, r, t, near,
                                        interpolation_scheme, tol, gi, verbose )
    return n_close, near_ind[ia], r[ia], t[ia], (d, cx, cy)


