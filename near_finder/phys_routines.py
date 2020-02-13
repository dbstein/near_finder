import numpy as np
import numba
import shapely.geometry
from near_finder.utilities import upsample, compute_normals

################################################################################
# Dense routine for points inside curve

def points_inside_curve(x, y, res, inside=True):
    """
    Computes, for all points, whether the points are inside
        or outside of the closed curve
    Inputs:
        x,      float(nx, ny): x-values for grid coordinates
        y,      float(nx, ny): y-values for grid coordinates
        res,    tuple:         result of call to gridpoints_near_curve / points_near_curve
        inside, bool: whether to compute points interior or exterior to curve
    """
    in_annulus = res[0]
    r = res[1]
    d = res[3][0]
    cx = res[3][1]
    cy = res[3][2]
    # get normals
    nx, ny = compute_normals(cx, cy)
    # compute bounding boundaries
    bx = cx - 0.9*d*nx
    by = cy - 0.9*d*ny
    ux = cx + 0.9*d*nx
    uy = cy + 0.9*d*ny
    # put bounding boundaries into shapely format
    bbdy = list(zip(bx, by))
    ubdy = list(zip(ux, uy))
    bpath = shapely.geometry.Polygon(bbdy)
    upath = shapely.geometry.Polygon(ubdy)
    # find an undersampled boundary that lies between bounding boundaries
    n = 5
    good = False
    while not good:
        new_t  = np.linspace(0, 2*np.pi, n, endpoint=False)
        rcx = upsample(cx, n)
        rcy = upsample(cy, n)
        new_path = shapely.geometry.Polygon(list(zip(rcx, rcy)))
        test1 = upath.contains(new_path)
        test2 = new_path.contains(bpath)
        good = test1 and test2
        if not good:
            n += 5
    # find points inside/outside of undersampled boundary
    result = numba_find_phys(x, y, rcx, rcy, inside)
    # now for points inside the annulus, adjust based on local coordinates
    if inside:
        result[in_annulus] = r[in_annulus] <= 0.0
    else:
        result[in_annulus] = r[in_annulus] >= 0.0
    return result

def numba_find_phys(x, y, bdyx, bdyy, inside):
    """
    Computes whether the points x, y are inside or outside of the polygon defined by the
    x-coordinates bdyx and the y-coordinates bdyy
    The polgon is assumed not to be closed (the last point is not replicated)
    """
    if inside:
        result = np.zeros(x.shape, dtype=bool)
        vecPointInPath(x.ravel(), y.ravel(), bdyx, bdyy, result.ravel())
    else:
        result = np.ones(x.shape, dtype=bool)
        vecPointNotInPath(x.ravel(), y.ravel(), bdyx, bdyy, result.ravel())
    return result.reshape(x.shape)
@numba.njit(parallel=True)
def vecPointNotInPath(x, y, polyx, polyy, result):
    m = x.shape[0]
    for i in numba.prange(m):
        result[i] = not isPointInPath(x[i], y[i], polyx, polyy)
@numba.njit(parallel=True)
def vecPointInPath(x, y, polyx, polyy, result):
    m = x.shape[0]
    for i in numba.prange(m):
        result[i] = isPointInPath(x[i], y[i], polyx, polyy)
@numba.njit
def isPointInPath(x, y, polyx, polyy):
    num = polyx.shape[0]
    i = 0
    j = num - 1
    c = False
    for i in range(num):
        pyi = polyy[i]
        pyj = polyy[j]
        pxi = polyx[i]
        pxj = polyx[j]
        if ((pyi > y) != (pyj > y)) and \
                (x < pxi + (pxj - pxi)*(y - pyi)/(pyj - pyi)):
            c = not c
        j = i
    return c

################################################################################
# Sparse routines

def numba_find_phys_grid(xv, yv, bdyx, bdyy, inside):
    """
    Computes whether the points spanned by xv, yv are inside of the polygon defined by the
    x-coordinates bdyx and the y-coordinates bdyy
    The polgon is assumed not to be closed (the last point is not replicated)
    """
    if inside:
        result = np.zeros([xv.size, yv.size], dtype=bool)
        vecGridPointInPath(xv, yv, bdyx, bdyy, result)
    else:
        result = np.ones([xv.size, yv.size], dtype=bool)
        vecGridPointNotInPath(xv, yv, bdyx, bdyy, result)
    return result
@numba.njit(parallel=True)
def vecGridPointInPath(xv, yv, polyx, polyy, result):
    mx = xv.size
    my = yv.size
    for i in numba.prange(mx):
        for j in range(my):
            result[i, j] = isPointInPath(xv[i], yv[j], polyx, polyy)
@numba.njit(parallel=True)
def vecGridPointNotInPath(xv, yv, polyx, polyy, result):
    mx = xv.size
    my = yv.size
    for i in numba.prange(mx):
        for j in range(my):
            result[i, j] = not isPointInPath(xv[i], yv[j], polyx, polyy)

def points_inside_curve_sparse(xv, yv, res, inside=True):
    """
    Returns a physical array inside a bounding box, and indeces into the full array

    res, tuple:         result of call to gridpoints_near_curve_sparse / points_near_curve_sparse
    """
    n_close = res[0]
    x_ind = res[1]
    y_ind = res[2]
    r = res[3]
    t = res[4]
    d = res[5][0]
    cx = res[5][1]
    cy = res[5][2]
    # get reduced xv/yv vectors constrained by bounding box
    xh = xv[1] - xv[0]
    cx_min = np.min(cx) - d
    cx_max = np.max(cx) + d
    xv_min_ind = max(0,       int((cx_min-d-xv[0])//xh))
    xv_max_ind = min(xv.size, int((cx_max+d-xv[0])//xh)+2)
    yh = yv[1] - yv[0]
    cy_min = np.min(cy) - d
    cy_max = np.max(cy) + d
    yv_min_ind = max(0,       int((cy_min-d-yv[0])//yh))
    yv_max_ind = min(yv.size, int((cy_max+d-yv[0])//yh)+2)
    xv_small = xv[xv_min_ind:xv_max_ind]
    yv_small = yv[yv_min_ind:yv_max_ind]

    # get normals
    nx, ny = compute_normals(cx, cy)
    # compute bounding boundaries
    bx = cx - 0.9*d*nx
    by = cy - 0.9*d*ny
    ux = cx + 0.9*d*nx
    uy = cy + 0.9*d*ny
    # put bounding boundaries into shapely format
    bbdy = list(zip(bx, by))
    ubdy = list(zip(ux, uy))
    bpath = shapely.geometry.Polygon(bbdy)
    upath = shapely.geometry.Polygon(ubdy)
    # find an undersampled boundary that lies between bounding boundaries
    n = 5
    good = False
    while not good:
        new_t  = np.linspace(0, 2*np.pi, n, endpoint=False)
        rcx = upsample(cx, n)
        rcy = upsample(cy, n)
        new_path = shapely.geometry.Polygon(list(zip(rcx, rcy)))
        test1 = upath.contains(new_path)
        test2 = new_path.contains(bpath)
        good = test1 and test2
        if not good:
            n += 5
    # find points inside/outside of undersampled boundary
    result = numba_find_phys_grid(xv_small, yv_small, rcx, rcy, inside)
    # now for points inside the annulus, adjust based on local coordinates
    if inside:
        result[x_ind-xv_min_ind, y_ind-yv_min_ind] = r <= 0.0
    else:
        result[x_ind-xv_min_ind, y_ind-yv_min_ind] = r >= 0.0
    return result, xv_min_ind, xv_max_ind, yv_min_ind, yv_max_ind

def numba_find_phys_grid_update(xv, yv, bdyx, bdyy, result, xm, ym, inside):
    """
    Computes whether the points spanned by xv, yv are inside of the polygon defined by the
    x-coordinates bdyx and the y-coordinates bdyy
    The polgon is assumed not to be closed (the last point is not replicated)
    """
    if inside:
        vecGridPointInPathUpdate(xv, yv, bdyx, bdyy, result, xm, ym)
    else:
        vecGridPointNotInPathUpdate(xv, yv, bdyx, bdyy, result, xm, ym)
@numba.njit(parallel=True)
def vecGridPointInPathUpdate(xv, yv, polyx, polyy, result, xm, ym):
    mx = xv.size
    my = yv.size
    for i in numba.prange(mx):
        xmi = xm + i
        for j in range(my):
            if isPointInPath(xv[i], yv[j], polyx, polyy):
                result[xmi, ym+j] = True
@numba.njit(parallel=True)
def vecGridPointNotInPathUpdate(xv, yv, polyx, polyy, result, xm, ym):
    mx = xv.size
    my = yv.size
    for i in numba.prange(mx):
        xmi = xm + i
        for j in range(my):
            if isPointInPath(xv[i], yv[j], polyx, polyy):
                result[xmi, ym+j] = False

def points_inside_curve_update(xv, yv, res, result, inside=True):
    """
    Updates a phys array:
        For inside=True, this means setting Falses to True for all interior values
        For inside=False, this means setting Trues to Falses for all interior values
    """
    x_ind = res[1]
    y_ind = res[2]
    r = res[3]
    t = res[4]
    d = res[5][0]
    cx = res[5][1]
    cy = res[5][2]
    # save what is there right now
    res_save = result[x_ind, y_ind].copy()
    # get reduced xv/yv vectors constrained by bounding box
    xh = xv[1] - xv[0]
    cx_min = np.min(cx) - d
    cx_max = np.max(cx) + d
    xv_min_ind = max(0,       int((cx_min-d-xv[0])//xh))
    xv_max_ind = min(xv.size, int((cx_max+d-xv[0])//xh)+2)
    yh = yv[1] - yv[0]
    cy_min = np.min(cy) - d
    cy_max = np.max(cy) + d
    yv_min_ind = max(0,       int((cy_min-d-yv[0])//yh))
    yv_max_ind = min(yv.size, int((cy_max+d-yv[0])//yh)+2)
    xv_small = xv[xv_min_ind:xv_max_ind]
    yv_small = yv[yv_min_ind:yv_max_ind]

    # get normals
    nx, ny = compute_normals(cx, cy)
    # compute bounding boundaries
    bx = cx - 0.9*d*nx
    by = cy - 0.9*d*ny
    ux = cx + 0.9*d*nx
    uy = cy + 0.9*d*ny
    # put bounding boundaries into shapely format
    bbdy = list(zip(bx, by))
    ubdy = list(zip(ux, uy))
    bpath = shapely.geometry.Polygon(bbdy)
    upath = shapely.geometry.Polygon(ubdy)
    # find an undersampled boundary that lies between bounding boundaries
    n = 5
    good = False
    while not good:
        new_t  = np.linspace(0, 2*np.pi, n, endpoint=False)
        rcx = upsample(cx, n)
        rcy = upsample(cy, n)
        new_path = shapely.geometry.Polygon(list(zip(rcx, rcy)))
        test1 = upath.contains(new_path)
        test2 = new_path.contains(bpath)
        good = test1 and test2
        if not good:
            n += 5
    # find points inside/outside of undersampled boundary
    numba_find_phys_grid_update(xv_small, yv_small, rcx, rcy, result, xv_min_ind, yv_min_ind, inside)
    # now for points inside the annulus, adjust based on local coordinates
    if inside:
        result[x_ind, y_ind] = np.logical_or(res_save, r <= 0.0)
    else:
        result[x_ind, y_ind] = np.logical_and(res_save, r >= 0.0)
