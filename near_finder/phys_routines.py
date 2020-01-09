import numpy as np
import numba
import shapely.geometry
from .utilities import upsample, compute_normals

def points_inside_curve(x, y, res):
    """
    Computes, for all points, whether the points are inside
        or outside of the closed curve
    Inputs:
        x,   float(nx, ny): x-values for grid coordinates
        y,   float(nx, ny): y-values for grid coordinates
        res, tuple:         result of call to gridpoints_near_curve / points_near_curve
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
    inside = numba_find_phys(x, y, rcx, rcy)
    # now for points inside the annulus, adjust based on local coordinates
    inside[in_annulus] = r[in_annulus] <= 0.0
    return inside

def numba_find_phys(x, y, bdyx, bdyy):
    """
    Computes whether the points x, y are inside of the polygon defined by the
    x-coordinates bdyx and the y-coordinates bdyy
    The polgon is assumed not to be closed (the last point is not replicated)
    """
    inside = np.zeros(x.shape, dtype=bool)
    vecPointInPath(x.ravel(), y.ravel(), bdyx, bdyy, inside.ravel())
    return inside.reshape(x.shape)
@numba.njit(parallel=True)
def vecPointInPath(x, y, polyx, polyy, inside):
    m = x.shape[0]
    for i in numba.prange(m):
        inside[i] = isPointInPath(x[i], y[i], polyx, polyy)
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
