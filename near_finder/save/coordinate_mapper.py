import numpy as np
import shapely.geometry
import shapely.prepared
import numba
from near_finder.nufft_interp import periodic_interp1d
from pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary import Global_Smooth_Boundary as GSB

from numba.typed import List
def list_to_typed_list(L):
    TL = List()
    for x in L: TL.append(x)
    return TL

################################################################################
# Coordinate routines specific for this case

@numba.njit(fastmath=True)
def _guess_ind_finder_centering(cx, cy, x, y, gi, d):
    _min = 1e300
    gi_min = (gi - d) % cx.size
    gi_max = (gi + d) % cx.size
    for i in range(gi_min, gi_max):
        dx = cx[i] - x
        dy = cy[i] - y
        d = np.sqrt(dx*dx + dy*dy)
        if d < _min:
            _min = d
            argmin = i
    return argmin
@numba.njit(fastmath=True, parallel=True)
def multi_guess_ind_finder_centering(cx, cy, x, y, inds, gi, d):
    for j in numba.prange(x.size):
        inds[j] = _guess_ind_finder_centering(cx, cy, x[j], y[j], gi[j], d)
def compute_local_coordinates(cx, cy, x, y, gi, nc_i, c_i, newton_tol=1e-12, 
                                                verbose=False, max_iterations=30):
    """
    Find using the coordinates:
    x = X + r n_x
    y = Y + r n_y
    """
    xshape = x.shape
    yshape = y.shape
    x = x.flatten()
    y = y.flatten()
    gi = gi.flatten()

    # function for computing (d^2)_s and its derivative
    def f(t, x, y):
        out = c_i(t)
        C = out[0]
        Cp = out[1]
        Cpp = out[2]
        X = C.real
        Y = C.imag
        Xp = Cp.real
        Yp = Cp.imag
        Xpp = Cpp.real
        Ypp = Cpp.imag
        f = Xp*(X-x) + Yp*(Y-y)
        J = Xpp*(X-x) + Ypp*(Y-y) + Xp*Xp + Yp*Yp
        return f, J

    guess_ind = np.empty(x.size, dtype=int)
    multi_guess_ind_finder_centering(cx, cy, x, y, guess_ind, gi, 10)

    # get starting guess
    t = 2*np.pi/cx.size * guess_ind

    # begin Newton iteration
    rem, J = f(t, x, y)
    mrem = np.abs(rem).max()
    if verbose:
        print('Newton tol: {:0.2e}'.format(mrem))
    iteration = 0
    while mrem > newton_tol:
        delt = -rem/J
        line_factor = 1.0
        while True:
            t_new = t + line_factor*delt
            rem_new, J = f(t_new, x, y)
            mrem_new = np.abs(rem_new).max()
            if (mrem_new < (1-0.5*line_factor)*mrem) or line_factor < 1e-4:
                t = t_new
                # put theta back in [0, 2 pi]
                t[t < 0] += 2*np.pi
                t[t > 2*np.pi] -= 2*np.pi
                rem = rem_new
                mrem = mrem_new
                break
            line_factor *= 0.5
        if verbose:
            print('Newton tol: {:0.2e}'.format(mrem))
        iteration += 1
        if iteration > max_iterations:
            raise Exception('Exceeded maximum number of iterations solving for coordinates .')

    # need to determine the sign now
    C = c_i(t)[0]
    X = C.real
    Y = C.imag
    NC = nc_i(t)
    NX = NC.real
    NY = NC.imag
    r = np.hypot(X-x, Y-y)

    xe1 = X + r*NX
    ye1 = Y + r*NY
    err1 = np.hypot(xe1-x, ye1-y)
    xe2 = X - r*NX
    ye2 = Y - r*NY
    err2 = np.hypot(xe2-x, ye2-y)

    sign = (err1 < err2).astype(int)*2 - 1

    return t, r*sign

################################################################################
# Build Quadtree

class Level(object):
    """
    Set of nodes all at the same level (with same width...)
    For use in constructing Tree objects
    """
    def __init__(self, xmin, ymin, width, parent_ind, intel):
        """
        Inputs:
            xmin,       f8[:], minimum x values for each node
            ymin,       f8[:], minimum y values for each node
            width,      f8,    width of each node (must be same in x/y directions)
            parent_ind, i8[:], index to find parent in prior level array
            intel,      dict of things preserved across levels to avoid recomputations
        """
        self.xmin = xmin
        self.ymin = ymin
        self.width = width
        self.half_width = 0.5*self.width
        self.parent_ind = parent_ind
        self.boundary = intel['bdy']
        self.icb = intel['icb']
        self.ecb = intel['ecb']
        self.imb = intel['imb']
        self.emb = intel['emb']
        self.aicb = intel['aicb']
        self.aecb = intel['aecb']
        self.c_base = intel['c_base']
        self.x_base = intel['x_base']
        self.y_base = intel['y_base']
        self.nc_i = intel['nc_i']
        self.c_i = intel['c_i']
        self.V = intel['V']
        self.VI = intel['VI']
        self.order = intel['order']
        self.tol = intel['tol']
        self.children_ind = -np.ones(self.xmin.shape[0], dtype=int)
        self.basic_computations()
        self.classify()
        self.compute_expansions()
        self.get_where_to_split()
    def basic_computations(self):
        self.xmid = self.xmin + self.half_width
        self.ymid = self.ymin + self.half_width
        self.xmax = self.xmin + self.width
        self.ymax = self.ymin + self.width
        self.n_node = self.xmin.shape[0]
        self.short_parent_ind = self.parent_ind[::4]
    def classify(self):
        self.squares = []
        codes = []
        for i in range(self.n_node):
            # compute square
            xmin = self.xmin[i]
            ymin = self.ymin[i]
            xmax = self.xmax[i]
            ymax = self.ymax[i]
            square = shapely.geometry.Polygon(((xmin,ymin), (xmin,ymax), (xmax,ymax), (xmax, ymin)))
            self.squares.append(square)
            # fully outside coord region? (i.e. disjoint and exterior)
            if self.aecb.PSH.disjoint(square):
                in_coords = False
                ex_coords = True
                code = 2
            # fully inside coord region? (i.e. disjoint and interior)
            elif self.aicb.PSH.contains(square):
                in_coords = False
                ex_coords = True
                code = 3
            # in mapping region
            elif self.emb.PSH.contains(square) and self.imb.PSH.disjoint(square):
                in_coords = True
                ex_coords = False
                code = 1
            # split region
            else:
                in_coords = False
                ex_coords = False
                code = 0
            # codes:
            # 0 --- split, don't compute expansion
            # 1 --- within coordinate region, do compute expansion
            # 2 --- fully outside mapping region (exterior point)
            # 3 --- fully inside mapping region (interior point)
            codes.append(code)
        self.codes = np.array(codes)
    def compute_expansions(self):
        bx = self.boundary.x
        by = self.boundary.y
        # gather up all x's / y's
        self.to_expand = self.codes == 1
        where_to_expand = np.where(self.to_expand)[0]
        n_expand = where_to_expand.size
        if n_expand > 0:
            xs = np.empty([n_expand, self.order*self.order], dtype=float)
            ys = np.empty([n_expand, self.order*self.order], dtype=float)
            dms = np.empty([n_expand, self.order*self.order], dtype=float)
            for i in range(n_expand):
                ind = where_to_expand[i]
                xs[i] = ((self.x_base + 1)/2.0*self.width + self.xmin[ind]).ravel()
                ys[i] = ((self.y_base + 1)/2.0*self.width + self.ymin[ind]).ravel()
                # compute guess index range
                ds = np.hypot(self.xmin[ind]+0.5*self.width-bx, self.ymin[ind]+0.5*self.width-by)
                dms[i] = np.argmin(ds)
            ts, rs = compute_local_coordinates(bx, by, xs, ys, dms, self.nc_i, self.c_i, newton_tol=0.1*self.tol)
            rs = rs.reshape(xs.shape)
            ts = ts.reshape(xs.shape)
            # fix up ts so that it looks continuous
            for i in range(n_expand):
                tt = ts[i]
                mt = tt.mean()
                tt[tt < mt - np.pi] += 2*np.pi
                tt[tt > mt + np.pi] -= 2*np.pi
            r_expansions = np.einsum('ij,...j->...i', self.VI, rs).reshape(n_expand, self.order, self.order)
            t_expansions = np.einsum('ij,...j->...i', self.VI, ts).reshape(n_expand, self.order, self.order)
            # check if tolerance is achieved
            okay = np.zeros(n_expand, dtype=bool)
            for i in range(n_expand):
                r_okay = self.check_coefs(r_expansions[i])
                t_okay = self.check_coefs(t_expansions[i])
                okay[i] = r_okay and t_okay
            # reduce expansions to the acceptable ones
            self.r_expansions = r_expansions[okay]
            self.t_expansions = t_expansions[okay]
            # get trackers into the good expansions
            self.good_expansions = np.zeros(self.n_node, dtype=bool)
            self.good_expansions[where_to_expand] = okay
            self.n_good_expansions = np.sum(okay)
            self.good_expansions_ind = -np.ones(self.n_node, dtype=int)
            self.good_expansions_ind[self.good_expansions] = np.arange(self.n_good_expansions)
            self.bad_expansions = np.logical_and(~self.good_expansions, self.to_expand)
        else:
            self.r_expansions = np.array([]).reshape(0,self.order,self.order)
            self.t_expansions = np.array([]).reshape(0,self.order,self.order)
            self.good_expansions = None
            self.n_good_expansions = 0
            self.good_expansions_ind = -np.ones(self.n_node, dtype=int)
            self.bad_expansions = None
    def check_coefs(self, coefs):
        c1 = coefs[-2:]
        c2 = coefs[:,-2:]
        c1t = np.abs(c1).max()
        c2t = np.abs(c2).max()
        okay1 = c1t < self.tol
        okay2 = c2t < self.tol
        return okay1 and okay2
    def get_where_to_split(self):
        if self.bad_expansions is None:
            self.where_to_split = self.codes == 0
        else:
            self.where_to_split = np.logical_or(self.codes == 0, self.bad_expansions)
        self.leaf = ~self.where_to_split
        self.number_to_split = np.sum(self.where_to_split)
        self.have_to_split = self.number_to_split > 0
        self.children_ind[self.where_to_split] = np.arange(self.number_to_split)*4
class Boundary(object):
    def __init__(self, c, speed_tol=None):
        self.c = c
        self.x = self.c.real
        self.y = self.c.imag
        self.GSB = GSB(x=self.x, y=self.y)
        self.SH = shapely.geometry.Polygon(zip(self.x, self.y))
        self.PSH = shapely.prepared.prep(self.SH)
        self.speed_tol = speed_tol
        if self.speed_tol is not None:
            self.check_value = self.check(self.speed_tol)
    def check(self, speed_tol):
        good = True
        min_speed = self.GSB.speed.min()
        if min_speed < speed_tol:
            good = False
        if not self.SH.is_valid:
            good = False
        return good
class CoordinateTree(object):
    """
    Linear Tree for dealing with local coordinates
    """
    def __init__(self, bx, by, inner_distance, outer_distance, order, tol, mw):
        # boundary is type GSB
        # first compute bounding boundaries for the coordinate region, and check
        # that they are reasonable.
        self.boundary = Boundary(bx + 1j*by)
        self.inner_distance = inner_distance
        self.outer_distance = outer_distance
        self.larger_distance = max(inner_distance, outer_distance)
        self.order = order
        self.tol = tol
        self.minimum_width = mw
        bdy = self.boundary.GSB
        minspeed = bdy.speed.min()
        # get coordinate boundaries
        self.icb = Boundary(bdy.c - inner_distance*bdy.normal_c, 0.1*minspeed)
        self.ecb = Boundary(bdy.c + outer_distance*bdy.normal_c, 0.1*minspeed)
        assert self.icb.check_value, 'coordinates bad; reduce inner_distance'
        assert self.ecb.check_value, 'coordinates bad; reduce outer_distance'
        # get adjusted coordinate boundaries (prevent curve vs. poly mishaps)
        rk = 1.0/self.icb.GSB.curvature
        dd = self.icb.GSB.speed*self.icb.GSB.dt
        ad = np.abs(rk) - np.sqrt(rk**2-(dd/2)**2)
        # this is the distance we have to move by assuming the curve is locally
        # approximated by a circle.  we'll move twice that, to be safe
        adjusted_inner_distance = inner_distance + 2*ad
        self.adjusted_icb = Boundary(bdy.c - adjusted_inner_distance*bdy.normal_c, 0.1*minspeed)
        assert self.adjusted_icb.check_value, 'coordinates bad; reduce inner_distance'
        rk = 1.0/self.ecb.GSB.curvature
        dd = self.ecb.GSB.speed*self.ecb.GSB.dt
        ad = np.abs(rk) - np.sqrt(rk**2-(dd/2)**2)
        adjusted_outer_distance = outer_distance + 2*ad
        self.adjusted_ecb = Boundary(bdy.c + adjusted_outer_distance*bdy.normal_c, 0.1*minspeed)
        assert self.adjusted_ecb.check_value, 'coordinates bad; reduce outer_distance'
        # now compute bounding boundaries for  the mapping region.
        max_distance = max(inner_distance, outer_distance)
        map_distance = max_distance + inner_distance
        okay = False
        while not okay:
            self.imb = Boundary(bdy.c - map_distance*bdy.normal_c, 0.1*minspeed)
            okay = self.imb.check_value
            if not okay: map_distance = adjusted_inner_distance + 0.5*(map_distance-adjusted_inner_distance)
        map_distance = max_distance + outer_distance
        okay = False
        while not okay:
            self.emb = Boundary(bdy.c + map_distance*bdy.normal_c, 0.1*minspeed)
            okay = self.emb.check_value
            if not okay: map_distance = adjusted_outer_distance + 0.5*(map_distance-adjusted_outer_distance)
        # compute xmin, xmax, ymin, ymax,
        _xmin = self.emb.x.min()
        _xmax = self.emb.x.max()
        _ymin = self.emb.y.min()
        _ymax = self.emb.y.max()
        _xran = _xmax - _xmin
        _yran = _ymax - _ymin
        width = max(_xran, _yran) * 1.05
        _xmid = _xmin + 0.5*_xran
        _ymid = _ymin + 0.5*_yran
        self.xmin = _xmid - 0.5*width
        self.ymin = _ymid - 0.5*width
        self.xmax = _xmid + 0.5*width
        self.ymax = _ymid + 0.5*width
        # generate chebyshev nodes
        self.c_base, _ = np.polynomial.chebyshev.chebgauss(self.order)
        self.y_base, self.x_base = np.meshgrid(self.c_base, self.c_base)
        # compute cheb reps of coordinate maps
        self.V = np.polynomial.chebyshev.chebvander2d(self.x_base.flatten(), self.y_base.flatten(), [self.order-1, self.order-1])
        self.VI = np.linalg.inv(self.V)
        # compute interpolaters for the boundary
        self.n_interpolater = periodic_interp1d(self.boundary.GSB.normal_c)
        all_cs = np.row_stack([self.boundary.c, self.boundary.GSB.cp, self.boundary.GSB.cpp])
        self.c_interpolater = periodic_interp1d(all_cs)
        # gather computed information into a dictionary so it doesn't have to
        # be recomputed by Levels
        self.intel = {
            'bdy'    : self.boundary,
            'icb'    : self.icb,
            'ecb'    : self.ecb,
            'imb'    : self.imb,
            'emb'    : self.emb,
            'aicb'   : self.adjusted_icb,
            'aecb'   : self.adjusted_ecb,
            'c_base' : self.c_base,
            'x_base' : self.x_base,
            'y_base' : self.y_base,
            'V'      : self.V,
            'VI'     : self.VI,
            'order'  : self.order,
            'tol'    : self.tol,
            'nc_i'   : self.n_interpolater,
            'c_i'    : self.c_interpolater,
        }
        # generate Levels list
        self.Levels = []
        # setup the first level
        xminarr = np.array((self.xmin,))
        yminarr = np.array((self.ymin,))
        parent_ind_arr = np.array((-1,))
        level_0 = Level(xminarr, yminarr, width, parent_ind_arr, self.intel)
        self.Levels.append(level_0)
        # loop to get the rest of the levels
        current_level = level_0
        while current_level.have_to_split and current_level.width >= 2*self.minimum_width:
            new_level = self.get_new_level(current_level)
            self.Levels.append(new_level)
            current_level = new_level
        # gather depths
        self.levels = len(self.Levels)
        # get aggregated information
        self.leafs         = list_to_typed_list([Level.leaf for Level in self.Levels])
        self.xmids         = list_to_typed_list([Level.xmid for Level in self.Levels])
        self.ymids         = list_to_typed_list([Level.ymid for Level in self.Levels])
        self.xmins         = list_to_typed_list([Level.xmin for Level in self.Levels])
        self.ymins         = list_to_typed_list([Level.ymin for Level in self.Levels])
        self.widths        = np.array([Level.width for Level in self.Levels])
        self.children_inds = list_to_typed_list([Level.children_ind for Level in self.Levels])
        self.r_expansions = list_to_typed_list([Level.r_expansions for Level in self.Levels])
        self.t_expansions = list_to_typed_list([Level.t_expansions for Level in self.Levels])
        self.ind_expansions = list_to_typed_list([Level.good_expansions_ind for Level in self.Levels])
        self.codes = list_to_typed_list([Level.codes for Level in self.Levels])
    def get_new_level(self, level):
        xmins = level.xmin[level.where_to_split]
        ymins = level.ymin[level.where_to_split]
        width = 0.5*level.width
        full_xmins = []
        full_ymins = []
        for i in range(xmins.size):
            full_xmins.append(xmins[i])
            full_ymins.append(ymins[i])
            full_xmins.append(xmins[i]+width)
            full_ymins.append(ymins[i])
            full_xmins.append(xmins[i])
            full_ymins.append(ymins[i]+width)
            full_xmins.append(xmins[i]+width)
            full_ymins.append(ymins[i]+width)
        xminarr = np.array(full_xmins)
        yminarr = np.array(full_ymins)
        parent_ind_arr = np.repeat(np.arange(level.n_node)[level.where_to_split], 4)
        return Level(xminarr, yminarr, width, parent_ind_arr, self.intel)

    """
    Information functions
    """
    def plot_boundaries(self, ax, mpl):
        ax.plot(self.icb.x, self.icb.y, color='red')
        ax.plot(self.ecb.x, self.ecb.y, color='red')
        ax.plot(self.adjusted_icb.x, self.adjusted_icb.y, color='red', linestyle='--')
        ax.plot(self.adjusted_ecb.x, self.adjusted_ecb.y, color='red', linestyle='--')
        ax.plot(self.imb.x, self.imb.y, color='pink')
        ax.plot(self.emb.x, self.emb.y, color='pink')
        ax.plot(self.boundary.x, self.boundary.y, color='blue')
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)

    def plot(self, ax, mpl, **kwargs):
        """
        Create a simple plot to visualize the tree
        Inputs:
            ax,     axis, required: on which to plot things
            mpl,    handle to matplotlib import
        """
        lines = []
        clines = []
        for level in self.Levels:
            nleaves = np.sum(level.leaf)
            xls = level.xmin[level.leaf]
            xhs = level.xmax[level.leaf]
            yls = level.ymin[level.leaf]
            yhs = level.ymax[level.leaf]
            lines.extend([[(xls[i], yls[i]), (xls[i], yhs[i])] for i in range(nleaves)])
            lines.extend([[(xhs[i], yls[i]), (xhs[i], yhs[i])] for i in range(nleaves)])
            lines.extend([[(xls[i], yls[i]), (xhs[i], yls[i])] for i in range(nleaves)])
            lines.extend([[(xls[i], yhs[i]), (xhs[i], yhs[i])] for i in range(nleaves)])
        lc = mpl.collections.LineCollection(lines, colors='lightgray')
        ax.add_collection(lc)
        for level in self.Levels:
            if level.good_expansions is not None:
                nexpansions = np.sum(level.good_expansions)
                xls = level.xmin[level.good_expansions]
                xhs = level.xmax[level.good_expansions]
                yls = level.ymin[level.good_expansions]
                yhs = level.ymax[level.good_expansions]
                clines.extend([[(xls[i], yls[i]), (xls[i], yhs[i])] for i in range(nexpansions)])
                clines.extend([[(xhs[i], yls[i]), (xhs[i], yhs[i])] for i in range(nexpansions)])
                clines.extend([[(xls[i], yls[i]), (xhs[i], yls[i])] for i in range(nexpansions)])
                clines.extend([[(xls[i], yhs[i]), (xhs[i], yhs[i])] for i in range(nexpansions)])
        lc = mpl.collections.LineCollection(clines, colors='black')
        ax.add_collection(lc)
        self.plot_boundaries(ax, mpl)

    def classify(self, xs, ys):
        return full_evaluate(xs, ys, self)

    def grid_classify(self, xv, yv):
        return full_evaluate_grid(xv, yv, self)

################################################################################
# Functions for classifying points and getting their coordinates

# @numba.njit(parallel=True, fastmath=True)
# def locate(xs, ys, leafs, children_ind, xmids, ymids, coef_ind):
#     level_ids = np.empty(xs.size, dtype=numba.int64)
#     inds = np.empty(xs.size, dtype=numba.int64)
#     coef_inds = np.empty(xs.size, dtype=numba.int64)
#     for i in numba.prange(xs.size):
#         x = xs[i]
#         y = ys[i]
#         level_id = 0
#         ind = 0
#         while not leafs[level_id][ind]:
#             xmid = xmids[level_id][ind]
#             ymid = ymids[level_id][ind]
#             ind = children_ind[level_id][ind]
#             if x > xmid and y <= ymid:
#                 ind += 1
#             elif x <= xmid and y > ymid:
#                 ind += 2
#             elif x > xmid and y > ymid:
#                 ind += 3
#             level_id += 1
#         level_ids[i] = level_id
#         inds[i] = ind
#         coef_inds[i] = coef_ind[level_id][ind]
#     return level_ids, inds, coef_inds
@numba.njit(parallel=True, fastmath=True)
def full_locate(xs, ys, leafs, children_ind, xmids, ymids, coef_ind, xm, xM, ym, yM, Codes):
    level_ids = np.empty(xs.size, dtype=numba.int64)
    inds = np.empty(xs.size, dtype=numba.int64)
    coef_inds = np.empty(xs.size, dtype=numba.int64)
    codes = np.empty(xs.size, dtype=numba.int64)
    for i in numba.prange(xs.size):
        x = xs[i]
        y = ys[i]
        # first, see if we're inside the bouding box
        okx = x >= xm and x <= xM
        oky = y >= ym and y <= yM
        ok = okx and oky
        if ok:
            level_id = 0
            ind = 0
            while not leafs[level_id][ind]:
                xmid = xmids[level_id][ind]
                ymid = ymids[level_id][ind]
                ind = children_ind[level_id][ind]
                if x > xmid and y <= ymid:
                    ind += 1
                elif x <= xmid and y > ymid:
                    ind += 2
                elif x > xmid and y > ymid:
                    ind += 3
                level_id += 1
            level_ids[i] = level_id
            inds[i] = ind
            coef_inds[i] = coef_ind[level_id][ind]
            codes[i] = Codes[level_id][ind]
        else:
            level_ids[i] = -1
            inds[i] = -1
            coef_inds[i] = -1
            codes[i] = 2 # fully outside mapping region code...
    return level_ids, inds, coef_inds, codes
@numba.njit(parallel=False, fastmath=True, inline='always')
def _numba_chbevl(x, c):
    x2 = 2*x
    c0 = c[-2]
    c1 = c[-1]
    for i in range(3, len(c) + 1):
        tmp = c0
        c0 = c[-i] - c1
        c1 = tmp + c1*x2
    return c0 + c1*x
@numba.njit(parallel=False, fastmath=True, inline='always')
def _numba_chbevl2(x, y, c, temp):
    order = c.shape[0]
    for i in range(order):
        temp[i] = _numba_chbevl(y, c[i])
    return _numba_chbevl(x, temp)
@numba.njit(parallel=True, fastmath=True)
def evaluate(xs, ys, level_ids, inds, xmins, ymins, widths, rcoefs, tcoefs, icoefs):
    rs = np.empty_like(xs)
    ts = np.empty_like(xs)
    for i in numba.prange(xs.size):
        x = xs[i]
        y = ys[i]
        level_id = level_ids[i]
        ind = inds[i]
        xmin = xmins[level_id][ind]
        ymin = ymins[level_id][ind]
        width = widths[level_id]
        # transform xs/ys back to [-1,1] interval to do cheb sum
        xt = (x - xmin) / width * 2.0 - 1.0
        yt = (y - ymin) / width * 2.0 - 1.0
        # now get the coefs
        icoef = icoefs[i]
        rcoef = rcoefs[level_id][icoef]
        tcoef = tcoefs[level_id][icoef]
        # evaluate
        temp = np.empty(rcoef.shape[0])
        rs[i] = _numba_chbevl2(xt, yt, rcoef, temp)
        ts[i] = _numba_chbevl2(xt, yt, tcoef, temp)
    return rs, ts
def full_evaluate(xs, ys, CTree):
    sh = xs.shape
    xs = xs.ravel()
    ys = ys.ravel()
    level_ids, inds, coef_inds, codes = full_locate(xs, ys, CTree.leafs, CTree.children_inds, CTree.xmids, CTree.ymids, CTree.ind_expansions, CTree.xmin, CTree.xmax, CTree.ymin, CTree.ymax, CTree.codes)
    # evaluate expansions for those where we have them
    good = coef_inds != -1
    x_short = xs[good]
    y_short = ys[good]
    level_ids_short = level_ids[good]
    inds_short = inds[good]
    coef_inds_short = coef_inds[good]
    r_short, t_short = evaluate(x_short, y_short, level_ids_short, inds_short, CTree.xmins, CTree.ymins, CTree.widths, CTree.r_expansions, CTree.t_expansions, coef_inds_short)
    # put these back into full-length vectors
    r = np.zeros_like(xs)
    t = np.zeros_like(xs)
    r[good] = r_short
    t[good] = t_short
    # get interior indicator
    interior = codes == 3
    interior[good] = r_short < 0
    # reshape everything
    r = r.reshape(sh)
    t = t.reshape(sh)
    good = good.reshape(sh)
    interior = interior.reshape(sh)
    return r, t, good, interior

# specialized function for a grid
@numba.njit
def _tag_near_points(x, y, xv, yv, d):
    close = np.zeros((xv.size, yv.size), dtype=numba.boolean)
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
                close[j, k] = close[j, k] or (dist2 < d2)
    return close
def full_evaluate_grid(xv, yv, CTree):
    bx = CTree.boundary.x
    by = CTree.boundary.y
    d = 1.2*CTree.larger_distance
    close =_tag_near_points(bx, by, xv, yv, d)
    wclose = np.where(close)
    cx = xv[wclose[0]]
    cy = yv[wclose[1]]
    out = full_evaluate(cx, cy, CTree)
    sh = (xv.size, yv.size)
    r = np.zeros(sh, dtype=float)
    t = np.zeros(sh, dtype=float)
    g = np.zeros(sh, dtype=bool)
    r[close] = out[0]
    t[close] = out[1]
    g[close] = out[2]
    return r, t, g


