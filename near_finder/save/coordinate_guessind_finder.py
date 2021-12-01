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

################################################################################
# Build Quadtree

class Level(object):
    """
    Set of nodes all at the same level (with same width...)
    For use in constructing Tree objects
    """
    def __init__(self, xmin, ymin, width, parent_ind, guess_ind, intel):
        """
        Inputs:
            xmin,       f8[:], minimum x values for each node
            ymin,       f8[:], minimum y values for each node
            width,      f8,    width of each node (must be same in x/y directions)
            parent_ind, i8[:], index to find parent in prior level array
            guess_ind,  i8[:], index into boundary for closest point to center of this cell, if its in coordinate region (-1 if not)
            intel,      dict of things preserved across levels to avoid recomputations
        """
        self.xmin = xmin
        self.ymin = ymin
        self.width = width
        self.half_width = 0.5*self.width
        self.parent_ind = parent_ind
        self.guess_ind = guess_ind
        self.boundary = intel['bdy']
        self.icb = intel['icb']
        self.ecb = intel['ecb']
        self.imb = intel['imb']
        self.emb = intel['emb']
        self.aicb = intel['aicb']
        self.aecb = intel['aecb']
        self.children_ind = -np.ones(self.xmin.shape[0], dtype=int)
        self.basic_computations()
        self.classify()
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
                # now compute the guess index
                xmid = self.xmid[i]
                ymid = self.ymid[i]
                ds = np.hypot(xmid-self.boundary.x, ymid-self.boundary.y)
                self.guess_ind[i] = np.argmin(ds)
            # split region
            else:
                in_coords = False
                ex_coords = False
                code = 0
            # codes:
            # 0 --- split
            # 1 --- within coordinate region
            # 2 --- fully outside mapping region (exterior point)
            # 3 --- fully inside mapping region (interior point)
            codes.append(code)
        self.codes = np.array(codes)
    def get_where_to_split(self):
        self.where_to_split = self.codes == 0
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
class CoordinateGuessIndFinder(object):
    """
    Linear Tree for dealing with local coordinates
    """
    def __init__(self, bx, by, inner_distance, outer_distance):
        # boundary is type GSB
        # first compute bounding boundaries for the coordinate region, and check
        # that they are reasonable.
        self.boundary = Boundary(bx + 1j*by)
        self.inner_distance = inner_distance
        self.outer_distance = outer_distance
        self.larger_distance = max(inner_distance, outer_distance)
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
        }
        # generate Levels list
        self.Levels = []
        # setup the first level
        xminarr = np.array((self.xmin,))
        yminarr = np.array((self.ymin,))
        parent_ind_arr = np.array((-1,))
        guess_ind_arr = np.array((-1,))
        level_0 = Level(xminarr, yminarr, width, parent_ind_arr, guess_ind_arr, self.intel)
        self.Levels.append(level_0)
        # loop to get the rest of the levels
        current_level = level_0
        while current_level.have_to_split:
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
        self.codes = list_to_typed_list([Level.codes for Level in self.Levels])
        self.guess_inds = list_to_typed_list([Level.guess_ind for Level in self.Levels])
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
        guess_ind_arr = -np.ones(parent_ind_arr.size, dtype=int)
        return Level(xminarr, yminarr, width, parent_ind_arr, guess_ind_arr, self.intel)

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
        lc = mpl.collections.LineCollection(lines, colors='black')
        ax.add_collection(lc)
        self.plot_boundaries(ax, mpl)

    def __call__(self, xs, ys, d=10):
        sh = xs.shape
        xs = xs.ravel()
        ys = ys.ravel()
        level_ids, inds, codes, gi = full_locate(xs, ys, self.leafs, self.children_inds, self.xmids, self.ymids, self.xmin, self.xmax, self.ymin, self.ymax, self.codes, self.guess_inds)
        # refine the guess indeces
        mr = codes == 1
        nmr = np.sum(mr)
        guess_ind = np.empty(nmr, dtype=int)
        multi_guess_ind_finder_centering(self.boundary.x, self.boundary.y, xs[mr], ys[mr], guess_ind, gi[mr], d)
        gi[mr] = guess_ind
        return codes.reshape(sh), gi.reshape(sh)

################################################################################
# Functions for classifying points and getting their coordinates

@numba.njit(parallel=True, fastmath=True)
def full_locate(xs, ys, leafs, children_ind, xmids, ymids, xm, xM, ym, yM, Codes, GuessInds):
    level_ids = np.empty(xs.size, dtype=np.int64)
    inds = np.empty(xs.size, dtype=np.int64)
    codes = np.empty(xs.size, dtype=np.int64)
    gis = np.empty(xs.size, dtype=np.int64)
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
            codes[i] = Codes[level_id][ind]
            gis[i] = GuessInds[level_id][ind]
        else:
            level_ids[i] = -1
            inds[i] = -1
            codes[i] = 2 # fully outside mapping region code...
            gis[i] = -1
    return level_ids, inds, codes, gis

