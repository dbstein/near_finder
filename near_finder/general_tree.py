import numpy as np
import numba
from pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary import Global_Smooth_Boundary as GSB

from numba.typed import List
def list_to_typed_list(L):
    TL = List()
    for x in L: TL.append(x)
    return TL

from near_finder.basic_coordinate_routines import multi_guess_ind_finder, compute_local_coordinates, multi_full_guess_ind_finder
from near_finder.basic_coordinate_routines import gi_finder_same_d, gi_finder_different_d
from near_finder.nufft_interp import periodic_interp1d

import pygeos

################################################################################
"""
Functions to check if bdy shifted by r in the normal direction is still 'okay'

This is measured by the Jacobian of the transform being non-singular out that shift
"""

def compute_max_dists(bdy):
    Rκ = 1.0/bdy.curvature
    Rκp = Rκ[Rκ >  0]
    Rκn = Rκ[Rκ <= 0]
    if Rκp.size > 0:
        max_inner_dist =  np.min(Rκp)
    else:
        max_inner_dist = np.inf
    if Rκn.size > 0:
        max_outer_dist = -np.max(Rκn)
    else:
        max_outer_dist = np.inf
    return max_inner_dist, max_outer_dist

def compute_ξ(bdy):
    return -(bdy.normal_x*bdy.cpp.real + bdy.normal_y*bdy.cpp.imag)/bdy.speed**2
def check_J(ξ, r):
    """
    Check the Jacobian of the coordinate transform at r to ensure |J|>0
    """
    R = np.min(1 + r*ξ)
    if R <= 0:
        raise Exception("Jacobian of coordinate transform is not defined for input distance.")
    return R
def get_outer_dist(ξ, R):
    r1 = (R-1)/np.max(ξ, where=ξ>0, initial=0.0)
    r2 = (1/R-1)/np.min(ξ, where=ξ<0, initial=-0.0)
    return min(r1, r2)
def get_inner_dist(ξ, R):
    r1 = -(R-1)/np.min(ξ, where=ξ<0, initial=-0.0)
    r2 = -(1/R-1)/np.max(ξ, where=ξ>0, initial=0.0)
    return min(r1, r2)
def get_adjustment(bdy):
    """
    Compute distance to move to ensure no polygon vs. curve mishaps
    assuming curve is locally approximated by a circle

    We then return twice this, for safeties sake
    """
    rk = 1.0/bdy.curvature
    dd = bdy.speed*bdy.dt
    dist = np.abs(rk) - np.sqrt(rk**2-(dd/2)**2)
    return 2*np.max(dist)

################################################################################
# General Level object for linear quadtree

class Level:
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
        self.guess_ind = np.full(self.parent_ind.size, -1)
        self.boundary = intel['bdy']
        # inner / outer coordinate boundaries
        self.icb = intel['icb']
        self.ecb = intel['ecb']
        # inner / outer mapping boundaries
        self.imb = intel['imb']
        self.emb = intel['emb']
        # adjusted inner / outer coordinate boundaries
        self.aicb = intel['aicb']
        self.aecb = intel['aecb']
        # wrapped boundaries for guess ind finders
        self.n = intel['n']
        self.ebx = intel['ebx']
        self.eby = intel['eby']
        self.intel = intel
    def __setup_and_classification__(self):
        self.basic_computations()
        self.classify()
        self.prepare_basic_guess_indeces()
    def __finalize__(self):
        self.get_where_to_split()
        self.leaf = ~self.where_to_split
        self.number_to_split = np.sum(self.where_to_split)
        self.have_to_split = self.number_to_split > 0
        self.children_ind[self.where_to_split] = np.arange(self.number_to_split)*4
    def basic_computations(self):
        self.children_ind = -np.ones(self.xmin.shape[0], dtype=int)
        self.xmid = self.xmin + self.half_width
        self.ymid = self.ymin + self.half_width
        self.xmax = self.xmin + self.width
        self.ymax = self.ymin + self.width
        self.n_node = self.xmin.shape[0]
        self.short_parent_ind = self.parent_ind[::4]
        # distance along boundary to look for guess-ind finder
        self.guess_d = int(np.ceil(self.width / np.min(self.boundary.GSB.dt*self.boundary.GSB.speed)))
    def classify(self):
        self.squares = pygeos.box(self.xmin, self.ymin, self.xmax, self.ymax)
        # can maybe optimize by ordering tests and only doing tests where needed???
        test1 = pygeos.disjoint(self.aecb.SH, self.squares)
        test2 = pygeos.contains(self.aicb.SH, self.squares)
        test3 = pygeos.contains(self.emb.SH,  self.squares)
        test4 = pygeos.disjoint(self.imb.SH,  self.squares)
        # test34 = np.logical_and(test3, test4) # old way, is wasteful, especially for FullCoordinateTree
        test34 = np.logical_and.reduce([test3, test4, ~test1, ~test2])
        self.codes = np.zeros(self.n_node, dtype=int)
        self.codes[test1] = 2
        self.codes[test2] = 3
        self.codes[test34] = 1
        self.test34 = test34 # useful for later
    def prepare_basic_guess_indeces(self):
        # prepare guess inds for code 1
        n_code1 = np.sum(self.test34)
        if n_code1 > 0:
            inds = np.empty(n_code1, dtype=int)
            multi_full_guess_ind_finder(self.boundary.x, self.boundary.y, self.xmid[self.test34], self.ymid[self.test34], inds)
            self.guess_ind[self.test34] = inds
    def get_where_to_split(self):
        raise NotImplementedError

    def split(self):
        xmins = self.xmin[self.where_to_split]
        ymins = self.ymin[self.where_to_split]
        width = 0.5*self.width
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
        parent_ind_arr = np.repeat(np.arange(self.n_node)[self.where_to_split], 4)
        return type(self)(xminarr, yminarr, width, parent_ind_arr, self.intel)

################################################################################
# LightweightCoordinateTree Level for linear quadtree

class LCT_Level(Level):
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
        super().__init__(xmin, ymin, width, parent_ind, intel)
        super().__setup_and_classification__()
        super().__finalize__()
    def get_where_to_split(self):
        self.where_to_split = self.codes == 0
        self.leaf = ~self.where_to_split
        self.number_to_split = np.sum(self.where_to_split)
        self.have_to_split = self.number_to_split > 0
        self.children_ind[self.where_to_split] = np.arange(self.number_to_split)*4
        self.good_expansions = np.logical_and(self.codes==1, ~self.where_to_split)

################################################################################
# FullCoordinateTree Level for linear quadtree

class FCT_Level(Level):
    def __init__(self, xmin, ymin, width, parent_ind, intel):
        """
        Inputs:
            xmin,       f8[:], minimum x values for each node
            ymin,       f8[:], minimum y values for each node
            width,      f8,    width of each node (must be same in x/y directions)
            parent_ind, i8[:], index to find parent in prior level array
            intel,      dict of things preserved across levels to avoid recomputations
        """
        super().__init__(xmin, ymin, width, parent_ind, intel)
        self.c_base = intel['c_base']
        self.x_base = intel['x_base']
        self.y_base = intel['y_base']
        self.nc_i = intel['nc_i']
        self.c_i = intel['c_i']
        self.V = intel['V']
        self.VI = intel['VI']
        self.order = intel['order']
        self.tol = intel['tol']
        super().__setup_and_classification__()
        self.compute_expansions()
        super().__finalize__()
    def compute_expansions(self):
        bx = self.boundary.x
        by = self.boundary.y
        # gather up all x's / y's
        self.to_expand = self.codes == 1
        where_to_expand = np.where(self.to_expand)[0]
        n_expand = where_to_expand.size
        if n_expand > 0:
            o2 = self.order*self.order
            xs = np.empty([n_expand, o2], dtype=float)
            ys = np.empty([n_expand, o2], dtype=float)
            gi = np.empty([n_expand, o2], dtype=int)
            # this loop can be replaced ....
            for i in range(n_expand):
                ind = where_to_expand[i]
                xs[i] = ((self.x_base + 1)/2.0*self.width + self.xmin[ind]).ravel()
                ys[i] = ((self.y_base + 1)/2.0*self.width + self.ymin[ind]).ravel()
                gi[i] = self.guess_ind[ind]
            # refine the guess inds
            gi, _ = gi_finder_same_d(self.ebx, self.eby, xs, ys, gi, self.guess_d, self.n)
            guess_s = self.boundary.GSB.t[gi]
            ts, rs = compute_local_coordinates(self.c_i, self.nc_i, xs, ys, guess_s, 0.1*self.tol, 50, False)
            # fix up ts so that it looks continuous within the box
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

################################################################################
# Heavyweight boundary class with prepared geometric data

class Boundary(object):
    """
    Heavyewight boundary class
    """
    def __init__(self, bdy):
        """
        Construct rich "Boundary" type. Input "bdy" can either be:
            (1) GlobalSmoothBoundary type (from pybie2d)
            (2) complex form of (x, y) points of boundary
        """
        if type(bdy) == GSB:
            self.GSB = bdy
        else:
            self.GSB = GSB(c=bdy)
        self.c = self.GSB.c
        self.x = self.GSB.x
        self.y = self.GSB.y
        self.SH = pygeos.polygons([*zip(self.x, self.y)])
        pygeos.prepare(self.SH)
    def is_valid(self):
        return self.SH.is_simple

################################################################################
# General CoordinateTree Class

class CoordinateTree:
    """
    Find local coordinates (s, r) given (x, y) using the coordinates:
        x = X(s) + r n_x(s)
        y = Y(s) + r n_y(s)
    Given a globally parametrized smooth, closed curve
        (defined by GlobalSmoothBoundary class)

    This transformation is local in the sense that it is well-defined only for
        sufficiently small r.  To be precise the Jacobian of this transformation is:
            J = φ(s)(1 - r n(s)*X_ss(s)/φ(s)^2) = φ(s)(1 + rξ(s)),
            where φ(s) = sqrt(X'(s)^2 + Y'(s)^2), the speed of the transformation
            and ξ(s) = -n(s)*X_ss(s)/φ(s)^2
        So in order for the transformation to be well defined, we must have that:
            min(rξ) > -1.
        In practice, we will want to keep this significantly greater than -1.

    These routines work with two bounding curves:
        (1) The bounding curves where we want to solve for coordinates
            (the coordinate region)
        (2) The bounding curves where we will try to solve for coordinates
            (the mapping region)
        Allowing breathing room between where we want coordinates and where
            we will try to solve for them enables a great may optimizations.

    The user can attempt to set coordinate/mapping regions in two different ways.
        (1) Set a inner_dist and an outer_dist for the coordinate region
            if this is chosen, the user provides an inner_dist and outer_dist,
            which define bounding curves:
                inner_curve = X - n*inner_dist
                outer_curve = X + n*outer_dist
            (note the assumption of an outward pointing normal!)

            In this case, no error is raised unless the Jacobian will actually
            become singular (i.e min(rξ) <= -1) for either r=outer_dist or r=-inner_dist.
            However if the transform is nearly singular, the algorithm will have to
            work much harder, and might fail.

            When this is used, the mapping region is chosen automatically,
                by choosing curves that push a given ratio further towards 
                singularity than the curves chosen by the user. By default,
                that ratio is taken to be 2. If the user would like to adjust it,
                they can change R. Increasing R from the default of 2 will reduce
                the size of the mapping area. This increases the speed of future
                coordinate solves, but also increases the time to form the tree.
        
        (2) Set the ratios R and S. 
            The coordinate region is then chosen so that:
                φ/R < |J(r)| < φR.
            I.e. to get outer_dist, we choose the largest value r such that:
                φ/R < |J(r)| < φR
            and to get an innner_dist, we choose the largest value r such that:
                φ/R < |J(-r)| < φR

            The mapping region is chosen similarly, but to satisfy:
                φ/S < |J(r)| < φS.

            Clearly, R must be > 1. The default value is 2; taking it too small
            is unnecessarily restrictive while taking it too large can force the
            solver to have to compute coordinate values where the transform is
            poorly defined.

            We note that a small adjustment is made to the coordinate curve,
            pushing it *slightly* further out than it would otherwise be, in
            order to ensure that we have no polynomial vs. curve type mishaps

            Of course, S>R, and when not set, is taken by default to be 2*R.

    This Class can operate in two modes: basic or interpolatory.
    If parameters is None, it operates in basic mode, otherwise parameters must
        be a dictionary, as detailed below, in whichi case it operates
        in interpolatory mode

    Basic Mode:
        This mode features fast construction and is typically appropriate when
        this function will only be called several times. Instantiation creates a
        minimial tree structure that allows O(log n) classification of points (x, y)
        into interior and exterior points, and points close enough to the curve
        to allow their coordinates to be computed.  This lookup also returns
        guess indeces for a Newton solver, allowing coordinates to be computed
        on-the-fly.
    Interpolatory mode:
        This mode has a slower construction time and is appropriate if this
        function will be called many times. Instantiation creates a rich tree
        that creates Chebyshev representations of the coordinate functions,
        and thus Newton solves for the coordinates do not have to be done on-the-fly.
        The user must provide a dictionary of the form:
            parameters = {
                'tol'   : tolerance to build Chebyshev representations to, should be less than 10^{-14},
                'order' : order of Chebyhsev polynomials to use
            }
            Neither need to be provided, and interpolatory mode can be instantiated with an empty
            dictionary parameters = {}.
            In this case, tol=1e-12 will be used by default. When order is not provided,
            it will be taken to be 2*int(np.ceil(-0.5*np.log10(tol)))
            i.e. the order is taken to be the number of digits requested (using only even orders)
    """
    def __init__(self, bdy, inner_dist=None, outer_dist=None, R=2, S=None):
        """
        Initialize CoordinateFinder
            bdy: type GlobalSmoothBoundary
        """
        self.bdy = bdy
        self._compute_bounding_curves(inner_dist, outer_dist, R, S)

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
        self.width = width     
        # compute interpolaters for the boundary
        self.n_interpolater = periodic_interp1d(self.bdy.normal_c)
        all_cs = np.row_stack([self.bdy.c, self.bdy.cp, self.bdy.cpp])
        self.c_interpolater = periodic_interp1d(all_cs)
        # wrapped boundaries for guess-index finders
        self.n = self.bdy.N
        self.ebx = np.pad(self.boundary.x, (self.n, self.n), mode='wrap')
        self.eby = np.pad(self.boundary.y, (self.n, self.n), mode='wrap')
        # gather computed information into a dictionary so it doesn't have to
        # be recomputed by Levels
        self.intel = {
            'bdy'    : self.boundary,
            'icb'    : self.icb,
            'ecb'    : self.ecb,
            'imb'    : self.imb,
            'emb'    : self.emb,
            'aicb'   : self.aicb,
            'aecb'   : self.aecb,
            'nc_i'   : self.n_interpolater,
            'c_i'    : self.c_interpolater,
            'n'      : self.n,
            'ebx'    : self.ebx,
            'eby'    : self.eby,
        }

    def _compute_bounding_curves(self, inner_dist, outer_dist, R, S):
        self.max_inner_dist, self.max_outer_dist = compute_max_dists(self.bdy)
        if inner_dist is not None and outer_dist is not None:
        # distances should be positive!
            if inner_dist < 0 or outer_dist < 0:
                raise Exception('Provide positive (unsigned) distances.')
            if inner_dist > self.max_inner_dist:
                raise Exception('Requested inner distance greater than max allowed by curvature.')
            if outer_dist > self.max_outer_dist:
                raise Exception('Requested outer distance greater than max allowed by curvature.')
            self.inner_coordinate_dist = inner_dist
            self.outer_coordinate_dist = outer_dist
            # now we set mapping dist by moving "R" towards the maximum
            self.inner_coordinate_ratio = self.inner_coordinate_dist / self.max_inner_dist
            self.inner_mapping_ratio = self.inner_coordinate_ratio + (1.0 - self.inner_coordinate_ratio) / R
            self.inner_mapping_dist = self.inner_mapping_ratio * self.max_inner_dist
            self.outer_coordinate_ratio = self.outer_coordinate_dist / self.max_outer_dist
            self.outer_mapping_ratio = self.outer_coordinate_ratio + (1.0 - self.outer_coordinate_ratio) / R
            self.outer_mapping_dist = self.outer_mapping_ratio * self.max_outer_dist
            # adjust so that they don't get too big (constrain each mapping_dist to be only its coordinate_dist + total_coordinate_dist)
            total_coordinate_width = self.inner_coordinate_dist + self.outer_coordinate_dist
            max_inner_mapping_dist = self.inner_coordinate_dist + total_coordinate_width
            max_outer_mapping_dist = self.outer_coordinate_dist + total_coordinate_width
            self.inner_mapping_dist = min(self.inner_mapping_dist, max_inner_mapping_dist)
            self.outer_mapping_dist = min(self.outer_mapping_dist, max_outer_mapping_dist)
        # ξ = compute_ξ(self.bdy)
        # if inner_dist is not None and outer_dist is not None:
        #     inverse_inner_R = check_J(ξ, -inner_dist)
        #     inverse_outer_R = check_J(ξ, outer_dist)
        #     adj = 1.0 - 1/R
        #     inner_S = 1/(adj*inverse_inner_R)
        #     outer_S = 1/(adj*inverse_outer_R)
        #     self.inner_coordinate_dist = inner_dist
        #     self.outer_coordinate_dist = outer_dist
        #     self.inner_mapping_dist = get_inner_dist(ξ, inner_S)
        #     self.outer_mapping_dist = get_outer_dist(ξ, outer_S)
        #     # yet, whatever happens, don't let these get bigger than the total coordinate width
        #     total_coordinate_width = self.inner_coordinate_dist + self.outer_coordinate_dist
        #     max_inner = self.inner_coordinate_dist + total_coordinate_width
        #     max_outer = self.outer_coordinate_dist + total_coordinate_width
        #     self.inner_mapping_dist = min(self.inner_mapping_dist, max_inner)
        #     self.outer_mapping_dist = min(self.outer_mapping_dist, max_outer)
        elif inner_dist is None and outer_dist is None:
            # THE LOGIC HERE NEEDS TO BE CHECKED

            # check to see if we have a circle
            if np.abs(ξ).max() < 1e-14:
                # we have a circle! set inner and outer_dists accordingly
                cx = np.sum(bdy.x*bdy.weights)/np.sum(bdy.weights)
                cy = np.sum(bdy.y*bdy.weights)/np.sum(bdy.weights)
                R = np.hypot(bdy.x[0] - cx, bdy.y[0] - cy)
                self.inner_coordinate_dist = R/2
                self.outer_coordinate_dist = R/2
                self.inner_mapping_dist = 3*R/4
                self.outer_mapping_dist = 3*R/4
            else:
                assert R>1, "R must be > 1"
                if S is None:
                    S = 2*R
                else:
                    assert S>R, "S must be > R"
                self.inner_coordinate_dist, self.outer_coordinate_dist = get_dists(ξ, R)
                self.inner_mapping_dist, self.outer_mapping_dist = get_dists(ξ, S)
        else:
            raise Exception("If providing one of inner_dist or outer_dist, must provide both")
        self.icb = Boundary(GSB(c=self.bdy.c - self.inner_coordinate_dist*self.bdy.normal_c))
        self.ecb = Boundary(GSB(c=self.bdy.c + self.outer_coordinate_dist*self.bdy.normal_c))
        self.imb = Boundary(GSB(c=self.bdy.c - self.inner_mapping_dist*self.bdy.normal_c))
        self.emb = Boundary(GSB(c=self.bdy.c + self.outer_mapping_dist*self.bdy.normal_c))
        # compute adjusted coordinate distances (to prevent polygon vs. curve mishaps)
        self.adjusted_inner_coordinate_dist = self.inner_coordinate_dist + get_adjustment(self.icb.GSB)
        self.adjusted_outer_coordinate_dist = self.outer_coordinate_dist + get_adjustment(self.ecb.GSB)
        # double check that our adjusted distances are less than mapping distances
        # for now, raise an error as I'm not sure how to automatically fix it,
        # and regardless, it probably indicates a severely underrefined curve
        bad1 = self.adjusted_inner_coordinate_dist > self.inner_mapping_dist
        bad2 = self.adjusted_outer_coordinate_dist > self.outer_mapping_dist
        if bad1 or bad2:
            raise Exception("Adjusted coordinate distances are larger than mapping distance.  You should refine your boundary.")
        self.aicb = Boundary(GSB(c=self.bdy.c - self.adjusted_inner_coordinate_dist*self.bdy.normal_c))
        self.aecb = Boundary(GSB(c=self.bdy.c + self.adjusted_outer_coordinate_dist*self.bdy.normal_c))
        # compatibility shim
        self.boundary = Boundary(self.bdy)

    def _build_tree(self):
        # setup the first level
        self.Levels = [self._get_initial_level()]
        # loop to get the rest of the levels
        while self.Levels[-1].have_to_split and self.Levels[-1].width >= 2*self.minimum_width:
            self.Levels.append(self.Levels[-1].split())
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

    """
    Information functions
    """
    def plot_boundaries(self, ax, mpl):
        ax.plot(self.icb.x, self.icb.y, color='red')
        ax.plot(self.ecb.x, self.ecb.y, color='red')
        ax.plot(self.aicb.x, self.aicb.y, color='red', linestyle='--')
        ax.plot(self.aecb.x, self.aecb.y, color='red', linestyle='--')
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

    def __call__(self, xs, ys, inner_coord=None, outer_coord=None, grid=False, **kwargs):
        if grid:
            return self.grid_classify(xs, ys, inner_coord, outer_coord, **kwargs)
        else:
            return self.classify(xs, ys, inner_coord, outer_coord, **kwargs)

    def grid_classify(self, xv, yv, inner_coord=None, outer_coord=None, **kwargs):
        close = _tag_near_points(self.bdy.x, self.bdy.y, xv, yv, 1.2*max(self.inner_coordinate_dist, self.outer_coordinate_dist))
        wclose = np.where(close)
        cx = xv[wclose[0]]
        cy = yv[wclose[1]]
        out = self.classify(cx, cy, inner_coord, outer_coord, **kwargs)
        sh = (xv.size, yv.size)
        codes = np.full(sh, np.nan, dtype=int)
        interior = np.full(sh, np.nan, dtype=bool)
        computed = np.zeros(sh, dtype=bool)
        r = np.full(sh, np.nan, dtype=float)
        t = np.full(sh, np.nan, dtype=float)
        codes[close] = out[0]
        interior[close] = out[1]
        computed[close] = out[2]
        r[close] = out[3]
        t[close] = out[4]
        return codes, interior, computed, r, t

    def classify(self, xs, ys, inner_coord=None, outer_coord=None, **kwargs):
        """
        inputs:
            xs, float(*)
            ys, float(*)
            inner_coord, optional, float
            outer_coord, optional, float
            **kwargs, optional, see below

        Classify the points, xs, ys, that is, return:
        code, interior, rs, ts

        inner_coord and outer_coord are distances (r) from the boundary for which
        corodinates will be computed. these may be more, but not less, restrictive
        than those for which the tree was initialized.  if they are left as None,
        the values for which the tree was initialized will be used.

        kwargs are to be passed along to Newton solver (newton_tol, max_iterations, verbose)
            newton_tol,     optional, float, defaults to 1e-12
            max_iterations, optional, int,   defaults to 50
            verbose,        optional, bool,  defaults to False

        ------------------------------------------------------------------------
        Returns:

        code, int(*):
            1 = within coordinate region, i.e. -inner_coord <= r <= outer_coord
            2 = outside of coordinate region, i.e. r > outer_coord (or coords undefined)
            3 = inside coordinate region, i.e. r < -inner_coord (or coords undefined)

        interior, bool(*):
            True --- point lies inside boundary   (r < 0)
            False --- point lies outside/on boundary (r >= 0)

        computed, bool(*):
            where coordinates were actually computed (everwhere code==1, and possibly more)

        rs, float(*):
            within coordinate region, gives signed distance to boundary (r coord)
        ts, float(*):
            within coordinate region, gives azimuthal coordinate (in [0, 2π))

        for both rs & ts --- be careful to note that while much of the array is
            nan, there will be r & t values wherever computed==True, which may
            lie outside of the coordinate region
        """
        if inner_coord is None: inner_coord = self.inner_coordinate_dist
        if outer_coord is None: outer_coord = self.outer_coordinate_dist

        if inner_coord > self.inner_coordinate_dist:
            raise Exception('Requested inner_coord exceeds guaranteed coordinate region.')
        if outer_coord > self.outer_coordinate_dist:
            raise Exception('Requested outer_coord exceeds guaranteed coordinate region.')

        # coarse classification
        sh = xs.shape
        xs = xs.ravel()
        ys = ys.ravel()
        level_ids, inds, codes, locs = full_locate(xs, ys, self.leafs, self.children_inds, self.xmids, self.ymids, self.xmin, self.xmax, self.ymin, self.ymax, self.codes, self.locatable)

        # compute coordinates
        code, interior, computed, rs, ts = self._compute_coordinates(xs, ys, inner_coord, outer_coord, level_ids, inds, codes, locs, **kwargs)
        # fix ts
        ts[ts < 0.0] += 2*np.pi
        ts[ts >= 2*np.pi] -= 2*np.pi
        return code.reshape(sh), interior.reshape(sh), computed.reshape(sh), rs.reshape(sh), ts.reshape(sh)

################################################################################
# Specific Implementation for LightweightCoordinateTree

class LightweightCoordinateTree(CoordinateTree):
    def __init__(self, bdy, inner_dist=None, outer_dist=None, R=2, S=None):
        self.minimum_width = 1e-15 # allow user to configure?
        super().__init__(bdy, inner_dist, outer_dist, R, S)
        # general build
        super()._build_tree()
        # class specific finalization
        self._finalize_tree()
    def _finalize_tree(self):
        self.level_guess_ds = np.array([L.guess_d for L in self.Levels])        
        self.guess_inds = list_to_typed_list([Level.guess_ind for Level in self.Levels])
        self.locatable = self.guess_inds
    def _get_initial_level(self):
        xminarr = np.array((self.xmin,))
        yminarr = np.array((self.ymin,))
        parent_ind_arr = np.array((-1,))
        return LCT_Level(xminarr, yminarr, self.width, parent_ind_arr, self.intel)

    def _compute_coordinates(self, xs, ys, inner_coord, outer_coord, level_ids, inds, codes, gis, **kwargs):
        if 'newton_tol' not in kwargs: kwargs['newton_tol'] = 1e-12
        if 'max_iterations' not in kwargs: kwargs['max_iterations'] = 50
        if 'verbose' not in kwargs: kwargs['verbose'] = False
        # initialize output arrays
        computed = np.zeros(xs.size, dtype=bool)
        interior = codes == 3
        ts = np.full_like(xs, np.nan)
        rs = np.full_like(ys, np.nan)
        # this is where we can try to solve for coordinates (but might not!)
        mr = codes == 1
        nmr = np.sum(mr)
        if nmr > 0:
            # get distances to refine guess-indeces by
            ds = self.level_guess_ds[level_ids]
            # refine our guess indeces
            guess_ind, guess_dists = gi_finder_different_d(self.ebx, self.eby, xs[mr], ys[mr], gis[mr], ds[mr], self.n)
            # now sign guess_dists --- so that we can toss points that are clearly outside of the requested region
            cs = xs[mr] + 1j*ys[mr]
            nc = self.boundary.GSB.normal_c[guess_ind]
            cc = self.boundary.GSB.c[guess_ind]
            cs1 = cc + guess_dists*nc
            cs2 = cc - guess_dists*nc
            err1 = np.abs(cs1-cs)
            err2 = np.abs(cs2-cs)
            sign = (err1 < err2).astype(int)*2 - 1
            guess_dists *= sign
            # now we have an estimate of the signed distance
            # throw away those that are definitely outside of the requested region ---
            # this currently assumes we have the best guess (within one gridpoint)
            # which I'm nearly sure is true but probably should prove
            good_dists1 = guess_dists >= -(inner_coord+self.boundary.GSB.max_h)
            good_dists2 = guess_dists <=  (outer_coord+self.boundary.GSB.max_h)
            good_dists = np.logical_and(good_dists1, good_dists2)
            # those with good_dists==True are those we want to expend coordinate effort for
            # reduce those we'll solve coords for down to those that we can't
            # easily discard
            ngd = np.sum(good_dists)
            gmr = np.zeros(xs.size, dtype=bool)
            if ngd > 0:
                good_guess_ind = guess_ind[good_dists]
                guess_t = self.boundary.GSB.t[good_guess_ind]
                gmr[mr] = good_dists
                # compute coordinates for these
                t, r = compute_local_coordinates(self.c_interpolater, self.n_interpolater, xs[gmr], ys[gmr], guess_t, **kwargs)
                # repack into full-size arrays, with nan's everywhere we didn't try to compute coords
                ts[gmr] = t
                rs[gmr] = r
                # fill in computed output
                computed[gmr] = True
            # correct interior
            code1_interior = np.zeros(nmr, dtype=bool)
            if ngd > 0:
                code1_interior[good_dists] = r < 0
            code1_interior[~good_dists] = guess_dists[~good_dists] < 0
            interior[mr] = code1_interior
            # fix code
            internal_code = np.ones(nmr, dtype=int)
            # the following may seem wrong for those who we know coords
            # but this will be overwritten momentarily anyway
            internal_code[guess_dists < 0] = 3
            internal_code[guess_dists > 0] = 2
            if ngd > 0:
                gmr_internal_code = np.ones(r.size, dtype=int)
                gmr_internal_code[r < -inner_coord] = 3
                gmr_internal_code[r > outer_coord] = 2
                internal_code[good_dists] = gmr_internal_code
            # now update the whole code
            codes[mr] = internal_code

        return codes, interior, computed, rs, ts

################################################################################
# Specific Implementation for FullCoordinateTree

class FullCoordinateTree(CoordinateTree):
    def __init__(self, bdy, inner_dist=None, outer_dist=None, R=2, S=None, parameters=None):
        
        super().__init__(bdy, inner_dist, outer_dist, R, S)

        # deal with Full Coordinate Specific Parameters
        if parameters is None:
            parameters = {}
        if type(parameters) != dict:
            raise Exception("parameters must be None or dict")
        tol = parameters['tol'] if 'tol' in parameters else 1e-12
        self._set_tol(tol)
        order = parameters['order'] if 'order' in parameters else self._get_order(self.tol)
        self._set_order(order)
        mw = parameters['minimum_width'] if 'minimum_width' in parameters else 1e-12
        self._set_mw(mw)

        # generate chebyshev nodes
        self.c_base, _ = np.polynomial.chebyshev.chebgauss(self.order)
        self.y_base, self.x_base = np.meshgrid(self.c_base, self.c_base)
        # compute cheb reps of coordinate maps
        self.V = np.polynomial.chebyshev.chebvander2d(self.x_base.flatten(), self.y_base.flatten(), [self.order-1, self.order-1])
        self.VI = np.linalg.inv(self.V)

        # add to intel dictionary
        self.intel['c_base'] = self.c_base
        self.intel['x_base'] = self.x_base
        self.intel['y_base'] = self.y_base
        self.intel['V'] = self.V
        self.intel['VI'] = self.VI
        self.intel['order'] = self.order
        self.intel['tol'] = self.tol

        # general build
        super()._build_tree()

        # class specific finalization
        self._finalize_tree()

    def _get_initial_level(self):
        xminarr = np.array((self.xmin,))
        yminarr = np.array((self.ymin,))
        parent_ind_arr = np.array((-1,))
        return FCT_Level(xminarr, yminarr, self.width, parent_ind_arr, self.intel)

    def _set_tol(self, tol):
        assert tol >= 1.0e-14, "tol must be >= 1.0e-14"
        self.tol = tol
    def _set_order(self, order):
        assert type(order) == int, "order must be an integer"
        self.order = order
    def _set_mw(self, mw):
        self.minimum_width = mw
    def _get_order(self, tol):
        return 2*int(np.ceil(-0.5*np.log10(tol)))

    def _finalize_tree(self):
        self.r_expansions = list_to_typed_list([Level.r_expansions for Level in self.Levels])
        self.t_expansions = list_to_typed_list([Level.t_expansions for Level in self.Levels])
        self.ind_expansions = list_to_typed_list([Level.good_expansions_ind for Level in self.Levels])
        self.locatable = self.ind_expansions

    def _compute_coordinates(self, xs, ys, inner_coord, outer_coord, level_ids, inds, codes, coef_inds, **kwargs):
        # evaluate expansions for those where we have them
        good = coef_inds != -1
        x_short = xs[good]
        y_short = ys[good]
        level_ids_short = level_ids[good]
        inds_short = inds[good]
        coef_inds_short = coef_inds[good]
        r_short, t_short = evaluate(x_short, y_short, level_ids_short, inds_short, self.xmins, self.ymins, self.widths, self.r_expansions, self.t_expansions, coef_inds_short)
        # put these back into full-length vectors
        r = np.full_like(xs, np.nan)
        t = np.full_like(xs, np.nan)
        r[good] = r_short
        t[good] = t_short
        # generate computed
        computed = good
        # get interior indicator
        interior = codes == 3
        interior[good] = r_short < 0
        # fix up the code variable
        internal_code = np.ones(x_short.size, dtype=int)
        internal_code[r_short < -inner_coord] = 3
        internal_code[r_short > outer_coord] = 2
        codes[good] = internal_code

        return codes, interior, computed, r, t

################################################################################
# Low-Level Functions for classification and coordinate finding

# Loc is CoefInds for Full // GuessInds for Lightweight
@numba.njit(parallel=True, fastmath=True)
def full_locate(xs, ys, leafs, children_ind, xmids, ymids, xm, xM, ym, yM, Codes, Loc):
    level_ids = np.empty(xs.size, dtype=numba.int64)
    inds = np.empty(xs.size, dtype=numba.int64)
    codes = np.empty(xs.size, dtype=numba.int64)
    locs = np.empty(xs.size, dtype=numba.int64)
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
            locs[i] = Loc[level_id][ind]
        else:
            level_ids[i] = -1
            inds[i] = -1
            codes[i] = 2 # fully outside mapping region code...
            locs[i] = -1
    return level_ids, inds, codes, locs

@numba.njit(parallel=False, fastmath=True)
def _numba_chbevl(x, c):
    x2 = 2*x
    c0 = c[-2]
    c1 = c[-1]
    for i in range(3, len(c) + 1):
        tmp = c0
        c0 = c[-i] - c1
        c1 = tmp + c1*x2
    return c0 + c1*x
@numba.njit(parallel=False, fastmath=True)
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


