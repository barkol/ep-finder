"""
Efficient exceptional-point (EP) scanner for 3D parameter spaces.

Algorithm overview
------------------
The scanner locates degeneracies (exceptional points) of a parameterised
non-Hermitian matrix M(kappa, g, gamma_plus) using an adaptive octree in the
three-dimensional parameter space (kappa, g, gamma_plus).

1. **Octree subdivision** — the parameter volume is recursively split into
   cells.  Large cells are subdivided unconditionally until a configurable
   leaf depth is reached.

2. **Geometric cone pre-filter** (optional, model-specific) — cells that
   cannot intersect the analytic EP cone are pruned at zero cost.

3. **Hierarchical winding-number test** — for each leaf cell the discriminant
   Delta(kappa, gamma_plus) is evaluated on a rectangular contour in the
   (kappa, gamma_plus) plane at fixed g = g_center:

   * *Tier 1 (pretest)* — 12 exact diagonalisations on a coarse contour.
     If the winding number W is clearly zero the cell is pruned early.
   * *Tier 2 (interpolated)* — additional boundary + interior evaluations
     feed a lightweight Matern-5/2 RBF interpolator (pure NumPy, no
     scikit-learn).  The interpolator predicts Delta on a dense contour
     to obtain a robust W estimate.
   * *Tier 3 (refinement)* — triggered only when W is ambiguous; more
     boundary points are added to the interpolator.

4. **EP localisation** — for cells with W = 1 the interpolator's predicted
   zero is iteratively refined: at each step the exact eigenvalue splitting
   is evaluated at the current guess (one diagonalisation) and a local
   interpolator is rebuilt until convergence.  A short Nelder-Mead fallback
   is used if interpolation-based refinement stalls.

5. **Parallelism** — independent cells are processed in parallel via
   ``multiprocessing.Pool``.

Typical cost: ~25 diagonalisations per cell (amortised) and ~5 per EP
localisation, compared with ~64 + ~200 for a naive dense-contour approach.

Public API
----------
>>> from ep_finder.scanner import scan
>>> eps, n_diags = scan(kmin=0.05, kmax=2.5, ...)
"""

import functools
import logging

import numpy as np
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from scipy.optimize import minimize
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Default algorithmic constants (overridable via scan() arguments)    #
# ------------------------------------------------------------------ #

_DEFAULT_EPSILON = 1.0
_DEFAULT_MAX_DEPTH = 10
_DEFAULT_MIN_CELL = 0.0005
_DEFAULT_LEAF_DEPTH = 3
_DEFAULT_W_THRESHOLD = 0.4
_DEFAULT_N_SPARSE = 6
_DEFAULT_N_DENSE = 32
_DEFAULT_N_INTERIOR = 4

# Internal algorithmic constants
_ZERO_DISC_THRESHOLD = 1e-30       # |Delta| below this treated as zero in winding number
_MIN_PARAM_BOUND = 0.01            # Minimum g / gamma_plus below which cells are pruned
_PRETEST_PTS_PER_SIDE = 3          # Tier 1 contour density
_TIER3_PTS_PER_SIDE = 10           # Tier 3 contour density
_PRETEST_W_RAW_TOL = 0.08          # Max |W_raw| for confident W=0 in tier 1
_PRETEST_MIN_DISC = 1e-4           # Min |Delta| for confident W=0 in tier 1
_PRETEST_PHASE_SPREAD = 0.5        # Max phase jump for confident W=0 in tier 1
_MIN_TRAIN_POINTS = 10             # Minimum interpolation training points
_CONVERGENCE_SPLIT = 1e-6          # Eigenvalue splitting convergence threshold
_LOCALIZE_MAX_REFINE = 6           # Max refinement iterations in EP localisation
_LOCALIZE_STAGNATION_TOL = 1e-12   # Position change below which refinement stops
_FALLBACK_DL_THRESHOLD = 0.5       # Delta-lambda above which Nelder-Mead fallback triggers
_FALLBACK_MAX_DIAGS = 15           # Max diagonalisations in Nelder-Mead fallback
_FALLBACK_MIN_GP = 0.001           # Min gamma_plus in fallback objective


# ------------------------------------------------------------------ #
# Named result types                                                  #
# ------------------------------------------------------------------ #

class WindingResult(NamedTuple):
    """Result of hierarchical winding-number computation."""
    w_rounded: int
    w_raw: float
    min_disc: float
    n_diags: int
    interpolator: Optional["FastInterpolator"]
    d_scale: float = 1.0


class LocalizationResult(NamedTuple):
    """Result of EP localisation."""
    ep: Optional[Dict]
    n_diags: int


# ================================================================== #
# 1. CELL                                                             #
# ================================================================== #

@dataclass
class Cell:
    """Axis-aligned box in (kappa, g, gamma_plus) space."""

    kmin: float
    kmax: float
    gmin: float
    gmax: float
    gpmin: float
    gpmax: float
    depth: int = 0

    @property
    def center(self):
        return (
            (self.kmin + self.kmax) / 2,
            (self.gmin + self.gmax) / 2,
            (self.gpmin + self.gpmax) / 2,
        )

    @property
    def size(self):
        return max(
            self.kmax - self.kmin,
            self.gmax - self.gmin,
            self.gpmax - self.gpmin,
        )

    def subdivide(self) -> List["Cell"]:
        km, gm, gpm = self.center
        children = []
        for klo, khi in [(self.kmin, km), (km, self.kmax)]:
            for glo, ghi in [(self.gmin, gm), (gm, self.gmax)]:
                for gplo, gphi in [(self.gpmin, gpm), (gpm, self.gpmax)]:
                    children.append(
                        Cell(klo, khi, glo, ghi, gplo, gphi, self.depth + 1)
                    )
        return children


# ================================================================== #
# 2. MODEL INTERFACE                                                  #
# ================================================================== #

def default_matrix_func(kappa, g, gamma_plus, epsilon=_DEFAULT_EPSILON):
    """Perina drift matrix (default model)."""
    g1d = 2 * gamma_plus
    g2a = 2 * gamma_plus
    return np.array(
        [
            [-1j * g1d / 2, g, epsilon, kappa],
            [-g, -1j * g1d / 2, -kappa, -epsilon],
            [epsilon, kappa, 1j * g2a / 2, g],
            [-kappa, -epsilon, -g, 1j * g2a / 2],
        ],
        dtype=complex,
    )


def eval_disc(kappa, g, gamma_plus, matrix_func):
    """Single diagonalisation -> discriminant Delta in C."""
    M = matrix_func(kappa, g, gamma_plus)
    evals = np.linalg.eigvals(M)
    n = len(evals)
    delta = 1.0 + 0j
    for i in range(n):
        for j in range(i + 1, n):
            delta *= (evals[i] - evals[j]) ** 2
    return delta


def eval_min_split(kappa, g, gamma_plus, matrix_func):
    """Single diagonalisation -> minimum eigenvalue splitting."""
    M = matrix_func(kappa, g, gamma_plus)
    evals = np.linalg.eigvals(M)
    n = len(evals)
    min_s = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            s = abs(evals[i] - evals[j])
            if s < min_s:
                min_s = s
    return min_s


# ================================================================== #
# 3. FAST RBF INTERPOLATION (pure numpy, no sklearn)                  #
# ================================================================== #

def matern52_kernel(r):
    """Matern nu=5/2 kernel: k(r) = (1 + sqrt(5)*r + 5r^2/3) exp(-sqrt(5)*r)."""
    s5r = np.sqrt(5.0) * r
    return (1.0 + s5r + s5r ** 2 / 3.0) * np.exp(-s5r)


class FastInterpolator:
    """
    Lightweight Matern 5/2 interpolator -- pure numpy.

    Orders of magnitude faster than sklearn GPR for small n.
    No hyperparameter optimisation -- length scale derived from cell size.
    """

    def __init__(self, X, y_re, y_im, length_scale, alpha=1e-10):
        """
        Parameters
        ----------
        X : ndarray, shape (n, 2)
            Training coordinates.
        y_re, y_im : ndarray, shape (n,)
            Real / imaginary parts of training values.
        length_scale : array-like, shape (2,)
            Characteristic length scales.
        alpha : float
            Regularisation (diagonal jitter).
        """
        self.X = X
        self.ls = np.array(length_scale)
        n = len(X)

        Xs = X / self.ls
        D = np.sqrt(
            np.maximum(
                np.sum(Xs ** 2, axis=1, keepdims=True)
                - 2 * Xs @ Xs.T
                + np.sum(Xs ** 2, axis=1, keepdims=True).T,
                0.0,
            )
        )

        K = matern52_kernel(D) + alpha * np.eye(n)

        self.w_re = np.linalg.solve(K, y_re)
        self.w_im = np.linalg.solve(K, y_im)

    def predict(self, Xnew):
        """Predict at new points.  Returns (re, im) arrays."""
        Xs = self.X / self.ls
        Xn = Xnew / self.ls
        D = np.sqrt(
            np.maximum(
                np.sum(Xn ** 2, axis=1, keepdims=True)
                - 2 * Xn @ Xs.T
                + np.sum(Xs ** 2, axis=1, keepdims=True).T,
                0.0,
            )
        )
        Kstar = matern52_kernel(D)
        return Kstar @ self.w_re, Kstar @ self.w_im

    def predict_complex(self, Xnew):
        """Predict complex values at new points."""
        re, im = self.predict(Xnew)
        return re + 1j * im


# ================================================================== #
# 4. CONE TEST (optional, model-specific)                             #
# ================================================================== #

def cone_intersects_cell(cell, epsilon):
    """Check if EP cone surface intersects the cell volume."""
    n_g = max(5, int((cell.gmax - cell.gmin) / 0.1) + 1)
    for g in np.linspace(cell.gmin, cell.gmax, n_g):
        for R in [abs(epsilon + g), abs(epsilon - g)]:
            k_closest = max(cell.kmin, min(cell.kmax, 0.0))
            gp_closest = max(cell.gpmin, min(cell.gpmax, 0.0))
            dist_min_sq = k_closest ** 2 + gp_closest ** 2
            k_far = max(abs(cell.kmin), abs(cell.kmax))
            gp_far = max(abs(cell.gpmin), abs(cell.gpmax))
            dist_max_sq = k_far ** 2 + gp_far ** 2
            if dist_min_sq <= R ** 2 <= dist_max_sq:
                return True
    return False


# ================================================================== #
# 5. WINDING NUMBER FROM INTERPOLATION                                #
# ================================================================== #

def winding_number(values):
    """Winding number of a complex curve around the origin."""
    total = 0.0
    n = len(values)
    for i in range(n):
        j = (i + 1) % n
        if abs(values[i]) < _ZERO_DISC_THRESHOLD or abs(values[j]) < _ZERO_DISC_THRESHOLD:
            continue
        total += np.angle(values[j] / values[i])
    return total / (2 * np.pi)


def _make_contour(k0, k1, gp0, gp1, n_per_side):
    """Build rectangular contour points in the (kappa, gamma_plus) plane."""
    n = n_per_side
    bottom = [(k0 + (k1 - k0) * i / n, gp0) for i in range(n)]
    right = [(k1, gp0 + (gp1 - gp0) * i / n) for i in range(n)]
    top = [(k1 - (k1 - k0) * i / n, gp1) for i in range(n)]
    left = [(k0, gp1 - (gp1 - gp0) * i / n) for i in range(n)]
    return bottom + right + top + left


def _eval_disc_contour(contour, gc, matrix_func):
    """Evaluate Delta on contour points.  Returns (disc_values, n_diags)."""
    disc_values = []
    n_diags = 0
    for kv, gpv in contour:
        try:
            d = eval_disc(kv, gc, gpv, matrix_func)
            n_diags += 1
            disc_values.append(d)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
            logger.debug("Discriminant evaluation failed at (%.6f, %.6f, %.6f): %s",
                         kv, gc, gpv, exc)
            disc_values.append(np.nan)
    return disc_values, n_diags


def _points_near(existing, candidate, tol=1e-10):
    """Return True if candidate (x, y) is within tol of any existing point."""
    for ex, ey in existing:
        if abs(ex - candidate[0]) < tol and abs(ey - candidate[1]) < tol:
            return True
    return False


def _dedup_points(existing_pts, new_pts, tol=1e-10):
    """Filter new_pts to exclude those already in existing_pts (within tol)."""
    if not existing_pts:
        return list(new_pts)
    existing = np.array(existing_pts)
    result = []
    for p in new_pts:
        diffs = np.abs(existing - np.array(p))
        if not np.any((diffs[:, 0] < tol) & (diffs[:, 1] < tol)):
            result.append(p)
    return result


def cell_winding_hierarchical(cell, matrix_func, n_sparse, n_dense,
                              n_interior, w_threshold):
    """
    Hierarchical winding number with three tiers.

    Returns WindingResult(w_rounded, w_raw, min_disc, n_diags, interpolator, d_scale).
    """
    kc, gc, gpc = cell.center
    k0, k1 = cell.kmin, cell.kmax
    gp0 = max(cell.gpmin, _MIN_PARAM_BOUND)
    gp1 = cell.gpmax

    _empty = WindingResult(0, 0.0, np.inf, 0, None)

    if gp0 >= gp1 or gc < _MIN_PARAM_BOUND or gpc < _MIN_PARAM_BOUND:
        return _empty

    # -- Tier 1: pretest --
    contour_t1 = _make_contour(k0, k1, gp0, gp1, n_per_side=_PRETEST_PTS_PER_SIDE)
    disc_t1, n_diags = _eval_disc_contour(contour_t1, gc, matrix_func)

    disc_t1_clean = [
        d for d in disc_t1 if not (isinstance(d, float) and np.isnan(d))
    ]
    if len(disc_t1_clean) < 8:
        return WindingResult(0, 0.0, np.inf, n_diags, None)

    disc_t1_arr = np.array(disc_t1_clean)
    w_t1 = winding_number(disc_t1_arr)
    w_t1_round = round(w_t1)
    min_disc_t1 = np.min(np.abs(disc_t1_arr))

    phase_spread = np.max(np.abs(np.diff(np.angle(disc_t1_arr))))
    if (
        w_t1_round == 0
        and abs(w_t1) < _PRETEST_W_RAW_TOL
        and min_disc_t1 > _PRETEST_MIN_DISC
        and phase_spread < _PRETEST_PHASE_SPREAD
    ):
        return WindingResult(0, w_t1, min_disc_t1, n_diags, None)

    # -- Tier 2: interpolated W --
    contour_t2 = _make_contour(k0, k1, gp0, gp1, n_per_side=n_sparse)
    k_int = np.linspace(k0, k1, n_interior + 2)[1:-1]
    gp_int = np.linspace(gp0, gp1, n_interior + 2)[1:-1]
    interior = [(ki, gi) for ki in k_int for gi in gp_int]

    new_points = _dedup_points(contour_t1, contour_t2 + interior)

    disc_new, nd2 = _eval_disc_contour(new_points, gc, matrix_func)
    n_diags += nd2

    X_train = []
    D_vals = []
    for pt, d in zip(contour_t1, disc_t1):
        if not (isinstance(d, float) and np.isnan(d)):
            X_train.append(list(pt))
            D_vals.append(d)
    for pt, d in zip(new_points, disc_new):
        if not (isinstance(d, float) and np.isnan(d)):
            X_train.append(list(pt))
            D_vals.append(d)

    if len(X_train) < _MIN_TRAIN_POINTS:
        return WindingResult(0, 0.0, np.inf, n_diags, None)

    X_train = np.array(X_train)
    D_vals = np.array(D_vals)

    D_scale = max(np.max(np.abs(D_vals.real)),
                  np.max(np.abs(D_vals.imag)), _ZERO_DISC_THRESHOLD)
    D_re_n = D_vals.real / D_scale
    D_im_n = D_vals.imag / D_scale

    ls = [max(k1 - k0, _MIN_PARAM_BOUND) / 2, max(gp1 - gp0, _MIN_PARAM_BOUND) / 2]
    try:
        interp = FastInterpolator(X_train, D_re_n, D_im_n, ls)
    except np.linalg.LinAlgError:
        return WindingResult(0, 0.0, np.inf, n_diags, None)

    contour_dense = np.array(
        _make_contour(k0, k1, gp0, gp1, n_per_side=n_dense)
    )
    D_dense = interp.predict_complex(contour_dense) * D_scale
    min_disc = np.min(np.abs(D_dense))
    w = winding_number(D_dense)

    w_round = round(w)
    if abs(w - w_round) > w_threshold:
        # -- Tier 3: ambiguous -- add more points --
        contour_t3 = _make_contour(k0, k1, gp0, gp1, n_per_side=_TIER3_PTS_PER_SIDE)
        extra = _dedup_points(X_train.tolist(), contour_t3)
        disc_extra, nd3 = _eval_disc_contour(extra, gc, matrix_func)
        n_diags += nd3

        for pt, d in zip(extra, disc_extra):
            if not (isinstance(d, float) and np.isnan(d)):
                X_train = np.vstack([X_train, [pt[0], pt[1]]])
                D_re_n = np.append(D_re_n, d.real / D_scale)
                D_im_n = np.append(D_im_n, d.imag / D_scale)

        try:
            interp = FastInterpolator(X_train, D_re_n, D_im_n, ls)
        except np.linalg.LinAlgError:
            return WindingResult(-1, w, min_disc, n_diags, None)

        D_dense = interp.predict_complex(contour_dense) * D_scale
        min_disc = np.min(np.abs(D_dense))
        w = winding_number(D_dense)
        w_round = round(w)

        if abs(w - w_round) > w_threshold:
            return WindingResult(-1, w, min_disc, n_diags, interp, D_scale)

    return WindingResult(w_round, w, min_disc, n_diags, interp, D_scale)


# ================================================================== #
# 6. EP LOCALISATION FROM INTERPOLATION                               #
# ================================================================== #

def localize_ep_interpolated(cell, matrix_func, interp, epsilon, d_scale=1.0):
    """
    Find EP via iterative interpolator refinement (Bayesian-style).

    Typical cost: 3--5 diagonalisations vs ~30 for Nelder-Mead.

    Returns LocalizationResult(ep, n_diags).
    """
    kc, gc, gpc = cell.center
    k0, k1 = cell.kmin, cell.kmax
    gp0 = max(cell.gpmin, _MIN_PARAM_BOUND)
    gp1 = cell.gpmax
    ls = [max(k1 - k0, _MIN_PARAM_BOUND) / 2, max(gp1 - gp0, _MIN_PARAM_BOUND) / 2]

    n_diags = 0

    # --- Initial: find zero of interpolator (free) ---
    def interp_objective(x):
        pt = np.array([[x[0], x[1]]])
        re, im = interp.predict(pt)
        return (re[0] ** 2 + im[0] ** 2) * d_scale ** 2

    result = minimize(
        interp_objective,
        [kc, gpc],
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-16, "maxiter": 500},
    )
    kv_opt = np.clip(result.x[0], k0, k1)
    gpv_opt = np.clip(result.x[1], gp0, gp1)

    # --- Iterative refinement ---
    refine_X = []
    refine_y = []  # min_split^2 values

    for iteration in range(_LOCALIZE_MAX_REFINE):
        dl = eval_min_split(kv_opt, gc, gpv_opt, matrix_func)
        n_diags += 1

        if dl < _CONVERGENCE_SPLIT:
            break

        refine_X.append([kv_opt, gpv_opt])
        refine_y.append(dl ** 2)

        if len(refine_X) >= 3:
            Xr = np.array(refine_X)
            yr = np.array(refine_y)
            try:
                local_interp = FastInterpolator(
                    Xr, yr, np.zeros_like(yr), ls, alpha=1e-10
                )

                def local_obj(x):
                    pt = np.array([[x[0], x[1]]])
                    pred_re, _ = local_interp.predict(pt)
                    return pred_re[0]

                res = minimize(
                    local_obj,
                    [kv_opt, gpv_opt],
                    method="Nelder-Mead",
                    options={"xatol": 1e-10, "fatol": 1e-16, "maxiter": 200},
                )
                kv_new = np.clip(res.x[0], k0, k1)
                gpv_new = np.clip(res.x[1], gp0, gp1)

                if abs(kv_new - kv_opt) + abs(gpv_new - gpv_opt) > _LOCALIZE_STAGNATION_TOL:
                    kv_opt, gpv_opt = kv_new, gpv_new
                else:
                    break  # stagnated
            except (np.linalg.LinAlgError, ValueError):
                pass
        else:
            # With < 3 points, try analytic cone guess
            for alpha in [epsilon + gc, abs(epsilon - gc)]:
                ksq = alpha ** 2 - gpv_opt ** 2
                if ksq > 0:
                    k_guess = np.sqrt(ksq)
                    if k0 <= k_guess <= k1:
                        kv_opt = k_guess
                        break
            else:
                kv_opt = kv_opt * 0.8 + kc * 0.2
                gpv_opt = gpv_opt * 0.8 + gpc * 0.2

    # Final evaluation
    dl = eval_min_split(kv_opt, gc, gpv_opt, matrix_func)
    n_diags += 1

    # Fallback: short Nelder-Mead if GPR refinement failed
    if dl > _FALLBACK_DL_THRESHOLD:
        n_fall = [0]

        def fallback_obj(x):
            if n_fall[0] >= _FALLBACK_MAX_DIAGS:
                return 1e10
            kv, gpv = x
            if gpv < _FALLBACK_MIN_GP:
                return 1e10
            n_fall[0] += 1
            return eval_min_split(kv, gc, gpv, matrix_func) ** 2

        res = minimize(
            fallback_obj,
            [kc, gpc],
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-16, "maxiter": 50},
        )
        n_diags += n_fall[0]
        kv_opt, gpv_opt = res.x
        dl = np.sqrt(max(res.fun, 0))

    if dl < _FALLBACK_DL_THRESHOLD:
        return LocalizationResult(
            {
                "kappa": abs(kv_opt),
                "g": gc,
                "gamma_plus": gpv_opt,
                "delta_lambda": dl,
                "depth": cell.depth,
            },
            n_diags,
        )

    return LocalizationResult(None, n_diags)


# ================================================================== #
# 7. PARALLEL WORKER                                                  #
# ================================================================== #

# Module-level config set by _init_worker for each Pool process.
_cfg: Optional[dict] = None


def _init_worker(cfg):
    """Initialiser called once per worker process."""
    global _cfg
    _cfg = cfg


def _worker(cell_tuple):
    """Process one cell.  Returns (action, ep, children, n_diags)."""
    cfg = _cfg
    matrix_func = cfg["matrix_func"]
    use_cone = cfg["use_cone"]
    epsilon = cfg["epsilon"]
    min_cell = cfg["min_cell"]
    leaf_depth = cfg["leaf_depth"]
    max_depth = cfg["max_depth"]
    n_sparse = cfg["n_sparse"]
    n_dense = cfg["n_dense"]
    n_interior = cfg["n_interior"]
    w_threshold = cfg["w_threshold"]

    cell = Cell(*cell_tuple)

    if cell.size < min_cell or cell.gmin < _MIN_PARAM_BOUND or cell.gpmin < _MIN_PARAM_BOUND:
        return ("prune", None, None, 0)

    if use_cone and not cone_intersects_cell(cell, epsilon):
        return ("prune", None, None, 0)

    # Large cells: subdivide without computing W
    if cell.depth < leaf_depth:
        children = cell.subdivide()
        ct = [
            (c.kmin, c.kmax, c.gmin, c.gmax, c.gpmin, c.gpmax, c.depth)
            for c in children
        ]
        return ("split", None, ct, 0)

    # Hierarchical winding number
    wr = cell_winding_hierarchical(
        cell, matrix_func, n_sparse, n_dense, n_interior, w_threshold
    )

    if wr.w_rounded == 0:
        return ("prune", None, None, wr.n_diags)

    if wr.w_rounded == 1 and wr.interpolator is not None:
        lr = localize_ep_interpolated(
            cell, matrix_func, wr.interpolator, epsilon, d_scale=wr.d_scale
        )
        return ("ep", lr.ep, None, wr.n_diags + lr.n_diags)

    # W > 1: subdivide
    if cell.depth >= max_depth:
        if wr.interpolator is not None:
            lr = localize_ep_interpolated(
                cell, matrix_func, wr.interpolator, epsilon, d_scale=wr.d_scale
            )
            return ("ep", lr.ep, None, wr.n_diags + lr.n_diags)
        return ("prune", None, None, wr.n_diags)

    children = cell.subdivide()
    ct = [
        (c.kmin, c.kmax, c.gmin, c.gmax, c.gpmin, c.gpmax, c.depth)
        for c in children
    ]
    return ("split", None, ct, wr.n_diags)


# ================================================================== #
# 8. MAIN PUBLIC API                                                  #
# ================================================================== #

def scan(
    kmin: float = 0.05,
    kmax: float = 2.5,
    gmin: float = 0.05,
    gmax: float = 2.2,
    gpmin: float = 0.05,
    gpmax: float = 2.0,
    matrix_func: Optional[Callable] = None,
    use_cone_test: bool = True,
    epsilon: float = _DEFAULT_EPSILON,
    n_workers: Optional[int] = None,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    min_cell: float = _DEFAULT_MIN_CELL,
    leaf_depth: int = _DEFAULT_LEAF_DEPTH,
    w_threshold: float = _DEFAULT_W_THRESHOLD,
    n_sparse: int = _DEFAULT_N_SPARSE,
    n_dense: int = _DEFAULT_N_DENSE,
    n_interior: int = _DEFAULT_N_INTERIOR,
    verbose: bool = True,
) -> Tuple[List[Dict], int]:
    """
    Efficient 3D EP scan with diagonalisation counting.

    Parameters
    ----------
    kmin, kmax : float
        Bounds for kappa.
    gmin, gmax : float
        Bounds for g.
    gpmin, gpmax : float
        Bounds for gamma_plus.
    matrix_func : callable(kappa, g, gamma_plus) -> ndarray, optional
        System matrix constructor.  Default: Perina drift matrix with
        the given *epsilon*.
    use_cone_test : bool
        Use geometric pre-filter (model-specific).  Set False for
        arbitrary models.
    epsilon : float
        Coupling constant passed to the default matrix function and to
        the cone test.
    n_workers : int or None
        Number of parallel workers (default: cpu_count()).
    max_depth : int
        Maximum octree depth.
    min_cell : float
        Minimum cell size below which cells are pruned.
    leaf_depth : int
        Depth at which winding-number evaluation begins.
    w_threshold : float
        Tolerance for rounding the winding number.
    n_sparse : int
        Contour points per side for exact evaluation (tier 2).
    n_dense : int
        Contour points per side for interpolated winding number.
    n_interior : int
        Interior grid size per axis for interpolation training.
    verbose : bool
        Print progress information.

    Returns
    -------
    eps : list of dict
        Found EPs, each with keys 'kappa', 'g', 'gamma_plus',
        'delta_lambda', 'depth'.
    total_diags : int
        Total number of matrix diagonalisations performed.
    """
    # --- Input validation ---
    for name, lo, hi in [("kappa", kmin, kmax), ("g", gmin, gmax),
                          ("gamma_plus", gpmin, gpmax)]:
        if lo >= hi:
            raise ValueError(f"{name} bounds are invalid: {lo} >= {hi}")

    if matrix_func is not None and not callable(matrix_func):
        raise TypeError("matrix_func must be callable(kappa, g, gamma_plus)")

    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")

    if matrix_func is None:
        # Use functools.partial so the function remains picklable for multiprocessing
        matrix_func = functools.partial(default_matrix_func, epsilon=epsilon)

    if n_workers is None:
        n_workers = cpu_count()

    # Worker configuration dict (passed via initialiser, no globals)
    cfg = {
        "matrix_func": matrix_func,
        "use_cone": use_cone_test,
        "epsilon": epsilon,
        "min_cell": min_cell,
        "leaf_depth": leaf_depth,
        "max_depth": max_depth,
        "n_sparse": n_sparse,
        "n_dense": n_dense,
        "n_interior": n_interior,
        "w_threshold": w_threshold,
    }

    N_INIT = 4
    k_edges = np.linspace(kmin, kmax, N_INIT + 1)
    g_edges = np.linspace(gmin, gmax, N_INIT + 1)
    gp_edges = np.linspace(gpmin, gpmax, N_INIT + 1)

    initial_cells = []
    for i in range(N_INIT):
        for j in range(N_INIT):
            for k in range(N_INIT):
                initial_cells.append(
                    (
                        k_edges[i], k_edges[i + 1],
                        g_edges[j], g_edges[j + 1],
                        gp_edges[k], gp_edges[k + 1],
                        0,
                    )
                )

    mode = "cone+interpolation" if use_cone_test else "interpolation only"
    logger.info("=" * 60)
    logger.info("Efficient EP Scanner (octree + fast interpolation)")
    logger.info("=" * 60)
    logger.info(
        "Region: kappa in [%s,%s], g in [%s,%s], gamma_plus in [%s,%s]",
        kmin, kmax, gmin, gmax, gpmin, gpmax,
    )
    logger.info("Mode: %s", mode)
    logger.info(
        "Sparse: %d/side, Dense: %d/side, Interior: %d^2",
        n_sparse, n_dense, n_interior,
    )
    logger.info("Workers: %d, Max depth: %d", n_workers, max_depth)

    all_eps: List[Dict] = []
    total_diags = 0
    stats = {
        "pruned": 0,
        "localized": 0,
        "subdivided": 0,
        "failed": 0,
        "too_deep": 0,
    }

    # For single-worker mode, skip Pool to avoid pickling overhead and
    # to support non-picklable matrix functions (e.g., local closures).
    use_pool = n_workers > 1

    if use_pool:
        pool = Pool(n_workers, initializer=_init_worker, initargs=(cfg,))
    else:
        _init_worker(cfg)
        pool = None

    try:
        for root_idx, root_tuple in enumerate(initial_cells):
            current_level = [root_tuple]

            while current_level:
                valid = []
                for ct in current_level:
                    c = Cell(*ct)
                    if c.size < min_cell:
                        stats["too_deep"] += 1
                    elif c.gmin < _MIN_PARAM_BOUND or c.gpmin < _MIN_PARAM_BOUND:
                        stats["pruned"] += 1
                    else:
                        valid.append(ct)
                if not valid:
                    break

                if use_pool:
                    outcomes = pool.map(_worker, valid)
                else:
                    outcomes = [_worker(ct) for ct in valid]

                next_level = []
                for action, ep, children, nd in outcomes:
                    total_diags += nd
                    if action == "prune":
                        stats["pruned"] += 1
                    elif action == "ep":
                        if ep is not None:
                            all_eps.append(ep)
                            stats["localized"] += 1
                        else:
                            stats["failed"] += 1
                    elif action == "split":
                        if children:
                            cd = children[0][-1]
                            if cd <= max_depth:
                                next_level.extend(children)
                                stats["subdivided"] += 1
                            else:
                                stats["too_deep"] += 1

                current_level = next_level

            if (root_idx + 1) % 16 == 0:
                logger.info(
                    "  Roots %d/%d: %d EPs, %d diags",
                    root_idx + 1, len(initial_cells), len(all_eps), total_diags,
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    logger.info("Stats: %s", stats)
    logger.info("Total diagonalizations: %d", total_diags)
    logger.info("Total EPs found: %d", len(all_eps))
    if all_eps:
        logger.info("Diags per EP: %.0f", total_diags / len(all_eps))

    return all_eps, total_diags
