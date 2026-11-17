"""Unit tests for the ep_finder.scanner module."""

import logging
import numpy as np
import pytest

from ep_finder.scanner import (
    Cell,
    FastInterpolator,
    WindingResult,
    LocalizationResult,
    cone_intersects_cell,
    scan,
    winding_number,
)


# ------------------------------------------------------------------ #
# test_cell_subdivide                                                  #
# ------------------------------------------------------------------ #

def test_cell_subdivide():
    """A cell subdivides into exactly 8 children at depth+1."""
    cell = Cell(0.0, 2.0, 0.0, 2.0, 0.0, 2.0, depth=3)
    children = cell.subdivide()
    assert len(children) == 8
    for child in children:
        assert child.depth == 4
    # Children should tile the parent volume without gaps
    vol_parent = 2.0 * 2.0 * 2.0
    vol_children = sum(
        (c.kmax - c.kmin) * (c.gmax - c.gmin) * (c.gpmax - c.gpmin)
        for c in children
    )
    assert abs(vol_children - vol_parent) < 1e-12


# ------------------------------------------------------------------ #
# test_fast_interpolator                                               #
# ------------------------------------------------------------------ #

def test_fast_interpolator():
    """FastInterpolator reproduces sin(x)*cos(y) at held-out points."""
    rng = np.random.default_rng(42)
    X_train = rng.uniform(0, np.pi, size=(40, 2))
    y_re = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1])
    y_im = np.zeros(len(X_train))

    interp = FastInterpolator(X_train, y_re, y_im, length_scale=[1.0, 1.0])

    X_test = rng.uniform(0.2, np.pi - 0.2, size=(10, 2))
    pred_re, pred_im = interp.predict(X_test)
    exact = np.sin(X_test[:, 0]) * np.cos(X_test[:, 1])

    assert np.allclose(pred_re, exact, atol=0.05), (
        f"Max error: {np.max(np.abs(pred_re - exact)):.4f}"
    )


# ------------------------------------------------------------------ #
# test_winding_number                                                  #
# ------------------------------------------------------------------ #

def test_winding_number():
    """Unit circle in C gives W=1; a shifted circle not enclosing 0 gives W=0."""
    # Unit circle around origin -> W = 1
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    values_enclosing = np.exp(1j * t)
    w = winding_number(values_enclosing)
    assert abs(w - 1.0) < 0.05, f"Expected W~1, got {w:.4f}"

    # Circle centred at 5+0j, radius 1 -> does not enclose origin -> W = 0
    values_non_enclosing = 5.0 + np.exp(1j * t)
    w0 = winding_number(values_non_enclosing)
    assert abs(w0) < 0.05, f"Expected W~0, got {w0:.4f}"


# ------------------------------------------------------------------ #
# test_cone_intersects                                                 #
# ------------------------------------------------------------------ #

def test_cone_intersects():
    """Known intersection and non-intersection with the EP cone (epsilon=1)."""
    # Cell centred near the cone surface kappa^2 + gamma_plus^2 = (1+g)^2
    # at g=0.5 -> R = 1.5, so kappa=1.5, gamma_plus=0 lies on cone
    cell_hit = Cell(1.3, 1.7, 0.4, 0.6, 0.0, 0.2, depth=0)
    assert cone_intersects_cell(cell_hit, epsilon=1.0)

    # Cell far from any cone surface
    cell_miss = Cell(5.0, 6.0, 0.1, 0.2, 5.0, 6.0, depth=0)
    assert not cone_intersects_cell(cell_miss, epsilon=1.0)


# ------------------------------------------------------------------ #
# test_scan_small                                                      #
# ------------------------------------------------------------------ #

def test_scan_small():
    """
    Scan a tiny region around a known EP of the default Perina model.

    For epsilon=1 the EP cone is kappa^2 + gamma_plus^2 = (epsilon +/- g)^2.
    At g=0.5 the '+' branch gives R=1.5, so (kappa=1.5, gamma_plus~0) is an EP.
    We scan a small box around that point and expect at least one EP.
    """
    eps, total_diags = scan(
        kmin=1.2,
        kmax=1.8,
        gmin=0.3,
        gmax=0.7,
        gpmin=0.05,
        gpmax=0.4,
        use_cone_test=True,
        epsilon=1.0,
        max_depth=7,
        leaf_depth=2,
        n_workers=1,
        verbose=False,
    )

    assert len(eps) >= 1, f"Expected at least 1 EP, found {len(eps)}"
    # The closest EP should be near kappa~1.5, g~0.5
    best = min(eps, key=lambda e: abs(e["kappa"] - 1.5) + abs(e["g"] - 0.5))
    assert abs(best["kappa"] - 1.5) < 0.2, f"kappa={best['kappa']:.4f}, expected ~1.5"
    assert abs(best["g"] - 0.5) < 0.2, f"g={best['g']:.4f}, expected ~0.5"


# ------------------------------------------------------------------ #
# test_scan_invalid_bounds                                             #
# ------------------------------------------------------------------ #

def test_scan_invalid_bounds():
    """scan() raises ValueError for inverted bounds."""
    with pytest.raises(ValueError, match="kappa bounds are invalid"):
        scan(kmin=2.0, kmax=1.0, verbose=False)

    with pytest.raises(ValueError, match="g bounds are invalid"):
        scan(gmin=1.0, gmax=0.5, verbose=False)

    with pytest.raises(ValueError, match="gamma_plus bounds are invalid"):
        scan(gpmin=1.0, gpmax=0.5, verbose=False)


def test_scan_invalid_matrix_func():
    """scan() raises TypeError for non-callable matrix_func."""
    with pytest.raises(TypeError, match="matrix_func must be callable"):
        scan(matrix_func="not_a_function", verbose=False)


def test_scan_invalid_max_depth():
    """scan() raises ValueError for max_depth < 1."""
    with pytest.raises(ValueError, match="max_depth must be >= 1"):
        scan(max_depth=0, verbose=False)


# ------------------------------------------------------------------ #
# test_scan_multiprocessing                                            #
# ------------------------------------------------------------------ #

def test_scan_multiprocessing():
    """Scan with n_workers=2 finds the same EPs as n_workers=1."""
    common = dict(
        kmin=1.2, kmax=1.8, gmin=0.3, gmax=0.7,
        gpmin=0.05, gpmax=0.4, use_cone_test=True,
        epsilon=1.0, max_depth=6, leaf_depth=2, verbose=False,
    )
    eps_single, _ = scan(n_workers=1, **common)
    eps_multi, _ = scan(n_workers=2, **common)

    assert len(eps_multi) >= 1, "Multiprocessing scan found no EPs"
    assert len(eps_single) == len(eps_multi), (
        f"Worker count mismatch: {len(eps_single)} vs {len(eps_multi)}"
    )


# ------------------------------------------------------------------ #
# test_scan_empty_region                                               #
# ------------------------------------------------------------------ #

def test_scan_empty_region():
    """Scanning a region far from any EP returns no results."""
    eps, total_diags = scan(
        kmin=10.0, kmax=11.0, gmin=10.0, gmax=11.0,
        gpmin=10.0, gpmax=11.0, use_cone_test=False,
        max_depth=5, leaf_depth=2, n_workers=1, verbose=False,
    )
    assert len(eps) == 0


# ------------------------------------------------------------------ #
# test_scan_custom_matrix_func                                         #
# ------------------------------------------------------------------ #

def _simple_matrix(kappa, g, gamma_plus):
    """Module-level 2x2 matrix for picklability with multiprocessing."""
    return np.array([
        [1j * gamma_plus, kappa],
        [kappa, -1j * gamma_plus],
    ], dtype=complex)


def test_scan_custom_matrix_func():
    """Scan with a custom 2x2 matrix runs without error."""
    eps, total_diags = scan(
        kmin=0.05, kmax=0.5, gmin=0.05, gmax=0.5,
        gpmin=0.05, gpmax=0.5,
        matrix_func=_simple_matrix,
        use_cone_test=False,
        max_depth=5, leaf_depth=2, n_workers=1, verbose=False,
    )
    assert isinstance(eps, list)
    assert isinstance(total_diags, int)


# ------------------------------------------------------------------ #
# test_named_tuples                                                    #
# ------------------------------------------------------------------ #

def test_named_tuples():
    """WindingResult and LocalizationResult are proper NamedTuples."""
    wr = WindingResult(1, 0.98, 1e-5, 12, None, 1.0)
    assert wr.w_rounded == 1
    assert wr.interpolator is None
    assert wr.d_scale == 1.0

    lr = LocalizationResult({"kappa": 1.5}, 5)
    assert lr.ep["kappa"] == 1.5
    assert lr.n_diags == 5


# ------------------------------------------------------------------ #
# test_logging_output                                                  #
# ------------------------------------------------------------------ #

def test_logging_output(caplog):
    """scan() emits log messages instead of print()."""
    with caplog.at_level(logging.INFO, logger="ep_finder.scanner"):
        scan(
            kmin=1.4, kmax=1.6, gmin=0.4, gmax=0.6,
            gpmin=0.05, gpmax=0.2,
            max_depth=4, leaf_depth=2, n_workers=1, verbose=False,
        )
    assert any("Efficient EP Scanner" in r.message for r in caplog.records)
