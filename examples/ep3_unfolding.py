"""
EP3 unfolding: 3x3 perturbed Jordan block with cusp EP structure.

Physical model: cascaded three-mode non-reciprocal chain with feedback.

    d/dt (a1, a2, a3)^T = M (a1, a2, a3)^T + F_noise

    M(a, b, c) = [[a,  1,  0 ],
                   [0,  b,  1 ],
                   [c,  0, -(a+b)]]

  - a, b: effective detunings of modes 1 and 2
  - c: feedback coupling 1 -> 3
  - tr M = 0 (balanced gain/loss)
  - Unidirectional cascade: M_12 = M_23 = 1

EP structure:
  - EP3 (triple coalescence) at a = b = c = 0 (Jordan block J_3)
  - EP2 surface (discriminant = 0) forms a cusp meeting at the EP3:
        c = ab(a+b) +/- sqrt(4/27) * (a^2 + ab + b^2)^{3/2}
  - A loop in the (a, c) plane at b = 0 encircles 2 EP2 branches
    sharing the middle eigenvalue, giving a 3-cycle monodromy.

This example scans the 3D parameter space with ep_finder and plots
the found EPs against the analytic EP2 surface.
"""

import logging
import sys

import numpy as np
import matplotlib.pyplot as plt

from ep_finder import scan

# EPFinder requires positive parameters and prunes near zero.
# We shift the coordinate system so that the EP3 at (0,0,0)
# maps to (S, S, S) in scanner coordinates.
S = 3.0


def matrix_ep3_shifted(kappa, g, gamma_plus):
    """EP3 model in shifted coordinates for EPFinder."""
    a = kappa - S
    b = g - S
    c = gamma_plus - S
    return np.array([
        [a, 1, 0],
        [0, b, 1],
        [c, 0, -(a + b)],
    ], dtype=complex)


def matrix_ep3(a, b, c):
    """EP3 model in physical coordinates."""
    return np.array([
        [a, 1, 0],
        [0, b, 1],
        [c, 0, -(a + b)],
    ], dtype=complex)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout)

    R = 2.5  # half-range around EP3

    # ---- Run EPFinder ----
    print("Running EPFinder on EP3 unfolding model...")
    eps_raw, total_diags = scan(
        kmin=S - R,
        kmax=S + R,
        gmin=S - R,
        gmax=S + R,
        gpmin=S - R,
        gpmax=S + R,
        matrix_func=matrix_ep3_shifted,
        use_cone_test=False,
        max_depth=8,
        leaf_depth=3,
        min_cell=0.002,
        n_dense=64,
        n_workers=1,
        verbose=True,
    )

    # ---- Refine with Nelder-Mead and filter ----
    from scipy.optimize import minimize

    def min_split_at(params):
        ev = np.linalg.eigvals(matrix_ep3(*params))
        return min(abs(ev[i] - ev[j])
                   for i in range(3) for j in range(i + 1, 3))

    eps = []
    seen = set()
    for ep in eps_raw:
        x0 = np.array([ep["kappa"] - S, ep["g"] - S, ep["gamma_plus"] - S])
        try:
            res = minimize(min_split_at, x0, method='Nelder-Mead',
                           options={'xatol': 1e-6, 'fatol': 1e-10,
                                    'maxiter': 500})
            a_r, b_r, c_r = res.x
            ms = res.fun
        except Exception:
            continue
        if ms > 0.01:
            continue
        key = (round(a_r, 3), round(b_r, 3), round(c_r, 3))
        if key in seen:
            continue
        seen.add(key)
        eps.append({"a": a_r, "b": b_r, "c": c_r, "split": ms})

    print(f"\nFound {len(eps_raw)} raw -> {len(eps)} refined (split < 0.01)")
    print(f"Total diagonalisations: {total_diags}")
    for i, ep in enumerate(eps[:15]):
        print(f"  EP {i + 1}: a={ep['a']:+.4f}  b={ep['b']:+.4f}  "
              f"c={ep['c']:+.4f}  split={ep['split']:.2e}")

    # ---- Analytic EP2 surface ----
    N_surf = 150
    ag = np.linspace(-R, R, N_surf)
    bg = np.linspace(-R, R, N_surf)
    A, B = np.meshgrid(ag, bg)
    P = A**2 + A * B + B**2
    coeff = np.sqrt(4.0 / 27.0)
    C_plus = A * B * (A + B) + coeff * P**1.5
    C_minus = A * B * (A + B) - coeff * P**1.5
    C_plus_m = np.where(np.abs(C_plus) <= R, C_plus, np.nan)
    C_minus_m = np.where(np.abs(C_minus) <= R, C_minus, np.nan)

    # ---- 3-cycle contour (b = 0, loop in a-c plane) ----
    r_contour = 1.5
    phi = np.linspace(0, 2 * np.pi, 500)
    a_loop = r_contour * np.cos(phi)
    c_loop = r_contour * np.sin(phi)
    b_loop = np.zeros_like(phi)

    # ---- Figure ----
    ep_a = np.array([ep["a"] for ep in eps])
    ep_b = np.array([ep["b"] for ep in eps])
    ep_c = np.array([ep["c"] for ep in eps])

    fig = plt.figure(figsize=(18, 14))

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot_surface(A, B, C_plus_m, alpha=0.2, color='#e74c3c',
                     rstride=4, cstride=4, linewidth=0)
    ax1.plot_surface(A, B, C_minus_m, alpha=0.2, color='#3498db',
                     rstride=4, cstride=4, linewidth=0)
    ax1.scatter(ep_a, ep_b, ep_c, c='black', s=15, alpha=0.8,
                zorder=10, label=f'EPFinder ({len(eps)} EPs)')
    ax1.scatter([0], [0], [0], c='red', s=150, marker='*',
                zorder=11, label='EP3')
    ax1.plot(a_loop, b_loop, c_loop, 'g-', lw=2.5, zorder=10,
             label='3-cycle contour')
    ax1.set_xlabel('a')
    ax1.set_ylabel('b')
    ax1.set_zlabel('c')
    ax1.set_title('EPFinder results + analytic EP2 surface')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.view_init(elev=25, azim=-55)

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.plot_surface(A, B, C_plus_m, alpha=0.2, color='#e74c3c',
                     rstride=4, cstride=4, linewidth=0)
    ax2.plot_surface(A, B, C_minus_m, alpha=0.2, color='#3498db',
                     rstride=4, cstride=4, linewidth=0)
    ax2.scatter(ep_a, ep_b, ep_c, c='black', s=15, alpha=0.8, zorder=10)
    ax2.scatter([0], [0], [0], c='red', s=150, marker='*', zorder=11)
    ax2.plot(a_loop, b_loop, c_loop, 'g-', lw=2.5, zorder=10)
    ax2.set_xlabel('a')
    ax2.set_ylabel('b')
    ax2.set_zlabel('c')
    ax2.set_title('View along b (contour plane)')
    ax2.view_init(elev=5, azim=0)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(ep_a, ep_c, c='black', s=8, alpha=0.5,
                label=f'EP projections ({len(eps)})')
    a1d = np.linspace(-R, R, 1000)
    ax3.plot(a1d, coeff * np.abs(a1d)**3, 'r-', lw=1.5,
             label='Analytic (b=0)')
    ax3.plot(a1d, -coeff * np.abs(a1d)**3, 'b-', lw=1.5)
    ax3.plot(0, 0, 'r*', ms=15, zorder=10, label='EP3')
    ax3.plot(a_loop, c_loop, 'g-', lw=2, label='3-cycle contour')
    ax3.set_xlabel('a')
    ax3.set_ylabel('c')
    ax3.set_title('Projection onto (a, c)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.25)
    ax3.set_xlim(-R, R)
    ax3.set_ylim(-R, R)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(ep_a, ep_b, c='black', s=8, alpha=0.5,
                label='EP projections')
    ax4.plot(0, 0, 'r*', ms=15, zorder=10, label='EP3')
    ax4.set_xlabel('a')
    ax4.set_ylabel('b')
    ax4.set_title('Projection onto (a, b)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.25)
    ax4.set_xlim(-R, R)
    ax4.set_ylim(-R, R)

    plt.suptitle('EPFinder scan: EP3 unfolding model\n'
                 'M = [[a,1,0], [0,b,1], [c,0,-(a+b)]]',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_ep3_epfinder.png', dpi=200, bbox_inches='tight')
    print(f"\nSaved: fig_ep3_epfinder.png")
    plt.show()


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
