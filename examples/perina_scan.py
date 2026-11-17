"""
Full scan of the Perina two-mode bosonic system for exceptional points.

This example uses the default model (4x4 drift matrix) with epsilon=1.0
and scans the three-dimensional parameter space (kappa, g, gamma_plus).
Results are printed and plotted as a 3D scatter with symmetry reflections.
"""

import numpy as np
import matplotlib.pyplot as plt
from ep_finder import scan


def main():
    # ---- Run the EP scan with default Perina model ----
    eps, total_diags = scan(
        kmin=0.05,
        kmax=2.5,
        gmin=0.05,
        gmax=2.2,
        gpmin=0.05,
        gpmax=2.0,
        epsilon=1.0,
        use_cone_test=True,
        max_depth=10,
        leaf_depth=3,
        verbose=True,
    )

    # ---- Results summary ----
    print("\n" + "=" * 60)
    print(f"Found {len(eps)} exceptional points")
    print(f"Total diagonalisations: {total_diags}")
    print("=" * 60)

    if not eps:
        print("No EPs found. Try widening the search region.")
        return

    for i, ep in enumerate(eps):
        print(
            f"  EP {i+1:3d}: kappa={ep['kappa']:.6f}  "
            f"g={ep['g']:.6f}  gamma_plus={ep['gamma_plus']:.6f}  "
            f"delta_lambda={ep['delta_lambda']:.2e}"
        )

    # ---- 3D scatter plot with symmetry reflections ----
    kappas = np.array([ep["kappa"] for ep in eps])
    gs = np.array([ep["g"] for ep in eps])
    gps = np.array([ep["gamma_plus"] for ep in eps])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # The Perina system has kappa <-> -kappa and gamma_plus <-> -gamma_plus
    # symmetry, so we reflect the found EPs into all four quadrants.
    for k_sign in [1, -1]:
        for gp_sign in [1, -1]:
            color = "royalblue" if k_sign * gp_sign > 0 else "coral"
            alpha = 1.0 if (k_sign == 1 and gp_sign == 1) else 0.45
            label = "scanned" if (k_sign == 1 and gp_sign == 1) else None
            ax.scatter(
                k_sign * kappas,
                gs,
                gp_sign * gps,
                c=color,
                s=12,
                alpha=alpha,
                label=label,
            )

    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$g$")
    ax.set_zlabel(r"$\gamma_+$")
    ax.set_title("Exceptional Points — Peřina two-mode bosonic system")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("fig_ep_scan.png", dpi=200)
    print(f"\nPlot saved to fig_ep_scan.png")
    plt.show()


if __name__ == "__main__":
    main()
