"""
Example: using ep_finder with a custom 2x2 non-Hermitian matrix.

This demonstrates that the scanner works with arbitrary models, not
just the built-in Perina system.  The cone pre-filter is disabled
(use_cone_test=False) since it is specific to the default model.
"""

import numpy as np
from ep_finder import scan


def matrix_func(kappa, g, gamma_plus):
    """Simple 2x2 non-Hermitian matrix with an EP when kappa^2 = g^2 + gamma_plus^2."""
    return np.array(
        [
            [1j * g, kappa + gamma_plus],
            [kappa - gamma_plus, -1j * g],
        ],
        dtype=complex,
    )


def main():
    eps, total_diags = scan(
        kmin=0.1,
        kmax=2.0,
        gmin=0.1,
        gmax=1.5,
        gpmin=0.05,
        gpmax=1.5,
        matrix_func=matrix_func,
        use_cone_test=False,       # no analytic cone for custom models
        max_depth=6,               # lower depth for speed
        leaf_depth=2,
        verbose=True,
    )

    print(f"\nFound {len(eps)} EPs with {total_diags} diagonalisations.")
    for i, ep in enumerate(eps):
        k, g, gp = ep["kappa"], ep["g"], ep["gamma_plus"]
        print(f"  EP {i+1}: kappa={k:.4f}  g={g:.4f}  gamma_plus={gp:.4f}")


if __name__ == "__main__":
    main()
