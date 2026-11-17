# ep-finder

`ep-finder` is a Python package that locates **exceptional points** (EPs) of non-Hermitian matrices that
depend on three real parameters, i.e. points where two or more eigenvalues *and* their eigenvectors coalesce.
The repository also contains the code, the cached numerical data and the technical notes behind the article

> K. Bartkiewicz, H. Wojewoda, and F. Nori, *Selecting eigenvalue permutations by static detunings and path
> interference in nonreciprocal bosonic networks* (2026).

Every figure of the article and of its technical notes is drawn by one command from the cached data in
`paper/data` (section [Reproducing the article](#reproducing-the-article)).

## Contents

| Path | What it holds |
|------|---------------|
| `ep_finder/` | the EP scanner (library) |
| `tests/` | unit tests of the library (`pytest`) |
| `examples/` | example scans: Peřina two-mode model, a custom 2×2 model, the EP3 unfolding of a 3×3 Jordan block |
| `paper/figures/` | one plotting script per figure of the article and of the technical notes |
| `paper/data/` | cached data read by the plotting scripts (`.npz`, `.json`, `.png`) and the exact values as `.csv` |
| `paper/data_scripts/` | scripts that rebuild the cached data from the result files |
| `paper/results/` | result files of the simulations (read-out Monte Carlo, dynamics, robustness counts) |
| `paper/verification/` | numerical checks of the derivation steps (technical notes, section N1) with expected output |
| `docs/technical_notes.pdf` | technical notes: numerical tests, read-out protocol, ladders, tests of the threshold law |

## The EP scanner

### Algorithm

The scanner subdivides the parameter box (κ, g, γ₊) by an octree and tests each cell in three stages.

1. **Geometric pre-filter (optional).** For models with a known EP surface (for the default model the cones
   κ² + γ₊² = (ε ± g)²) cells that cannot intersect the surface are pruned without any diagonalization.
2. **Hierarchical winding number.** The discriminant Δ = Π(λᵢ − λⱼ)² is single-valued and complex for real
   parameters, and its winding number around zero on a closed contour counts the enclosed EPs. A coarse contour
   (12 diagonalizations) rejects empty cells; a Matérn-5/2 interpolator (pure NumPy) built from a few more
   points predicts Δ on a dense contour; more boundary points are added only if the winding number is ambiguous.
3. **Interpolator-guided localization.** In a cell with winding number 1 the zero of the interpolated Δ is
   refined by exact evaluations and local re-interpolation, with a Nelder–Mead fallback.

Independent cells are processed in parallel (`multiprocessing`).

### Installation

```bash
pip install -e .            # library only (numpy, scipy)
pip install -e ".[dev]"     # plus pytest
pip install -e ".[paper]"   # plus matplotlib, pillow, sympy for paper/
```

### Quick start

```python
from ep_finder import scan

# default model: the Peřina two-mode bosonic system
eps, n_diags = scan(kmin=0.05, kmax=2.5, gmin=0.05, gmax=2.2, gpmin=0.05, gpmax=2.0)
print(f"Found {len(eps)} EPs using {n_diags} diagonalizations")
for ep in eps[:5]:
    print(f"  kappa={ep['kappa']:.4f}, g={ep['g']:.4f}, "
          f"gamma_plus={ep['gamma_plus']:.4f}, delta_lambda={ep['delta_lambda']:.2e}")
```

### Custom model

```python
import numpy as np
from ep_finder import scan

def my_matrix(kappa, g, gamma_plus):
    """Any non-Hermitian matrix depending on three parameters."""
    return np.array([[1j * g, kappa + gamma_plus],
                     [kappa - gamma_plus, -1j * g]], dtype=complex)

eps, n_diags = scan(kmin=0.1, kmax=3.0, gmin=0.1, gmax=3.0, gpmin=0.1, gpmax=2.0,
                    matrix_func=my_matrix, use_cone_test=False, max_depth=8)
```

For measured data, wrap an interpolant of the measured matrices (e.g. `scipy.interpolate.RegularGridInterpolator`)
as `matrix_func` and set `use_cone_test=False`.

### API

`scan(**kwargs) -> (eps, n_diags)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kmin, kmax` | `0.05, 2.5` | κ range |
| `gmin, gmax` | `0.05, 2.2` | g range |
| `gpmin, gpmax` | `0.05, 2.0` | γ₊ range |
| `matrix_func` | Peřina 4×4 drift matrix | `callable(kappa, g, gamma_plus) -> ndarray` |
| `use_cone_test` | `True` | geometric pre-filter (set `False` for other models) |
| `epsilon` | `1.0` | energy scale of the default model and of the cone test |
| `max_depth` | `10` | maximum octree depth |
| `min_cell` | `0.0005` | minimum cell size |
| `n_workers` | `cpu_count()` | parallel workers |
| `verbose` | `True` | print progress |

`eps` is a list of dicts with keys `kappa`, `g`, `gamma_plus`, `delta_lambda`, `depth`; `n_diags` is the total
number of diagonalizations. Further options are documented in the docstring of `ep_finder.scanner.scan`.

The default model is the drift matrix of the two-mode bosonic system of Peřina Jr. *et al.*, Quantum **6**, 883
(2022), with beam-splitter coupling ε, two-mode squeezing κ, single-mode squeezing g and gain–loss γ₊. Its EPs
lie on the cones κ² + γ₊² = (ε ± g)². The interpolator-guided search follows the Gaussian-process approach of
P. Egenlauf, P. Rommel, and J. Main, Mach. Learn.: Sci. Technol. **5**, 015045 (2024).

### Tests and examples

```bash
pytest                              # 13 tests, about 15 s
python examples/custom_model.py
python examples/perina_scan.py      # writes fig_ep_scan.png
python examples/ep3_unfolding.py    # writes fig_ep3_epfinder.png
```

## Reproducing the article

### Requirements

* Python ≥ 3.9 with NumPy, SciPy, Matplotlib, Pillow and SymPy (`pip install -e ".[paper]"`);
* a LaTeX installation with `pdflatex`, `amsmath`, `amssymb`, `bm` and Computer Modern fonts (Matplotlib text is
  typeset by LaTeX), and `pdfcrop`;
* [GLE](https://glx.sourceforge.io) 4.3 for figures 3, 5, 6 and N1 (the scripts call `gle` from `PATH`, or the
  executable named by the environment variable `GLE`);
* optional: [Blender](https://www.blender.org) 4.5 to re-render the four panels of figure 1.

The scripts can be started from any directory. Figures are written to `paper/figures/output/`, intermediate
files to `paper/figures/build/`. Each script takes about one second.

### One command per figure

| Figure | Content | Command | Cached data (`paper/data/`) |
|--------|---------|---------|-----------------------------|
| 1 | eigenvalue permutation under one swept phase | `python paper/figures/fig01_permutations.py` | `fig01_render_{a,b,c,d}.png` |
| 2 | undetuned ring: loop, gap, robustness, radius, time scales, link model | `python paper/figures/fig02_undetuned_ring.py` | `ring_dynamics.npz`, `tables.npz` |
| 3 | programming by static detunings | `python paper/figures/fig03_programming.py` | `fig03_programming.npz` |
| 4 | spectrum, state and group at N = 3 | `python paper/figures/fig04_spectrum_state_group.py` | `ring_dynamics.npz` |
| 5 | interference-selected permutation | `python paper/figures/fig05_interference.py` | `fig05_interference.npz`, `readout_certification.npz` |
| 6 | read-out of rings in simulation | `python paper/figures/fig06_readout.py` | `fig06_readout.npz`, `readout_certification.npz` |
| N1 | read-out maps (technical notes) | `python paper/figures/figN1_readout_maps.py` | `figN1_readout_maps.npz` |
| N2 | ladders per cycle type (technical notes) | `python paper/figures/figN2_ladders.py` | `tables.npz` |

`python paper/figures/make_all.py` draws all eight. The article and the technical notes contain no tables; the
exact values shown in the figures are in `paper/data/*.csv`:

| File | Values |
|------|--------|
| `tab_radius.csv` | radius of persistence and nearest collision, N = 3…14 (figure 2(e)) |
| `tab_sweep_time.csv` | gap, adiabatic and orbit times, mean ΔΓ (figure 2(f)) |
| `tab_link_model.csv` | link model against κ′ (figure 2(g,h)) |
| `tab_ladders.csv`, `tab_link_devices.csv` | ladders per cycle type, ideal and link model (figure N2) |
| `sigma_table.csv` | detunings of every labeled σ for N ≤ 6 (technical notes, section N3) |
| `tab_readout_certification.csv` | certification maps of figures 5(e) and 6(d,e) |
| `tab_readout_maps.csv` | read-out maps of figure N1 |

### Rebuilding the cached data

```bash
python paper/data_scripts/make_all_data.py
```

rebuilds, in a few seconds, `fig01_branches.npz`, `fig01_blender.json`, `ring_dynamics.npz`, `tables.npz`,
`readout_certification.npz`, `figN1_readout_maps.npz` and the two read-out `.csv` files from `paper/results/` and
the `.csv` tables; the output is identical to the shipped files. The panels of figure 1 are rendered by

```bash
blender -b -P paper/data_scripts/render_fig01_blender.py -- 200   # samples per pixel, as for the shipped panels
```

(path tracing, so a new render differs from the shipped one by sampling noise only). The remaining caches
(`fig03_programming.npz`, `fig05_interference.npz`, `fig06_readout.npz`), the files in `paper/results/` and the
`.csv` tables are produced by the simulations below.

### Simulations

`paper/simulations/` holds the code that produces the result files and the remaining caches. Every script resumes
from checkpoints (in its own git-ignored `work/` directory, for `readout/` in `results/readout/`) and writes its final output to `paper/data/` or
`paper/results/`. Several runs take hours; the shipped results let you skip them.

**`programming/`** (figure 3, figure N2, `sigma_table.csv`): `ring_lib.py` (ring model, tracker, exact σ rule,
full link model). Ideal model: `ladder_optimize.py 6` → `program_sigma.py` → `error_mc.py` → `budget.py` →
`tables.py` (writes `data/sigma_table.csv` and `data/tab_ladders.csv`). Link model at κ′ = 20ε and 40ε:
`link_per_sigma.py κ 2 5` → `link_retry.py` → `link_budget.py` → `label_check.py` → `link_fix.py` →
`link_budget.py` (writes `data/tab_link_devices.csv`); `link_design.py` and `link_all_sigma.py` provide the design
and tracking functions. `fig03_data.py` (about 2 min) reproduces `data/fig03_programming.npz` byte for byte from
the intermediates in `results/programming/`.

**`readout/`** (figure 6, figure N1, read-out result files): `readout_lib.py`, `devices.py`, `protocol.py`,
`link_model.py` (2N-mode link model). The intermediate results are shipped in `results/readout/` (`geom.npz`,
`fig3c/`, `proto/`, `link_devices/`, 7 MB; file names carry the configuration keys used by the scripts), so
`aggregate_protocol.py` (→ `data/fig06_readout.npz`, `results/readout_protocol_tables.json`) and
`aggregate_link_devices.py` (→ `results/readout_link_devices.json`) rebuild the outputs in seconds, identical to the
shipped files. The sweeps that produce the intermediates are `geometry.py`, `single_spectrum_fits.py 400 8`,
`protocol_sweep.py 104 12 A` (and stages B, E, P) and `link_devices_sweep.py 100 8` (hours); they skip every
configuration already present. Set `READOUT_WORK` to compute the intermediates in another directory.

**`network/`** (four-mode network, figure 5): `network_lib.py` (tracking, ideal and link model with one reservoir
per link); `build_fig05_interference.py` → `data/fig05_interference.npz` (about 10 s, only if missing);
`run_readout.py` and `run_readout_probe_errors.py` → `results/network_readout_full.jsonl` and
`results/network_readout_phased_probe.jsonl` (read-out by model fitting at κ′ = 40ε; resumable, seeded per task,
several hours on 20 cores; with the shipped files they report "tasks 0"); `link_model_types.py` (cycle type and gap
of the slow modes against φ and κ′); `min_gaps_fig5d.py` recomputes the minimum gaps of figure 5(d) (about 1 min).

**`ring/`** (figures 2 and 4, appendix C): `ring_tools.py` (eigenvalue continuation and exact discriminant sign
counts; needs `python-flint` and `mpmath`); `robustness_mc.py` → `results/robustness_mc.npz` (Monte Carlo of the
orbit outcome against disorder, 2 × 25 × 20 000 draws, multiprocessing); `regimes_dynamics.py` →
`results/regimes_dynamics.npz` (80-digit integration of the state over six orbits); `sweep_time_table.py` with
`high_precision_propagator.py` → `data/tab_sweep_time.csv`; `nearest_collision.py` →
`results/ring_nearest_collision.npz` and `data/tab_radius.csv` (SLSQP from 300 random starts per N). Rerun from the
checkpoints, all four reproduce the shipped files exactly. The producer of `data/tab_link_model.csv` (link model
against κ′, figure 2(g,h)) is not part of this repository.

### Result files

`paper/results/` holds the raw results read by the data scripts. The field names are short abbreviations:
in `network_readout_*.jsonl` each line is one fitted trial with true type `prawda`, fitted type `typ`, verdict
`wer` (`pewne` certified, `odrzut` rejected, `nierozstrz` undecided), fit variant `war` (`prior`: local detunings
as a Gaussian constraint), relative phase `phi`, link `w`, SNR `snr` and error levels `e`, `ep`, `fp`;
in `readout_link_devices.json` the fractions `ok` (certified with the true type), `zle` (false certificate),
`odrz` (rejected), `elem` (labeled σ correct) and `zach` (device keeps its type) are given per device, error level
and SNR.

### Numerical checks of the derivation

```bash
python paper/verification/run_all.py     # about 40 s
```

runs the checks V1–V6 of section N1 of the technical notes (drift matrix from the master equation, determinant
identity and path expansion, cycle count of Theorem 1, threshold law, four-mode network, Newton-hull form of
Proposition 2) and compares their output with `paper/verification/expected/`.

### EP manifolds (appendix C of the article)

The numbers of the EP-manifold maps in appendix C come from the runs of the `ep-finder` octree search made in the
bachelor's thesis of H. Wojewoda (*Quantum Liouvillian exceptional points for bosonic fields with quadratic
Hamiltonians: the Gaussian Process Regression approach*, Adam Mickiewicz University, Poznan, 2026). In those runs
the octree refinement is guided by Gaussian-process regression of the discriminant, every candidate passes four
spectral tests, and no analytic localization is used. The tests rejected none of the 29 926 candidates of the ring
(every degeneracy of the ring is an EP, since rank(M − λ) ≥ N − 1) and 1046 candidates of the Peřina model. For the
N = 3 ring at real parameters the search returns 32 481 ± 8 points on both EP2 sheets (three seeds), with median
eigensplitting (1.1961 ± 0.0006) × 10⁻⁸ and closest point 5.52 × 10⁻³ from the EP3; for the Peřina model it returns
48 242 ± 13 points on the double cones ρ = |g ± ε|, ρ² = γ² + κ². The thesis code and these runs are not part of this
repository; the library in `ep_finder/` implements the octree and interpolator-guided search described above.

## Citation

If you use this code, please cite

> K. Bartkiewicz, H. Wojewoda, and F. Nori, *Selecting eigenvalue permutations by static detunings and path
> interference in nonreciprocal bosonic networks* (2026).

## Authors

Karol Bartkiewicz (maintainer, bark@amu.edu.pl), Hubert Wojewoda and Franco Nori; contributions are listed in
[AUTHORS](AUTHORS).

## AI disclosure

We used the AI coding assistant Claude Code (Anthropic; version 2.1.326, with Claude Opus models, Claude Opus 5.5
in the final stage) to write and run the analysis code and the plotting scripts, to search and check the
literature, to check the numbers in the text against the result files, and to draft and edit parts of the text of
the article. The AI assistant also took part in developing the results, by proposing ideas and carrying out
derivations and simulations. We verified all derivations by reading them line by line, by numerical checks with
code independent of the formulas, and by independent attempts to refute every derivation. All numbers and figures
are produced by our code from the result files, and no image in the article is generated by an AI model (the
panels of figure 1 are rendered by Blender from the computed eigenvalue branches). The same assistant was used to
assemble this repository, to adapt the figure scripts to its layout and to write this README. The authors
reviewed the code, reran the analyses, checked every number, statement and cited source, and take full
responsibility for the content of the article and of this repository.

## License

MIT, see [LICENSE](LICENSE).
