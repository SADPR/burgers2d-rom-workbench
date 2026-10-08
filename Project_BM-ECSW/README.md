# BM-ECSW documents

## General formulation and numerical example

Open **`BM_ECSW_LSPG_notes.pdf`**. Its source is `BM_ECSW_LSPG_notes.tex`. The document presents the general formulation followed by a Burgers numerical example.

The technical notes present the HDM and LSPG formulation, classical and by-mode ECSW, modal NNLS training, the Gauss--Newton-type BM update, and the online solver before the Burgers application and results. Sections 1--3 state the general method using arbitrary residual-block sizes, training configurations, and symbolic tolerances. Burgers data, solver settings, timings, and numerical comparisons are confined to the application and timing appendix. The online algorithms explicitly evaluate the local residual blocks and derivatives, assemble the reduced residual and update matrix, solve for the joint coordinate correction, and update the coordinates. They show full-step iterations with a prescribed relative reduced-residual tolerance. Plateau thresholds, iteration limits, and numerical step controls are recorded only for the Burgers experiment. The LSPG equations follow `ares2026closure.pdf`; the sampling notation follows `chapman2017accelerated.pdf`: matrix `G`, target `b`, weights `xi`, and training tolerance `tau`. Generalized coordinates are `q`. Vectors and matrices are bold, with no underlines.

The numerical comparison uses the same 96-mode basis, 225 consecutive projected HDM pairs, seed 42, and relative training tolerance `1e-5` for both methods. Classical ECSW has 2 492 weighted cells; BM-ECSW has 300. The technical notes report only this matched comparison, without an abstract or introduction. BM-ECSW is attributed to Sebastian Rodriguez at the beginning.

## Original proposal and reading notes

`ECSW_General_by_basis.tex` and its PDF contain Sebastian Rodriguez's original proposal and the blue English reading notes. Their original text is preserved. The earlier appended implementation chapter is retained as a historical source in `archive/`; the independent notes provide the implementation report.

## Compile

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error BM_ECSW_LSPG_notes.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error ECSW_General_by_basis.tex
```

Compilation reads the included local tables and figures and does not run simulations.

## Regenerate the figures and tables

From the repository root:

```bash
OPENBLAS_NUM_THREADS=20 OMP_NUM_THREADS=20 .venv/bin/python Project_BM-ECSW/generate_results_figures.py
```

The generator reads saved data only:

- `Results/ECSW_BM_GN_benchmark_20261005/runs.csv` supplies repeated timings and trajectory errors.
- `Results/bm_ecsw_lspg_comparison.txt` supplies the recorded training residuals.
- The two weight files explicitly listed in `RULES` supply reduced-mesh and positive-weight counts.
- `Results/param_snaps/` supplies the HDM trajectory, and the two saved HPROM trajectories supply the solution slices.
- `figures/burgers_problem_3d.pdf` is copied from `Project_YvonMaday/Results_Paper/Figures/euclidean_hprom/burgers_problem_3d.pdf`.

The HPROM benchmark directory also stores individual reports, input hashes, and actual numerical-library thread metadata. The benchmark driver is `benchmark_ecsw_bm.py`. Its reported experiment is three fresh-process repetitions per method and parameter, executed sequentially with 20 native CPU threads. Timings cover mesh/operator setup, initial projection, and 500 time steps within the production HPROM function. Classical and BM use different nonlinear stopping quantities, as documented in the notes.

The Burgers description reflects the implemented conservative flux differences and trapezoidal time-discrete residual. The published closure paper provides notation and benchmark context; its reduced dimension, training configuration, and reported timings are not substituted for those of this experiment.

## FOM speedup reference

`benchmark_fom.py` executes the production FOM solver once per queried parameter, using 20 native numerical-library threads. It measures operator setup and all 500 time steps directly; it does not call the snapshot cache loader. Fresh-process reports and timings are stored in `Results/FOM_benchmark_20261004/`. Comparisons with the saved FOM trajectories occur after timing. Existing snapshots are not overwritten.

All speedup factors in the technical notes use the fresh FOM times as their reference. Per-parameter speedup is the FOM time divided by the mean of three HPROM runs. The overall speedup is the mean FOM time over the three parameters divided by the corresponding overall mean HPROM time. Main tables use minutes, maximum global Euclidean errors, and mesh sizes `n_e` and `n_e^+`; individual HPROM and FOM timings are retained in the appendix. The FOM uses its production `1e-12` residual tolerance, while the HPROM criteria are documented in the notes.

## BM online solution

The BM trajectories in the notes use `--bm-jacobian gauss_newton`.
The update matrix is assembled from local residual first derivatives and
mode-dependent weights. The projected-equation norm is monitored with
relative tolerance `1e-5` and relative plateau tolerance `1e-2`.
All 4 500 steps in the nine BM timing runs attain the projected tolerance;
no plateau stops or failed steps occur.

Reproduce the matched classical/BM benchmark from the repository root:

```bash
.venv/bin/python benchmark_ecsw_bm.py --repeats 3 --threads 20 --bm-jacobian gauss_newton
```

The measurements included in the PDF are in
[`Results/ECSW_BM_GN_benchmark_20261005/summary.md`](../Results/ECSW_BM_GN_benchmark_20261005/summary.md).
The tables and solution slices are regenerated from this benchmark.
The FOM speedup reference remains the measured production runs in
`Results/FOM_benchmark_20261004/`.

## Candidate-score derivation

The training section derives Algorithm 3's signed correlation sum from the
global quadratic fitting objective and connects it to the classical ECSW
selection criterion. It compares a common candidate-weight increment
across modes; subsequent modal NNLS fits allow different weights.

## Offline training times

Table 1 includes single-run estimates from fresh ECSW and BM-ECSW training on 20 native threads. Both methods rebuild the training matrix and weights using the same 96-mode basis, 225 projected state pairs, seed 42, and relative training tolerance `1e-5`. Times include configuration selection, its sampling visualization, loading cached HDM trajectories, matrix assembly, and greedy NNLS fitting; HDM generation, POD construction/loading, final output processing, and online integration are excluded.

The driver is `benchmark_ecsw_bm_offline.py`. Reports, commands, timing data, input hashes, and separately trained weights are stored under `Results/ECSW_BM_offline_benchmark_20261005/`. Existing models and online benchmark outputs are preserved. `generate_results_figures.py` reads the new `runs.csv` to regenerate Table 1 and its offline timing macros.
