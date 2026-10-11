# BM-ECSW documents

## General formulation and numerical example

Open **`BM_ECSW_LSPG_notes.pdf`**. Its source is `BM_ECSW_LSPG_notes.tex`. The document presents the general formulation followed by a Burgers numerical example.

The technical notes present the HDM and LSPG formulation, classical and by-mode ECSW, modal NNLS training, the Gauss--Newton-type BM update, and the online solver before the Burgers application and results. Sections 1--3 state the general method using arbitrary residual-block sizes, training configurations, and symbolic tolerances. Burgers data, solver settings, timings, and numerical comparisons are presented in the application and computational appendices. The online algorithms explicitly evaluate the local residual blocks and derivatives, assemble the reduced residual and update matrix, solve for the joint coordinate correction, and update the coordinates. They show full-step iterations with a prescribed relative reduced-residual tolerance. Plateau thresholds, iteration limits, and numerical step controls are recorded only for the Burgers experiment. The LSPG equations follow `ares2026closure.pdf`; the sampling notation follows `chapman2017accelerated.pdf`: matrix `G`, target `b`, weights `xi`, and training tolerance `tau`. Generalized coordinates are `q`. Vectors and matrices are bold, with no underlines.

The numerical comparison uses the same 96-mode basis, 225 consecutive projected HDM pairs, seed 42, and relative training tolerance `1e-5` for both methods. Classical ECSW has 2 492 weighted cells; BM-ECSW has 300. The main results use this matched comparison; Appendix A.4 additionally examines the earlier rules and their online behavior. The notes have no abstract or introduction. BM-ECSW is attributed to Sebastian Rodriguez at the beginning.

Appendix A first derives the general gradient-preservation condition (A.1), then applies it to LSPG objectives and online correction matrices (A.2) and to mechanical potentials, Lagrangian structure, and stability (A.3). A.4 presents the AERO-F and Burgers computational evidence, distinguishing exact-Jacobian tests, correction-matrix diagnostics, and observed online behavior.

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

## Burgers stability diagnostics

The Burgers study in Appendix A.4.1 compares the saved BM and classical rules trained with 90
and 225 state pairs. Twelve fresh 500-step trajectories use the production
classical Gauss--Newton and BM Gauss--Newton-type solvers. Results and input hashes are in
`Results/BM_ECSW_stability_20261009/`; `summary.txt` records the protocol and limits.

The diagnostic driver `diagnose_bm_stability.py` also evaluates the derivative
of the implicit time-step map at roots from common previous PROM states.
Exact derivatives are used only for these sensitivity probes, not to produce
the online trajectories. The symmetric part of the correction matrix and the
actual time-step sensitivity are different quantities.

To regenerate only the appendix tables and figure from saved data:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python Project_BM-ECSW/generate_stability_results.py
```

To repeat the experiment, use a new output directory:

```bash
OPENBLAS_NUM_THREADS=10 OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10 .venv/bin/python diagnose_bm_stability.py --threads 10 --output-dir Results/BM_ECSW_stability_new
```

The callbacks record matrices without changing solver returns. Instrumented
times are not used for speed comparisons. No cubature weights are retrained.

The additional classical 90-pair online runs are stored in
`Results/BM_ECSW_stability_classical90_20261009/` and are produced by
`diagnose_classical90_stability.py`. The table generator reads the additional
dataset listed in `Results/BM_ECSW_stability_20261009/table_inputs.json`.
The original comparison trajectories and sensitivity probes are retained.


## Exact-Jacobian gradient test in Burgers

`diagnose_bm_gradient_structure.py` evaluates the exact derivative of the
projected equations, including the residual-weighted second derivatives.
It reuses the saved trajectories and common PROM configurations. No weights
are retrained and no trajectories are integrated again.

The reports, input hashes, 6 000 trajectory-state evaluations, 24 common-state
evaluations, and 40 finite-difference checks are in
`Results/BM_ECSW_gradient_structure_20261009/`. Nonsymmetry here directly tests
the existence of a common scalar objective, unlike the approximate correction
matrix diagnostic. Both BM rules fail this exact gradient test, including the
successful 225-configuration rule. This does not establish a unique cause of
trajectory failure.

Regenerate the table using the saved measurements:

```bash
.venv/bin/python Project_BM-ECSW/generate_gradient_structure_table.py
```

Reproduce the diagnostic with a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python diagnose_bm_gradient_structure.py --threads 1 --output-dir Results/BM_ECSW_gradient_structure_new
```
