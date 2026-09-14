# B3 cubature budget study

The tested draw budgets are 3072 and 2048: 75% and 50% of the original
4096 draws. They use prefixes of the same importance-sampling sequence,
seed 418, and the same mixture of 90% trial-basis importance and 10% uniform
probability. These are exploratory sampling budgets, not prescribed final
cell counts and not SVD tolerances.

Everything else is held fixed: the Euclidean 151-mode basis, HPROM-trained
master ANN, nine training trajectories, ten training times per trajectory,
moment blocks and normalization, positive-weight constraints, and the
L-BFGS-B settings (1500 iterations maximum, ftol=1e-12, gtol=1e-8, maxcor=20).
Every online solve uses predictor reuse, rank 3, one thread, and 500 steps.
The correction space is rebuilt at every step.

## Reusing the training moments

The existing matrix has the form A_parent = M_parent diag(w_parent), where
M contains the normalized cell contributions and w_parent contains the
initial importance-sampling weights. For a nested child support S:

    A_child = A_parent[:, positions(S)]
              diag(w_child[S] / w_parent[S]).

The target vector is unchanged. This is an exact change of column scaling;
it preserves every training moment and requires no new training trajectory.
Unit tests compare both the objective matrix and its gradient against
direct assembly. The cached model, basis, reference state, data metadata,
nine coefficient trajectories, matrix, and target are checked or hashed.

Fresh first-anchor moments are checked for compatibility with the Sherlock
cache. Float32 neural-network inference need not be bitwise portable across
numerical environments. The local differences were approximately 3e-6 to
5e-6 in relative Frobenius norm. The diagnostic compatibility limit is 1e-4;
it is separate from fitting and rule acceptance tolerances. Both new rules
use the original cached Sherlock matrix rather than mixing two assemblies.

## Selection fixed before reporting

Each candidate must pass all original fourteen held-out operator checks:

- Generalized Gram eigenvalues in [0.8, 1.2].
- Relative full-coordinate gradient error at most 0.05.
- True linearized residual after the sampled B3 update at most 1.05 times
  the corresponding full-residual B3 update.

Additionally, at each of the two validation parameters, the full-trajectory
coefficient error against the linear HPROM may increase by at most 5%
relative to the original 3532-cell B3 rule. This is a predeclared engineering
acceptance threshold, not a theorem about solution accuracy. A finite
trajectory alone is insufficient for acceptance.

The smallest retained support among passing candidates is selected. The
decision is written to selection.json before loading reporting trajectories.
Only the selected rule and the baseline are then compared on the four
reporting points. Reporting outcomes cannot change the selection in this
experiment. If neither candidate passes, the baseline is retained.

An optimizer iteration-limit exit is reported explicitly. Acceptance is
based on independent operator and trajectory checks, not on claiming the
NNLS optimum was reached.

## Execution and output

Use the project Python environment to run:

    python Project_YvonMaday/study_case2_b3_cubature_budget.py \
      --output Project_YvonMaday/Results_Paper/euclidean_b3_budget_local

Offline work uses two threads by default; online solves use one. Results
are separate from the original campaign and manuscript. Completed fits,
audits, and trajectories can be reused under the same saved protocol.
The script needs the downloaded positive_fit4096 matrix.npy and target.npy.

Outputs include protocol.json, cache_reproduction.json, each rule and fitting
diagnostics, validation_operator_audit.json, validation_comparison.json,
selection.json, and, if a smaller rule is accepted, reporting_comparison.csv.

## Local results

Both candidates passed the predeclared validation criteria. The 2048-draw
rule was selected before reporting evaluations; it retains 1506 cells.

| Draw budget | Positive cells | Stencil cells | Worst validation error ratio |
| --- | --- | --- | --- |
| 4096 (original) | 3532 | 9882 | 1.0000 |
| 3072 | 2472 | 7061 | 1.0241 |
| 2048 (selected) | 1506 | 4385 | 1.0474 |

The selected rule's held-out Gram eigenvalues ranged from 0.84463 to
1.08513; its maximum relative gradient error was 0.020805 and its maximum
true linearized residual ratio was 1.00613. Both fits reached their 1500
iteration limit: neither is presented as a certified NNLS optimum or a
minimum-cardinality cubature rule.

Errors below are percentages. State errors use the HDM; coefficient errors
use the corresponding linear HPROM trajectory. Means use only the first
three reporting points.

| Measure | Original 3532 cells | Selected 1506 cells |
| --- | --- | --- |
| Mean in-domain state error | 0.455282 | 0.456310 |
| Extrapolation state error | 0.844954 | 0.850651 |
| Mean in-domain coefficient error | 0.242924 | 0.257384 |
| Extrapolation coefficient error | 0.336350 | 0.421671 |
| Mean in-domain local online time (s) | 5.575285 | 2.533933 |

The local time reduction is 54.55%, with unchanged total iteration counts
at all four reporting points. Each time is a single measurement without
full-trajectory warmup, not a timing variability estimate and not a
Sherlock measurement. The selected rule reduces the positive support by
57.36% while barely changing state error. However, coefficient error
increases by 5.95% in-domain on average and 25.37% in extrapolation. The
5% validation allowance is not a guarantee for unseen parameters. The
smallest rule passes validation only narrowly; accuracy is not identical.

The result supports testing this rule on Sherlock, not reducing its support
further or replacing the established campaign without retaining a record.
No manuscript figures or production campaign results were changed.

### Independent 1024-draw stress test

A subsequent test used the first 1024 draws of the same sequence and was
stored separately in `euclidean_b3_budget1024_local`. Its fit retained 686
positive cells and required a 2026-cell augmented stencil. It failed every
predeclared operator criterion: the Gram eigenvalue interval was
[0.66711, 1.21542], the maximum relative gradient error was 0.07236, and
the maximum true linearized-residual ratio was 1.11750. The two validation
coefficient-error ratios were 1.49944 and 1.29773, well above 1.05.

The study therefore stopped before evaluating any reporting parameter. The
1024-draw rule is rejected. Among the tested rules, 2048 draws and 1506
positive cells remain the smallest accepted configuration.

## Sherlock timing comparison

With the existing campaign and active Python environment, execute:

    bash Project_YvonMaday/run_case2_b3_selected_rule_benchmark.sh

This performs eight trajectories: original and selected rules, four
reporting parameters each, one numerical thread, predictor reuse in both,
no retraining or fitting, and no full-trajectory warmups. It checks model,
basis and rule hashes before starting. Results go to the separate directory
`Results_Paper/euclidean_b3_budget_sherlock` under Project_YvonMaday; an
existing output directory is rejected rather than overwritten. Per-point
state and coefficient errors, iterations, times, qN arrays, environment
metadata, and a summary are preserved.
