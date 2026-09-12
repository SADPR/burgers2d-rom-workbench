# Residual-adaptive Case 2 with positive empirical cubature

This experiment uses the baseline Euclidean 151-mode basis and its existing
master ANN, trained on nine parameters with two separate regression-validation
parameters. It adds no trajectories to the ANN training set. It investigates
whether the successful full-residual B3 correction survives residual sampling.

## Weighted objective and direction construction

Write the basis as `V_tot = [V, Vbar]`, with 10 primary and 141 secondary
coordinates. Let `S` restrict residual rows to the selected cells, including
both velocity components, and let `L` restrict states to the corresponding
upstream stencil. Let `D = diag(sqrt(xi), sqrt(xi))` for positive cell weights.
The sampled residual and its derivative are

```
r_xi(u) = D S r(u)
J_xi(u) = D S J(u) L^T.
```

The implementation evaluates these expressions directly on the stencil; it
does not first assemble a full residual or full Jacobian. It keeps the original
coefficient coordinates: `V_L = L V` and `Vbar_L = L Vbar` are not
reorthonormalized after restriction.

At each time-step predictor, form

```
A_xi = J_xi V_L
Q_xi = orth(A_xi)
P_xi = I - Q_xi Q_xi^T
C_xi = P_xi J_xi Vbar_L
g_xi = C_xi^T P_xi r_xi
H_xi = C_xi^T C_xi
range(B_r) = span(g_xi, H_xi g_xi, ..., H_xi^(r-1) g_xi).
```

These are the exact primary-elimination and Krylov expressions for the sampled
least-squares objective. They are approximations to their full-residual
counterparts. The code applies `H_xi` by matrix-vector products instead of
forming the full 141-column secondary Jacobian online.

Freeze `B_r` for the time step, and solve

```
min_(q,a) ||r_xi(u_ref + V q + Vbar [M(mu,t) + B_r a])||_2^2.
```

There are 13 unknowns for r=3. The tangent is `[V_L, Vbar_L B_r]`; there is no
derivative of B_r inside this frozen-space solve. Both the direction construction
and the final solve use sqrt(xi). Applying xi to residual rows would instead
minimize an objective with squared cubature weights.

At r=0 this reduces to the sampled Case-2 solve. Sampling every cell with weight
one recovers the full-residual algorithm. Krylov breakdown can produce fewer
than r directions. Rank deficiency of the sampled primary Jacobian is an error,
not a reason to silently change the problem.

## Offline rule

A Case-2 rule trained only on the primary projected residual does not by itself
control the secondary sensitivity needed by B3. This prototype trains a new rule
on three groups of cell contributions at each baseline parameter-time anchor:

1. `(J V_tot)_e^T r_e`, keeping all 151 coordinate-gradient components.
2. `(J T_r)_e^T (J T_r)_e`, keeping the upper triangle of the symmetric
   Gauss-Newton matrix for the 10 primary and three full-residual B3 directions.
3. `r_e^T r_e`, the objective contribution.

The predictor uses the primary coordinates and previous state from the existing
linear PROM trajectory, and the current secondary coordinates from the ANN.
Training times are explicit in `rule_s*/config.json`; the previous column is
always the immediately preceding time step. This is an offline choice of
anchors, not teacher forcing during deployment. Online trajectories use their
own previous state, including the previous secondary correction.

Each group at each anchor is divided by its Frobenius norm. The symmetric
off-diagonal Gram entries carry sqrt(2), preserving the symmetric Frobenius norm.
A seeded streaming randomized range approximation and a small eigendecomposition
compress these moments without storing their full concatenation. Positive ECM
integrates the retained entity basis and a constant. Full uncompressed training
moment errors are recomputed after selection.

This compression and the finite set of adaptive tangent anchors do not prove a
uniform embedding of every possible 151-dimensional Jacobian range. In
particular, accurately fitting a few training residuals is insufficient to
guarantee accurate Krylov directions. The separate validation audit computes
generalized eigenvalues of the sampled versus full `V_tot^T J^T J V_tot`,
gradient errors, and the true full linearized residual after each correction.
The two held-out validation trajectories then test accumulated time-integration
error. The two off-grid reporting parameters and the extrapolatory parameter
are not used to train the rule. As in the paper, the verification parameter is
the center of the original training grid and is therefore already one of the
nine training parameters.

## Successful extension of the cubature construction

The first 257-cell compressed ECM rule failed validation. Its sampled
151-coordinate Jacobian Gram matrix had nearly invisible directions. Increasing
the information retained about the full trial space resolved this failure.

First, draw a candidate support using the complete trial-basis cell leverage:

```
ell_e = ||V_tot[e,:]||_2^2 + ||V_tot[Ncells+e,:]||_2^2
p_e = 0.9 ell_e / sum(ell) + 0.1 / Ncells
xi_e^(0) = count_e / (4096 p_e).
```

These are 4096 draws with replacement, giving 3933 distinct cells. The initial
positive importance quadrature is a control experiment; it is not the greedy
ECM selection algorithm. Its input is only the existing trial basis.

On this support solve a convex nonnegative moment-fitting problem. For each of
the three dynamic moment blocks at each of the 90 training anchors, let
`C_b xi` be its cubature sum and `d_b = C_b 1` its exact sum. Let
`G(xi) = sum_e xi_e V_tot,e^T V_tot,e`. Minimize

```
sum_b ||C_b xi - d_b||_2^2 / ||d_b||_2^2
    + (270 / 151) ||G(xi) - I_151||_F^2,
subject to xi >= 0 on the candidate support and xi = 0 elsewhere.
```

The mass term has the same total weight as the 270 normalized dynamic blocks;
its normalization is `||I_151||_F^2 = 151`. A numerical floor of 1e-12 is applied
to dynamic block target norms before division. The variables supplied to the
optimizer are `xi / xi^(0)`, a positive diagonal change of variables, not an
additional penalty toward the initial weights.

This remains a positive empirical quadrature built by nonnegative least squares,
but uses an importance-selected candidate support and a full trial-space Gram
constraint in addition to projected-residual moments. It must not be described
as an unmodified application of the original greedy ECM algorithm.

The implemented L-BFGS-B fit uses exact objective gradients and a budget of
1500 iterations. It reached that budget rather than satisfying its convergence
criterion; the projected gradient infinity norm was about 2.8e-5. The final
weights remain feasible, and the following independently measured properties
justify this experimental rule:

- 3564 positive cell weights, requiring 9962 stencil cells out of 62500.
- Trial-space mass eigenvalues between 0.97245 and 1.02290.
- On 14 held-out parameter-time probes, Jacobian Gram generalized eigenvalues
  between 0.97090 and 1.02394; worst full-coordinate gradient error 0.6281%.
- The true full linearized residual after the sampled B3 correction is at most
  1.00365 times its full-residual B3 counterpart on those probes.
- Complete 500-step validation coefficient errors: HPROM+B3 0.2095% and
  0.2665%; PROM+B3 0.2068% and 0.2549%.

The rule was frozen using these validation results before running the two
off-grid and extrapolation reporting trajectories. Those results are recorded
in `Results_Paper/euclidean_case2_hyperreduction/REPORT.md` and its CSV/figure.
The state-error in-domain means are 0.4481% for HPROM+B3 and 0.4468% for
PROM+B3; extrapolation errors are 0.8486% and 0.8503%, respectively. These
observations support the extension for this campaign, not a uniform guarantee
over parameter space.

## Local execution

From the repository root, using the existing local environment:

```bash
.venv/bin/python -m pytest -q tests/test_case2_hyperreduction.py tests/test_case2_residual_correction.py
.venv/bin/python -u Project_YvonMaday/run_case2_hyperreduction.py train
.venv/bin/python -u Project_YvonMaday/audit_case2_hyperreduction.py
.venv/bin/python -u Project_YvonMaday/run_case2_hyperreduction.py validation
```

To reproduce the successful rule and its validation:

```bash
.venv/bin/python Project_YvonMaday/build_case2_leverage_rule.py --draws 1024 4096
.venv/bin/python -u Project_YvonMaday/fit_case2_positive_cubature.py
.venv/bin/python -u Project_YvonMaday/audit_case2_hyperreduction.py \
  --rule-file Project_YvonMaday/Results_Paper/euclidean_case2_hyperreduction/positive_fit4096/weights.npy
.venv/bin/python -u Project_YvonMaday/run_case2_hyperreduction.py validation \
  --rule-file Project_YvonMaday/Results_Paper/euclidean_case2_hyperreduction/positive_fit4096/weights.npy
```

The output root is
`Project_YvonMaday/Results_Paper/euclidean_case2_hyperreduction`.
Completed stages are checked against configuration and input fingerprints before
being reused. Rules with different compression ranks have different directories.
Existing paper results and model checkpoints are not overwritten.

Local timings are exploratory single-run measurements. Definitive comparisons
will require the same Sherlock node and thread configuration for all methods.
The HPROM clock includes ANN inference, stencil-offset construction, all B3
updates, and time integration. Mesh/basis restriction and the projection of the
known initial state are setup operations. Full-state error reconstruction is
postprocessing and lies outside the online clock.

## Sherlock deployment

Upload the small code/rule package from the local repository. The existing
Euclidean basis, master ANN and reference trajectories are already expected in
the Sherlock repository and are verified by SHA256 before starting.

```bash
cd /home/sares/burgers2d-rom-workbench
rsync -avz --files-from=Project_YvonMaday/case2_hyper_upload.txt ./ \
  sadpr@login.sherlock.stanford.edu:/scratch/users/sadpr/Code3Aug/burgers2d-rom-workbench/
```

Inside the allocated compute node, with the existing `myenv` active:

```bash
if cd /scratch/users/sadpr/Code3Aug/burgers2d-rom-workbench; then
    export PYTHON_BIN="$(command -v python)"
    if bash Project_YvonMaday/run_case2_hyper_sherlock.sh; then
        printf '\nLa comparacion termino correctamente.\n'
    else
        STATUS=$?
        printf '\nLa comparacion fallo con codigo %s; la sesion sigue abierta.\n' "$STATUS"
    fi
fi
```

This runs eight 500-step trajectories sequentially: HPROM Case 2 and HPROM
Case 2+B3, each at the four reporting parameters. It uses 24 numerical threads,
one measurement per trajectory, and no full-trajectory warmups. It does not
train a network or regenerate the rule. The shell is executed as a child
process with `bash`; do not source it into the allocation's interactive shell.

The script prints the newly created result directory, records the machine,
environment and CPU affinity, and writes `comparison.csv`, `comparison.txt`,
the eight coefficient trajectories and their summaries. The whole printed
directory is the result to download afterward. These measurements compare the
two HPROM variants on one machine. A speed-up relative to the HDM additionally
requires a matched HDM timing; no such number is inferred from the local tests.
