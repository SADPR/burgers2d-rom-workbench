# Residual-space matrix weights for LSPG

This experiment replaces the scalar cubature weight of each selected cell by a fixed symmetric positive semidefinite matrix in **residual space**. It is a separate variant from the modal weights of BM-ECSW.

For the two-component Burgers residual, each matrix has size 2 by 2:

\[
\Phi_A(\boldsymbol q)=\frac12\sum_{e\in\widetilde{\mathcal E}}
\boldsymbol r_e(\boldsymbol q)^T\boldsymbol A_e\boldsymbol r_e(\boldsymbol q),
\qquad \boldsymbol A_e=\boldsymbol A_e^T\succeq0.
\]

The matrices are fixed during the online solution. Differentiation gives

\[
\nabla_{\boldsymbol q}\Phi_A=\sum_e\boldsymbol W_e^T\boldsymbol A_e\boldsymbol r_e,
\qquad
\widehat{\boldsymbol J}_A=\sum_e\boldsymbol W_e^T\boldsymbol A_e\boldsymbol W_e.
\]

The second expression is the Gauss–Newton matrix, not the exact Hessian. It is symmetric positive semidefinite. A common scalar objective exists for every state where the residual is differentiable; this property is not a condition checked only at training snapshots. Neither positive definiteness nor global nonlinear convergence follows automatically.

Setting \(\boldsymbol A_e=\xi_e\boldsymbol I_2\) recovers classical ECSW. A general \(\boldsymbol A_e\) has three independent entries. It cannot reproduce all the freedom of the 96 independent modal weights used by BM-ECSW.

## Training

Let \(\boldsymbol w_{u,e}\) and \(\boldsymbol w_{v,e}\) be the two rows of \(\boldsymbol W_e\), written as column vectors. For each training state,

\[
\boldsymbol W_e^T\boldsymbol A_e\boldsymbol r_e
= A_{e,11}\,r_{u,e}\boldsymbol w_{u,e}
+ A_{e,12}\,(r_{v,e}\boldsymbol w_{u,e}+r_{u,e}\boldsymbol w_{v,e})
+ A_{e,22}\,r_{v,e}\boldsymbol w_{v,e}.
\]

Stacking these vectors over the training pairs gives three matrices, \(\boldsymbol G_{11},\boldsymbol G_{12},\boldsymbol G_{22}\). The target remains the full LSPG projected residual at those pairs.

At fitting residual \(\boldsymbol h\), form the symmetric correlation matrix

\[
\boldsymbol S_e=
\begin{pmatrix}
\boldsymbol G_{11}(:,e)^T\boldsymbol h&\tfrac12\boldsymbol G_{12}(:,e)^T\boldsymbol h\\
\tfrac12\boldsymbol G_{12}(:,e)^T\boldsymbol h&\boldsymbol G_{22}(:,e)^T\boldsymbol h
\end{pmatrix}.
\]

Adding \(t\boldsymbol v\boldsymbol v^T\) at that cell has initial objective derivative \(-\boldsymbol v^T\boldsymbol S_e\boldsymbol v\). The largest eigenvalue selects the greatest initial decrease among unit-trace rank-one PSD directions. This comparison selects a new cell; its subsequent fitted matrix is not restricted to that direction or to rank one.

At each joint refit, all three symmetric entries of **every selected matrix** are optimized against the original target:

\[
\min_{\{\boldsymbol A_e\succeq0\}}
\frac12\left\|
\boldsymbol G_{11}\boldsymbol a_{11}
+\boldsymbol G_{12}\boldsymbol a_{12}
+\boldsymbol G_{22}\boldsymbol a_{22}
-\boldsymbol b\right\|_2^2.
\]

Matrices on unselected cells remain zero. After each new cell is selected, the complete selected set is jointly refitted. Existing matrices may change their magnitudes, directions and rank. The process starts with no cells and stops at relative projected-residual error \(10^{-5}\), evaluated on the original training rows. There is no prescribed cell budget or SVD truncation.

For the two-component blocks, the PSD constraint has an exact second-order cone representation:

\[
\boldsymbol A_e=
\begin{pmatrix}t_e+z_e&b_e\\b_e&t_e-z_e\end{pmatrix}
\succeq0
\quad\Longleftrightarrow\quad
 t_e\ge\sqrt{z_e^2+b_e^2}.
\]

Clarabel solves these coupled convex refits, leaving all three coordinates free. Its numerical status, feasibility residuals and duality gaps are recorded separately from the training tolerance. The fit is scaled uniformly per cell and solved as a correction to the preceding feasible matrices; neither operation changes the optimization problem. The quadratic products are updated when a cell is added, and the error is checked directly using the original rows. Only roundoff-sized cone violations may be projected away, with the correction recorded. If the solver reports insufficient numerical progress, a feasible iterate can become the coordinate origin for a retry of the same problem. A final factorization fallback uses QDLDL instead of Faer. The objective, cone constraints and numerical tolerances remain unchanged, and a failed solver status is never accepted as a successful refit. All attempts and their costs are recorded.

This remains a greedy cell-selection procedure, not an exact solution of the minimum-cardinality problem. Positive semidefiniteness and the common objective follow from the matrix construction; fewer sampled cells, lower error and faster execution remain empirical questions.

The initial fixed-direction NNLS prototype was cancelled before reaching the training tolerance. It is archived under `Results/ECSW_BM_PSD_metric_20261008/`; it produced no final rule or online comparison. Its limitation was an offline fitting choice: fixed directions are not required to preserve the common objective. The current implementation uses full PSD matrix refits.

## Online solution

For each sampled block, factor \(\boldsymbol A_e=\boldsymbol B_e^T\boldsymbol B_e\). The existing Gauss–Newton least-squares solve then uses \(\boldsymbol B_e\boldsymbol r_e\) and \(\boldsymbol B_e\boldsymbol W_e\). An eigenvalue factorization accommodates rank-deficient PSD matrices. Both components and their Jacobian rows are transformed with the same factor.

The state is evaluated on the augmented mesh and residuals on the sampled cells, as in the existing global linear HPROM. No residual Hessians are required. The stopping settings and least-squares solver are those of classical ECSW.

## Reproduce

From the repository root:

```bash
.venv/bin/python -m pip install -r requirements-residual-metric.txt
.venv/bin/python -u run_residual_metric_experiment.py --threads 10 --tolerance 1e-5
```

The runner creates a new dated directory under `Results/`. It rebuilds the training matrices from the cached HDM trajectories, trains from zero, and runs the three test parameters for classical ECSW, BM-ECSW and the new matrix-weighted rule. Existing POD models and cubature weights are only read.

Configuration: 96 modes, 225 projected consecutive state pairs, nine training parameters, sampling seed 42, and 500 online time steps. Each online measurement runs in a fresh process. Timings are single-run estimates; they do not establish repeatability.

The assembly batches the same least-squares projections used by the existing trainer and vectorizes local contributions. Its timing therefore cannot be interpreted as a like-for-like comparison with the older assembly implementation. Assembly and greedy selection times are reported separately.

The production procedure refits all selected matrices after every added cell. Column-contiguous float64 training matrices avoid strided column reads. `checkpoint.npz` preserves the selected indices, cell scales and fitted cone coordinates after every refit; the production run still starts from zero. `selection_history.jsonl` stores progress after every refit, and `status.json` shows the latest completed refit. The thread limit is ten so that the concurrent AERO-F work retains CPU capacity.

The saved rule is `residual_metric_weights.npy`; `training.json` records tolerance attainment, sampled and augmented cells, eigenvalues, conic refit diagnostics and timing. `online_results.csv` and `summary.md` report the online comparison. Temporary training matrices are removed after a successful comparison; `--keep-training` retains them.

This experiment does not impose an integrability penalty on the existing BM weights, and it makes no claim about conservation of mechanical potential energy.

Solver interface: [Clarabel Python documentation](https://clarabel.org/stable/python/getting_started_py/).

The discontinued batch-of-five experiment is archived separately. Batching changes the selection path and was removed from the production procedure. Column-contiguous float64 storage and incremental Gram products are retained: they avoid repeated work without modifying the optimization problem or the per-cell selection/refitting sequence.
