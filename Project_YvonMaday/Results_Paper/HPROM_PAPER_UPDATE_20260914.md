# HPROM-centred Euclidean manuscript checkpoint

## Scope

The working paper remains `manuscript_paper_euclidean.tex`. Its previous
PROM-centred source is preserved as
`manuscript_paper_euclidean_prom_checkpoint_20260914.tex`.
The historical `manuscript_paper.tex` was not replaced.

No solver, training procedure, checkpoint, cubature weights, or raw trajectory
was modified or rerun in this reporting update.

## Main-text changes

- Reframed the abstract and introduction around hyperreduced deployment.
- Retained the detailed PROM/LSPG and model-specific ECM theory and notation.
- Added the weighted sampled Schur/Krylov derivation: direction construction
  and nonlinear correction use the same square-root-weighted residual and
  Jacobian. Global POD orthogonality does not imply local weighted orthogonality.
- Explained positive dynamic and complete-mass moment fitting separately from
  conventional SVD-compressed ECM. The 4096 parameter counts candidate draws,
  not retained cells or an SVD tolerance.
- Replaced main PROM accuracy/enrichment comparisons with the baseline and
  nested 9+8, 9+12, 9+18 HPROM campaigns, including B3 at every budget.
- Used one model/parameter order throughout accuracy tables and figures.
- Added baseline HPROM cost measurements and the separate validated B3 support
  comparison with predictor reuse in both variants.
- Moved existing full-residual rank/equal-dimensional and perturbation controls
  to explicitly identified PROM appendices. No HPROM equivalents were fabricated.
- Regenerated all main HPROM sampling, cut-plane, coordinate-history, heat-map,
  and enrichment assets with LaTeX text, consistent model colours, solid model
  curves, dashed intrusive/direct separators, and shared unclipped limits.

## Audit and interpretation

`tables/euclidean_hprom/metrics.csv` contains the unrounded recomputed errors.
`tables/euclidean_hprom/audit.json` records source SHA-256 fingerprints,
references, and figure ranges.

The generator verifies complete 151-by-501 reporting coordinates, finite
positive weights, 17/21/27 nested HPROM training trajectories, 20 HPROM-trained
checkpoints with HPROM external validation, sample counts, unchanged activation
and network size, and four accepted B3 rules. State errors are recomputed using
the Euclidean orthogonal decomposition and checked against saved summaries;
B3 complete-coordinate errors are also checked against their summaries.
Every main coefficient error uses the frozen linear HPROM, never linear PROM.

Important qualifications retained in the paper:

- Verification is a training parameter; only the two off-grid reporting points
  are unseen in-domain parameters. Regression validation is a separate pair.
- B3 uses the known full-linear-space initial state, while conventional learned
  runners use their own ANN/decoder initial representation. Main differences
  do not isolate correction orientation alone. The matched control is PROM-only.
- B3 moment fits use nine baseline HPROM teachers at every budget, with the
  corresponding master checkpoint. Enriched conventional rules are also rebuilt.
- Operator acceptance compares generalized sampled/full **Jacobian-action**
  Grams. Its LS ratio compares sampled B3 against full-residual B3 linearized
  updates, both scored in full residual norm, not an unrestricted 151-mode solve.
- Enrichment results use 4096-draw rules. The 2048-draw/1506-positive-cell rule
  belongs only to a separately validated baseline optimization.
- Baseline timing uses one Sherlock node: HDM and linear HPROM have 24 numerical
  threads, learned deployments one. Single intrusive measurements provide no
  timing variability estimate. Enrichment-node timings are not mixed into them.
- Direct inference uses ten repeated loaded-network coefficient predictions;
  it excludes state reconstruction/loading/I/O. Its HDM/time ratio is not an
  end-to-end simulator speedup.
- 9+8 improves in-domain regression but worsens extrapolation for conventional
  Case 2 and POD-NN-ROM. The coverage explanation is plausible, not a causal
  isolation of one additional LHS point.

## Regenerate and Verify

From the repository root, using the existing local environment:

```bash
MPLCONFIGDIR=/tmp/hprom-paper-mpl .venv/bin/python \
  Project_YvonMaday/Results_Paper/generate_euclidean_hprom_paper_assets.py

.venv/bin/python -m pytest -q \
  tests/test_hprom_paper_assets.py \
  tests/test_case2_hyperreduction.py \
  tests/test_case2_residual_correction.py \
  tests/test_ecm_consistency.py \
  tests/test_case2_cubature_budget.py

cd Project_YvonMaday/Results_Paper
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript_paper_euclidean.tex
```

Wait for asset generation to finish before compiling or inspecting the PDF.
