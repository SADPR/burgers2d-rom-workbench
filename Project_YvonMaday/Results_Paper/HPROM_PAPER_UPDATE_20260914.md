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
  conventional SVD-compressed ECM. The 2048 parameter counts candidate draws,
  not retained cells or an SVD tolerance.
- Replaced main PROM accuracy/enrichment comparisons with the baseline and
  nested 9+8, 9+12, 9+18 HPROM campaigns, including B3 at every budget.
- Used one model/parameter order throughout accuracy tables and figures.
- Added baseline HPROM cost measurements and the separate validated B3 support
  comparison with predictor reuse in both variants.
- After downloading the corrected B3 enrichment runs, replaced all main B3
  sources with the validated 2048-draw rules. Baseline uses the already measured
  frozen selected deployment; lhs8/lhs12/lhs18 use their independently refitted
  rules. Kept 4096 only as the matched support-ablation comparator.
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
- Every main B3 result now uses 2048 draws and predictor reuse. Positive cells
  are 1506/1533/1534/1512 across 9/9+8/9+12/9+18. Operator bounds are unchanged.
- State errors are nearly unchanged from 4096, but the enriched complete-
  coefficient means increase to 0.099/0.082/0.080 percent. The update explicitly
  reports this support-versus-fidelity tradeoff instead of claiming identical
  trajectories or a universal error floor.
- Baseline timing uses one Sherlock node: HDM and linear HPROM have 24 numerical
  threads, learned deployments one. Single intrusive measurements provide no
  timing variability estimate. Enrichment-node timings are not mixed into them.
- Direct inference uses ten repeated loaded-network coefficient predictions;
  it excludes state reconstruction/loading/I/O. Its HDM/time ratio is not an
  end-to-end simulator speedup.
- 9+8 improves in-domain regression but worsens extrapolation for conventional
  Case 2 and POD-NN-ROM. The coverage explanation is plausible, not a causal
  isolation of one additional LHS point.

## Corrected B3 Summary (2048 Draws)

| Data | Positive cells | In-domain state mean (%) | Extrapolation state (%) | In-domain coefficient mean (%) |
| --- | ---: | ---: | ---: | ---: |
| 9 | 1506 | 0.456310 | 0.850651 | 0.257384 |
| 9+8 | 1533 | 0.442130 | 0.876287 | 0.098761 |
| 9+12 | 1534 | 0.444235 | 0.849756 | 0.082494 |
| 9+18 | 1512 | 0.444004 | 0.847998 | 0.079848 |

Baseline B3 mean time is 8.433171 s, with an 87.36 HDM/time ratio against
736.729178 s. It is faster than Case 1 (21.098 s) and Case 3 (18.856 s),
but slower than uncorrected Case 2 (5.682 s). These are the existing matched
single-measurement Sherlock data, not new local timings. No training or
solver was run while updating the paper.

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
