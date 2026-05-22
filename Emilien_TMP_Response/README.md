# Emilien Temporary Response Bundle

This folder is isolated so it can be removed safely without affecting the repository.

## Contents

- `emilien_response_points.tex`: updated response notes + ready-to-send email draft + optional figure section.
- `emilien_response_points.pdf`: compiled PDF.
- `email_draft_to_emilien.txt`: plain-text email version.
- `scripts/plot_low_vs_high_modes.py`: script used to generate the illustrative coefficient plot.
- `figures/low_vs_high_modes_mu1_5.500_mu2_0.0150.png`: generated low-vs-high mode plot.
- `figures/low_vs_high_modes_summary.txt`: roughness metric summary.
- `scripts/plot_gpr_vs_true_coeffs.py`: script used to compare true vs predicted coefficients.
- `figures/gpr_vs_true_modes_1_4_and_101_104_mu1_4.56_mu2_0.019.png`: overlay of true vs predicted `q1-4` and `q101-104` at test `mu=(4.56, 0.019)`.
- `figures/gpr_vs_true_modes_1_4_and_101_104_summary.txt`: per-mode error summary for that comparison.
- `data_ref/*.npy`: local copied inputs used for plotting.

## Rebuild figure and PDF

```bash
python3 Emilien_TMP_Response/scripts/plot_low_vs_high_modes.py
python3 Emilien_TMP_Response/scripts/plot_gpr_vs_true_coeffs.py
cd Emilien_TMP_Response && pdflatex -interaction=nonstopmode -halt-on-error emilien_response_points.tex
```

## Remove everything

```bash
rm -rf Emilien_TMP_Response
```
