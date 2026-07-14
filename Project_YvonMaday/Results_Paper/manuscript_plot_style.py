"""Shared visual conventions for PROM and HPROM manuscript diagnostics."""

from __future__ import annotations

# Established manuscript palette. A model keeps this color in every PROM and
# HPROM figure: state overlays, coefficient curves, and coefficient heatmaps.
# Color alone identifies the model: dense multi-model coefficient diagnostics
# remain readable only when every model trajectory is solid.
HDM_COLOR = "#111111"
METHOD_COLORS = {
    "linear": "#D62728",
    "case1": "#1F77B4",
    "case2_n10": "#17BECF",
    "case2_n20": "#8C564B",
    "case3": "#2CA02C",
    "podae": "#9467BD",
    "podnn": "#FF7F0E",
    "poddl": "#E377C2",
}

# The Case--2 dimension sweep is not a collection of different methods, but
# these manuscript hues make it visually compatible with the model-comparison
# figures. In particular, n=0 is the direct POD-NN map, n=10 is the standard
# Case--2 model, and n=20 is its larger counterpart.
CASE2_SWEEP_COLORS = {
    0: METHOD_COLORS["podnn"],
    3: METHOD_COLORS["case1"],
    5: METHOD_COLORS["case3"],
    10: METHOD_COLORS["case2_n10"],
    20: METHOD_COLORS["case2_n20"],
    30: METHOD_COLORS["podae"],
    50: METHOD_COLORS["poddl"],
    100: METHOD_COLORS["linear"],
}

METHOD_LINE_STYLES = {
    "linear": "-",
    "case1": "-",
    "case2_n10": "-",
    "case2_n20": "-",
    "case3": "-",
    "podae": "-",
    "podnn": "-",
    "poddl": "-",
}

# These limits are shared by the PROM and HPROM baseline figures.  They were
# selected from the union of all four evaluation trajectories, not per panel.
STATE_UX_YLIM = (2.0, 6.8)
# Both spatial cut-plane directions must accommodate the lower branch visible
# in $u_x(x_{\mathrm{mid}},y)$; these limits are shared by PROM and HPROM.
STATE_CUTPLANE_YLIM = (0.0, 6.8)
# The online enriched models can attain absolute coefficient-trajectory
# errors below 0.5.  Retain a common logarithmic range for PROM and HPROM
# diagnostics without clipping those valid low-error curves.
COEFF_ABS_YLIM = (1.0e-2, 2.0e3)
COEFF_REL_PERCENT_YLIM = (1.0e-2, 5.0e2)
COEFF_ABS_HEAT_VMAX = 10.0
COEFF_REL_PERCENT_HEAT_VMAX = 20.0
