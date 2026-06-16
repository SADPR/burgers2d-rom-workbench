# Burgers presentation animations

Run:

```bash
MPLCONFIGDIR=/tmp/mplconfig PYTHONDONTWRITEBYTECODE=1 \
python3 generate_burgers_presentation_assets.py
```

The generator reads the current `mlspg_hprom_main` and
`mlspg_hprom_enrichment` campaigns and writes presentation-ready assets to
`outputs/`. Large state trajectories are memory mapped; the script does not
copy them.

The comparison GIFs deliberately show one nonlinear model at a time. This
keeps the HDM reference and the two centerline cuts legible while still
covering every model family.
