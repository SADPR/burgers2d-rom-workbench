# Euclidean HPROM campaign

This campaign produces the HPROM counterpart of the Euclidean PROM study under
`Results_Paper/euclidean_hprom_main`. It does not read models or cubature rules
from the previous MLSG-sensitive campaign.

## Mathematical consistency

The fixed Euclidean basis has 151 modes. The linear HPROM first constructs one
positive ECM rule from the nine available HDM training trajectories. Its nine
HPROM coefficient trajectories are the training targets for every learned
family; two additional linear-HPROM trajectories are used only for regression
validation. Thus the learned maps are HPROM-consistent rather than projections
of HDM snapshots or copies of PROM-generated coordinates.

The intrusive families are evaluated with separate rules:

- Linear HPROM: its fixed linear rule.
- HPROM-ANN Case 1: a rule trained with the Case-1 manifold and tangent.
- HPROM-ANN Case 2: a rule trained with the fixed parameter-time tail and the
  primary tangent.
- HPROM-ANN Case 3: a rule trained with the Case-3 manifold and tangent.
- HPROM-POD-AE: a rule trained with the decoder and decoder tangent.
- HPROM-ANN Case 2+B3: a positive rule fitted to the full-coordinate gradient,
  adaptive reduced Gram matrix, residual energy, and complete Euclidean
  trial-space mass moments.

The Case-2+B3 rule uses only the nine training trajectories during fitting. Its
operator accuracy and rollout are checked at the two held-out regression
parameters. The four reporting points remain inaccessible until
`selection.json` records that the validation-only audit passed.

POD-NN-ROM and POD-DL-ROM do not evaluate a residual and therefore do not use
ECM. They are included in the final HPROM table as HPROM-data-trained,
non-intrusive comparison methods, not as HPROMs.

## Timing protocol

All intrusive `online_solve_elapsed_s` values are obtained from one measured
solve per reporting point and exclude ECM construction, HDM evaluation,
plotting, and output. The two non-intrusive maps are timed after
checkpoint loading, with ten repeated predictions of the complete 501-step
coefficient trajectory. One fresh HDM solve is timed at each of the three
in-domain reporting points. The reported speed-up is

```
mean fresh-HDM time / mean online-model time
```

over the same three in-domain parameters. Offline training and rule
construction are excluded.

## Sherlock execution

Use the already active `myenv` inside an allocated compute node. The runner
never activates another environment and does not call `srun` or `sbatch`.

```bash
cd /scratch/users/sadpr/Code3Aug/burgers2d-rom-workbench
export PYTHON_BIN="$(command -v python)"

if bash Project_YvonMaday/run_euclidean_hprom_campaign.sh all; then
    printf '\nEuclidean HPROM campaign completed.\n'
else
    status=$?
    printf '\nCampaign stopped with code %s; the allocation remains open.\n' "$status"
fi
```

Every stage is resumable. To restart only one stage, replace `all` by one of
`prepare`, `train`, `rules`, `b3`, `online`, `timing`, or `summary`.

The final machine-readable and LaTeX tables are written under
`Results_Paper/euclidean_hprom_main/reporting`.
