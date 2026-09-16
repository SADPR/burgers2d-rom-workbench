# Deploy the accepted ECM rule on Sherlock

The local baseline experiment accepted an ECM rule with 839 residual cells
(2318 stencil cells), compared with 1506 (4385) for the previous rule.
The following benchmark compares those two frozen rules with the same ANN,
solver, known initial state and predictor reuse. It advances eight trajectories:
two rules at each of four parameters, once, sequentially, with one numerical
thread. It performs no ANN training, cubature fitting, full HDM solve, or
full-trajectory warmup. It uses the active `myenv` environment.

The time ratio compares the two B3 deployments; it is not an HDM speed-up.
Any later HDM speed-up table needs its appropriately matched HDM reference.
These commands do not run the enrichment cases: each enriched map needs its
own ECM rule and validation.

## Upload from the local computer

```bash
cd /home/sares/burgers2d-rom-workbench

rsync -avzR \
  burgers/{case2_hyperreduction,case2_residual_correction,core,config,ecsw_utils}.py \
  Project_YvonMaday/{run_case2_hyperreduction,run_case2_local_corrections,run_prom_ann_case_2,study_case2_b3_cubature_budget,benchmark_case2_b3_ecm}.py \
  Project_YvonMaday/run_case2_b3_ecm_benchmark.sh \
  Project_YvonMaday/Results_Paper/euclidean_b3_ecm_deployment/ \
  sadpr@login.sherlock.stanford.edu:/scratch/users/sadpr/Code3Aug/burgers2d-rom-workbench/
```

This transfers the small deployment bundle, including both weight arrays.
The 16.7 GB offline moment matrix is not transferred. Existing baseline
checkpoints, the POD basis, linear references and HDM scoring snapshots must
already exist in their usual Sherlock locations; the runner checks them.

## Run inside the allocated node with myenv active

Paste the following block into the allocation. The driver is a child bash
process, and a failure is caught by `if` without explicitly exiting the shell.

```bash
if cd /scratch/users/sadpr/Code3Aug/burgers2d-rom-workbench; then
    export PYTHON_BIN="$(command -v python)"
    if bash Project_YvonMaday/run_case2_b3_ecm_benchmark.sh; then
        printf '\nComparacion ECM terminada correctamente.\n'
    else
        STATUS=$?
        printf '\nLa comparacion fallo con codigo %s. La sesion sigue abierta.\n' "$STATUS"
    fi
else
    printf '\nNo se encontro el directorio. La sesion sigue abierta.\n'
fi
```

It prints a new result directory beginning with
`Project_YvonMaday/Results_Paper/euclidean_b3_ecm_sherlock_`.
Download that entire directory after completion. It contains `summary.txt`,
`comparison.csv`, `comparison.json`, environment/protocol records and all eight
coordinate trajectories. No existing campaign output is overwritten.

To resume an interrupted benchmark on the same host, set
`B3_ECM_BENCHMARK_OUTPUT` to its printed absolute result directory before
rerunning the block. Completed trajectories are reused. The runner refuses
to mix hosts or changed rules/inputs in that same directory.
