# Same-node Euclidean Case-2 benchmark

## Existing allocated node (preferred)

With your usual PROM Python environment active, enter the already allocated
compute node and run from the isolated experiment root:

```bash
bash Project_YvonMaday/run_case2_sherlock_benchmark.sh
```

This does not submit a job or call srun. It checks inputs and solver imports,
and runs the tests only if pytest is installed in the active Python environment,
then launches 16 trajectories sequentially using 24 numerical threads.
The allocation should have at least 24 CPUs, 16 GB memory, and enough remaining
walltime. Its remaining walltime is not extended by this script. Avoid running
other workloads concurrently in this allocation while measuring timings.
Unlike the optional batch launcher, the shell launcher does not set CPU affinity;
it inherits the affinity of the allocated shell and records it in environment.json.
Results and benchmark.log are written to a new directory whose name begins
with euclidean_case2_sherlock_JOBID_, printed when the benchmark starts.
The shell must remain alive until completion; use your existing persistent
terminal session if needed. No original results or checkpoints are replaced.

This is the selected residual-adaptive rank-3 online method, not ANN retraining
and not the previous offline response-penalty experiment. No HPROM is run.
The underlying numerical runners are unchanged from the local experiments.

## Existing training and validation data

Reuse the ANN trained on the original nine Euclidean linear-PROM trajectories
with the existing two validation parameters. Rank 3 was selected in the local
validation experiments. No training, validation rollout, or new parameter-data
generation is required for this deployment. The correction directions are
constructed online from the current residual and Jacobian. The four reporting
trajectories per method only measure accuracy and elapsed time.

## Protocol

- One existing allocation, one node, 24 BLAS/PyTorch numerical threads per solve.
- Four methods: Case 2, adaptive Case 2 with rank 3, Case 1, and Case 3.
  All four use the common projection of the prescribed initial state.
- Four manuscript parameters, 500 steps per trajectory.
- One measured trajectory per method/point, no full-trajectory warmups.
  Total: 16 trajectories. This replaces the earlier 264-trajectory protocol.
- Sequential runs, shuffled method order with a fixed seed.
- Timings include initialization, ANN evaluation and adaptive-space construction.
  File loading, accuracy scoring and output-array saving are excluded.
- One elapsed time and one state error are reported per point. Standard
  deviation is blank, since one measurement cannot estimate timing variability.
  The CSV retains its mean/min/max fields; with one sample these are identical.
  The in-domain mean averages the first three parameters, not repeated timings.
  Runtime startup effects are included; there is no claim of warmed-up timing.
  Setting 24 threads does not guarantee that every operation uses all 24 CPUs,
  or that 24 is the fastest thread count for each method.
- This does not claim an HDM speed-up: a matched HDM timing is not measured here.
- Same node does not guarantee no interference from other jobs. Slurm allocates
  cores and the task is CPU-bound; the job does not reserve the entire node.
- A fresh output directory is created for each Slurm job. Resuming/mixing timing
  samples from a different job, machine or implementation is rejected.
- Input SHA-256 checks require the same basis, reference state, ANN checkpoints,
  reference trajectories and reporting HDM arrays as the local experiment.

## Deployment

Upload using `case2_sherlock_upload.txt` as an rsync files list into a NEW
subdirectory `Code3Aug/burgers2d-rom-workbench/_case2_krylov3`. It includes a source snapshot, not data,
local result arrays, or the local virtual environment. Never use rsync --delete.

In that isolated directory, create `Project_YvonMaday/Results_Paper` and symlink
its `MetricStudy` and `euclidean_prom_main` entries to the corresponding original
Code3Aug directories. Symlink `Project_YvonMaday/Results` to the original HDM
directory. The runners only read those inputs. New benchmark outputs remain in
the isolated experiment, not in the original manuscript campaign.

Use the existing Sherlock PROM Python environment. It needs numpy, scipy,
torch (including torch.func/functorch), matplotlib, scikit-learn, threadpoolctl,
and optionally pytest for the preflight tests. Missing pytest is explicitly
reported as skipped, not as a successful remote test run. The shell launcher
never installs packages or activates another environment. Do not upload the local .venv.

From the isolated root inside the allocated node, with myenv activated:

```bash
set +e
export PYTHON_BIN="$(command -v python)"
if bash Project_YvonMaday/run_case2_sherlock_benchmark.sh; then
    printf '\nBenchmark completed.\n'
else
    printf '\nBenchmark failed; the interactive shell remains open.\n'
fi
```

The optional batch file also defaults to 16 trajectories and 24 threads,
but is not used for the interactive deployment above.

Results: `_case2_krylov3/Project_YvonMaday/Results_Paper/euclidean_case2_sherlock_JOBID_SUFFIX/`.
Read `timing_summary.txt`, `timing_summary.csv`, and `environment.json`.
Individual summaries and reduced trajectories remain available by repetition.
The text summary says COMPLETE only after all 16 trajectories have finished.
The manifest records the method list, warmup count, repetition count, and total
trajectory count. Optional Python CLI flags permit explicitly requested larger
studies; the shell launcher explicitly requests the 16-trajectory protocol.
