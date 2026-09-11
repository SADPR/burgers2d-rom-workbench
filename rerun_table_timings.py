#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Rebuild the online-performance table with a single code version.

The published table mixes global rows measured in April/May with local rows
measured in September, and two of the global rows can no longer be reproduced
from the artifacts on disk. This driver re-derives every row that the current
code affects and re-measures all of them.

What it does *not* touch: the POD bases, the quadratic coefficient matrix, the
trained GPR models, the clustering and the HDM snapshots. Those are kept as
they are -- the randomized SVD in this repository is not seeded, so rebuilding
a basis would silently change every number downstream.

Isolation
---------
Every row gets its own working directory under --work, with the model trees
reachable through symlinks and a private `Results/`. The runners write through
a relative `results_dir="Results"` and several of them save ECSW weights next
to the model, so a shared directory is exactly how mismatched figures and
numbers crept into the previous table: within a family the N_c=3/5/8 runs all
write `local_hprom_summary_mu1_4.56_mu2_0.019.txt`. One directory per row
removes that entirely.

`POD/` is materialised as a real directory holding symlinks, because
`run_hprom.py` derives its ECSW weight path from `pod_dir` and would otherwise
write the rebuilt weights back into the repository.
"""

import argparse
import csv
import io
import os
import re
import statistics
import sys
import time
import warnings

warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.abspath(__file__))
POINTS = [(4.56, 0.019), (4.75, 0.020), (5.19, 0.026)]

# Model trees the runners reach through relative paths. Read-only, so a
# directory symlink is enough.
LINK_DIRS = ["Quadratic", "POD-GPR",
             "LocalPODSweep", "LocalQuadraticSweep", "LocalPOD-GPRSweep"]

# label, runner module, summary prefix, extra main() kwargs.
#
# Each local candidate was matched to its table row by N_e *and* by the
# reported avg/max error. N_e alone is ambiguous: in the POD-GPR sweep it is
# set by n_primary, so nc5/nprimary6 and nc8/nprimary6 both give N_e = 541.
ROWS = [
    ("HPROM (global)", "run_hprom", "hprom", {"num_modes": 96}),
    ("HQPROM (global)", "run_hqprom", "hqprom", {}),
    ("HPROM-GPR (global)", "run_hprom_gpr", "hprom_gpr", {}),

    ("Local HPROM (Nc=3)", "run_local_hprom", "local_hprom",
     {"local_model_file": "LocalPODSweep/nc3_postfix/tol1.0e-03/local_pod_data.npz"}),
    ("Local HPROM (Nc=5)", "run_local_hprom", "local_hprom",
     {"local_model_file": "LocalPODSweep/nc5_postfix/tol1.5e-03/local_pod_data.npz"}),
    ("Local HPROM (Nc=8)", "run_local_hprom", "local_hprom",
     {"local_model_file": "LocalPODSweep/nc8_postfix/tol1.5e-03/local_pod_data.npz"}),

    ("Local HQPROM (Nc=3)", "run_local_hqprom", "local_hqprom",
     {"local_model_file": "LocalQuadraticSweep/nc3_alphahigh/z1.2_a1e06/local_qm_data.npz",
      "selector_mode": "quadratic"}),
    ("Local HQPROM (Nc=5)", "run_local_hqprom", "local_hqprom",
     {"local_model_file": "LocalQuadraticSweep/local_hqprom_sweep_nc5_postfix/z1_a1e04/local_qm_data.npz",
      "selector_mode": "quadratic"}),
    ("Local HQPROM (Nc=8)", "run_local_hqprom", "local_hqprom",
     {"local_model_file": "LocalQuadraticSweep/local_hqprom_sweep_nc8_postfix/z0.7_a1e04/local_qm_data.npz",
      "selector_mode": "quadratic"}),

    ("Local HPROM-GPR (Nc=3)", "run_local_hprom_gpr", "local_hprom_gpr",
     {"local_model_file": "LocalPOD-GPRSweep/nc3_check/nprimary9/all_offline.npz",
      "verbose": False}),
    ("Local HPROM-GPR (Nc=5)", "run_local_hprom_gpr", "local_hprom_gpr",
     {"local_model_file": "LocalPOD-GPRSweep/nc5/nprimary7/all_offline.npz",
      "verbose": False}),
    ("Local HPROM-GPR (Nc=8)", "run_local_hprom_gpr", "local_hprom_gpr",
     {"local_model_file": "LocalPOD-GPRSweep/nc8/nprimary6/all_offline.npz",
      "verbose": False}),
]

# The local runners keep their ECSW weights next to the model.
LOCAL_WEIGHTS = {
    "run_local_hprom": "local_hprom_ecsw_weights.npy",
    "run_local_hqprom": "local_hqprom_ecsw_weights.npy",
    "run_local_hprom_gpr": "ecsw_weights.npy",
}

# Where each global runner keeps its weights, relative to its work directory.
GLOBAL_WEIGHTS = {
    "run_hprom": "POD/ecsw_weights_lspg.npy",       # derived from pod_dir
    "run_hqprom": "hqprom_ecsw_weights.npy",
    "run_hprom_gpr": "ecsw_weights_gpr.npy",
}

# Read-only files run_hprom.py needs beside the weights it writes.
POD_FILES = ["basis.npy", "sigma.npy", "u_ref.npy"]

FIELDS = ["row", "mu1", "mu2", "online_s", "err_percent", "gn_iterations",
          "num_modes", "n_e", "jac_s", "res_s", "linsolve_s", "ecsw_s", "wall_s"]


def slug(label):
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def setup_row(work):
    """Build one row's private working directory."""
    os.makedirs(os.path.join(work, "Results"), exist_ok=True)

    for name in LINK_DIRS:
        src, dst = os.path.join(REPO, name), os.path.join(work, name)
        if os.path.isdir(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    # A real POD/ so rebuilt weights land here, not in the repository.
    pod = os.path.join(work, "POD")
    os.makedirs(pod, exist_ok=True)
    for name in POD_FILES:
        src, dst = os.path.join(REPO, "POD", name), os.path.join(pod, name)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    # The HDM snapshots are the error reference. Symlink them file by file into
    # a real directory so no run can write back into the repository copy.
    snaps = os.path.join(work, "Results", "param_snaps")
    os.makedirs(snaps, exist_ok=True)
    src = os.path.join(REPO, "Results", "param_snaps")
    if os.path.isdir(src):
        for name in os.listdir(src):
            if name.endswith(".npy") and not os.path.exists(os.path.join(snaps, name)):
                os.symlink(os.path.join(src, name), os.path.join(snaps, name))


def read_summary(prefix, mu1, mu2):
    path = "Results/%s_summary_mu1_%.2f_mu2_%.3f.txt" % (prefix, mu1, mu2)
    out = {}
    if os.path.exists(path):
        for line in io.open(path, encoding="utf-8", errors="replace"):
            if ":" in line:
                key, _, val = line.partition(":")
                out[key.strip()] = val.strip()
    return out


def num(summary, *keys):
    for k in keys:
        if k in summary:
            try:
                return float(summary[k])
            except ValueError:
                pass
    return float("nan")


def write_csv(path, records):
    """Rewrite the CSV from scratch, so a job that dies partway through still
    leaves the rows it finished."""
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in FIELDS})


def resolve_kwargs(modname, extra, work, rebuild):
    """Absolutise model paths and point every write at `work`.

    Returns (kwargs, weights_path). When rebuilding, the weights path is inside
    `work`, so a rebuilt cubature never lands on top of the copy the
    repository ships. When reusing, it points at that shipped copy.
    """
    kwargs = dict(extra)
    model = kwargs.get("local_model_file")

    if model is not None:
        kwargs["local_model_file"] = os.path.join(REPO, model)
        weights = (os.path.join(work, LOCAL_WEIGHTS[modname]) if rebuild
                   else os.path.join(REPO, os.path.dirname(model),
                                     LOCAL_WEIGHTS[modname]))
        kwargs["weights_file"] = weights
        return kwargs, weights

    # Globals. run_hprom derives its weight path from pod_dir, so it always
    # writes into the private POD/ built by setup_row; the other two take an
    # explicit weights_file.
    if modname == "run_hprom":
        kwargs["pod_dir"] = os.path.join(work, "POD")
        return kwargs, os.path.join(work, GLOBAL_WEIGHTS[modname])

    weights = os.path.join(work, GLOBAL_WEIGHTS[modname])
    if not rebuild:
        weights = os.path.join(REPO, "POD-GPR/pod_gpr_model/ecsw_weights_gpr.npy"
                               if modname == "run_hprom_gpr"
                               else "Results/hqprom_ecsw_weights.npy")
    kwargs["weights_file"] = weights
    return kwargs, weights


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=os.path.join(REPO, "Results_Paper", "FinalTable"),
                    help="Root for the per-row working directories.")
    ap.add_argument("--out", default=None,
                    help="Output CSV (default: <work>/<slug>.csv, or "
                         "<work>/table_timings.csv when several rows run).")
    ap.add_argument("--only", default=None,
                    help="Substring filter on the row label, e.g. 'Local HQPROM'.")
    ap.add_argument("--repeats", type=int, default=3,
                    help="Runs per point. The first is discarded as warm-up "
                         "(cold model load, first-touch page faults) and the "
                         "median of the rest is reported.")
    ap.add_argument("--ecsw-snapshot-percent", type=float, default=None,
                    help="Override the fraction of snapshots the cubature is "
                         "trained on (the runners default to 2.0). A small "
                         "value makes --rebuild-ecsw cheap enough to smoke "
                         "test, but do not report numbers from it.")
    ap.add_argument("--rebuild-ecsw", action="store_true",
                    help="Recompute the ECSW weights for the selected rows "
                         "into their private directory. Needed for the global "
                         "linear, whose training-matrix assembly now evaluates "
                         "the residual at the projected snapshot; and for the "
                         "other two globals, whose published rows are not "
                         "reproducible from the artifacts on disk.")
    args = ap.parse_args()

    rows = [r for r in ROWS if args.only is None or args.only in r[0]]
    if not rows:
        print("No row matches --only %r" % args.only)
        return 1

    out_csv = args.out or os.path.join(
        args.work, "%s.csv" % slug(rows[0][0]) if len(rows) == 1 else "table_timings.csv")
    os.makedirs(args.work, exist_ok=True)
    out_csv = os.path.abspath(out_csv)
    sys.path.insert(0, REPO)

    records = []
    for label, modname, prefix, extra in rows:
        work = os.path.join(args.work, slug(label))
        setup_row(work)
        rebuild = bool(args.rebuild_ecsw)
        kwargs, weights = resolve_kwargs(modname, extra, work, rebuild)
        if args.ecsw_snapshot_percent is not None:
            kwargs["ecsw_snapshot_percent"] = args.ecsw_snapshot_percent

        model = kwargs.get("local_model_file")
        if model and not os.path.exists(model):
            print("SKIP  %-24s missing %s" % (label, model), flush=True)
            continue

        # run_hprom has no weights_file argument, so when reusing we place the
        # shipped copy where it expects to find it.
        if not rebuild and modname == "run_hprom" and not os.path.exists(weights):
            os.symlink(os.path.join(REPO, "POD/ecsw_weights_lspg.npy"), weights)

        os.chdir(work)
        mod = __import__(modname)
        print("\n=== %s%s ===" % (label, "   [rebuilding ECSW]" if rebuild else ""),
              flush=True)
        print("    work: %s" % work, flush=True)

        for mu1, mu2 in POINTS:
            times, rec = [], {}
            for rep in range(args.repeats):
                buf, real = io.StringIO(), sys.stdout
                sys.stdout = buf
                try:
                    wall0 = time.perf_counter()
                    elapsed, err = mod.main(mu1=mu1, mu2=mu2,
                                            compute_ecsw=rebuild, **kwargs)
                    wall = time.perf_counter() - wall0
                except Exception:
                    sys.stdout = real
                    print(buf.getvalue()[-3000:], flush=True)
                    raise
                finally:
                    sys.stdout = real

                if rebuild:
                    # The weights are saved now; every later run reuses them.
                    rebuild = False
                    if modname == "run_hprom":
                        kwargs.pop("pod_dir", None)
                        kwargs["pod_dir"] = os.path.join(work, "POD")
                if rep or args.repeats == 1:
                    times.append(elapsed)

                s = read_summary(prefix, mu1, mu2)
                rec = {
                    "row": label, "mu1": mu1, "mu2": mu2,
                    "err_percent": err, "wall_s": wall,
                    "gn_iterations": int(num(s, "gn_iterations_total") or 0),
                    "num_modes": s.get("num_modes_retained",
                                       s.get("basis_size",
                                             s.get("retained_modes_per_cluster", ""))),
                    "n_e": int(num(s, "num_nonzero_weights") or 0),
                    "jac_s": num(s, "jacobian_time_seconds"),
                    "res_s": num(s, "residual_time_seconds"),
                    "linsolve_s": num(s, "linear_solve_time_seconds"),
                    "ecsw_s": num(s, "ecsw_time_seconds"),
                }

            rec["online_s"] = statistics.median(times)
            records.append(rec)
            print("  mu=(%.2f,%.3f)  %7.2f s  err=%.6f%%  its=%d  Ne=%d"
                  % (mu1, mu2, rec["online_s"], rec["err_percent"],
                     rec["gn_iterations"], rec["n_e"]), flush=True)

            write_csv(out_csv, records)

    print("\n" + "=" * 80)
    print("%-24s %9s %9s %9s %7s %10s" % (
        "row", "mean s", "avg err%", "max err%", "Ne", "modes"))
    print("=" * 80)
    for label in [r[0] for r in rows]:
        got = [r for r in records if r["row"] == label]
        if not got:
            continue
        n = float(len(got))
        print("%-24s %9.2f %9.3f %9.3f %7d %10s"
              % (label, sum(r["online_s"] for r in got) / n,
                 sum(r["err_percent"] for r in got) / n,
                 max(r["err_percent"] for r in got),
                 got[0]["n_e"], got[0]["num_modes"]))
    print("\nCSV: %s" % out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
