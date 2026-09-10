#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Re-run every row of the online-performance table with one code version.

The table currently mixes global rows measured in April/May with local rows
measured in September, so its wall-clock and speed-up columns are not
comparable. This driver re-runs all twelve models -- three global and three
local families at N_c = 3, 5, 8 -- from the exact artifacts each row was
reported with, and writes one CSV.

Everything happens inside a private working directory with the model trees
symlinked in, so the runners' relative `results_dir="Results"` lands there and
nothing in the repository's Results/ is overwritten. That directory is shared
by every run and is what silently mismatched figures and numbers before.

Only the online solve is re-run (`compute_ecsw=False`); the saved ECSW weights
are reused. The one exception is the global HPROM, whose ECSW assembly changed
with the projected-state consistency fix -- pass --recompute-ecsw-hprom to
rebuild its weights, which will also change its N_e and its error.
"""

import argparse
import csv
import io
import os
import statistics
import sys
import time
import warnings

warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.abspath(__file__))
POINTS = [(4.56, 0.019), (4.75, 0.020), (5.19, 0.026)]

# The model trees the runners reach through relative paths.
LINK_DIRS = [
    "POD", "Quadratic", "POD-GPR",
    "LocalPODSweep", "LocalQuadraticSweep", "LocalPOD-GPRSweep",
]

# label, runner module, summary prefix, extra main() kwargs.
# Every local candidate below was matched to its table row by N_e *and* by the
# reported avg/max error, because N_e alone is ambiguous: in the POD-GPR sweep
# it is set by n_primary, so nc5/nprimary6 and nc8/nprimary6 share N_e = 541.
ROWS = [
    ("HPROM (global)", "run_hprom", "hprom", {"num_modes": 96}),
    ("HQPROM (global)", "run_hqprom", "hqprom",
     {"weights_file": REPO + "/Results/hqprom_ecsw_weights.npy"}),
    ("HPROM-GPR (global)", "run_hprom_gpr", "hprom_gpr",
     {"weights_file": REPO + "/POD-GPR/pod_gpr_model/ecsw_weights_gpr.npy"}),

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

# The local runners take their weights from the model's own directory.
WEIGHT_NAME = {
    "run_local_hprom": "local_hprom_ecsw_weights.npy",
    "run_local_hqprom": "local_hqprom_ecsw_weights.npy",
    "run_local_hprom_gpr": "ecsw_weights.npy",
}

FIELDS = ["row", "mu1", "mu2", "online_s", "err_percent", "gn_iterations",
          "num_modes", "n_e", "jac_s", "res_s", "linsolve_s", "wall_s"]


def write_csv(path, records):
    """Rewrite the CSV from scratch.

    Called after every row so that a job which dies partway through still
    leaves the rows it did finish.
    """
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in FIELDS})


def setup(work):
    """Build the private working directory."""
    os.makedirs(os.path.join(work, "Results"), exist_ok=True)
    for name in LINK_DIRS:
        src, dst = os.path.join(REPO, name), os.path.join(work, name)
        if os.path.isdir(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    # The HDM snapshots are the error reference. Symlink the files one by one
    # into a real directory so no run can write back into the repository copy.
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
    if not os.path.exists(path):
        return out
    for line in io.open(path, encoding="utf-8"):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default=os.path.join(REPO, "TableRerun"),
                    help="Private working directory (default: TableRerun/).")
    ap.add_argument("--out", default=None,
                    help="Output CSV (default: <work>/table_timings.csv).")
    ap.add_argument("--only", default=None,
                    help="Substring filter on the row label, e.g. 'Local HQPROM'.")
    ap.add_argument("--repeats", type=int, default=3,
                    help="Runs per point. The first is discarded as warm-up "
                         "(cold model load, first-touch page faults) and the "
                         "median of the rest is reported. Use 1 to time the "
                         "single cold run instead.")
    ap.add_argument("--recompute-ecsw-hprom", action="store_true",
                    help="Rebuild the global HPROM ECSW weights, which the "
                         "projected-state consistency fix changes.")
    args = ap.parse_args()

    rows = [r for r in ROWS if args.only is None or args.only in r[0]]
    if not rows:
        print("No row matches --only %r" % args.only)
        return 1

    setup(args.work)
    out_csv = args.out or os.path.join(args.work, "table_timings.csv")
    os.chdir(args.work)
    sys.path.insert(0, REPO)

    records = []
    for label, modname, prefix, extra in rows:
        kwargs = dict(extra)

        model = kwargs.get("local_model_file")
        if model is not None:
            kwargs["local_model_file"] = os.path.join(REPO, model)
            kwargs["weights_file"] = os.path.join(
                REPO, os.path.dirname(model), WEIGHT_NAME[modname])

        compute_ecsw = bool(args.recompute_ecsw_hprom and label == "HPROM (global)")
        if compute_ecsw:
            kwargs.pop("weights_file", None)

        missing = [p for p in (kwargs.get("local_model_file"), kwargs.get("weights_file"))
                   if p and not os.path.exists(p)]
        if missing:
            print("SKIP  %-24s missing: %s" % (label, ", ".join(missing)), flush=True)
            continue

        mod = __import__(modname)
        print("\n=== %s%s ===" % (label, "  [recomputing ECSW]" if compute_ecsw else ""),
              flush=True)

        for mu1, mu2 in POINTS:
            times, rec = [], {}
            for rep in range(args.repeats):
                buf, real = io.StringIO(), sys.stdout
                sys.stdout = buf
                try:
                    wall0 = time.perf_counter()
                    elapsed, err = mod.main(
                        mu1=mu1, mu2=mu2,
                        compute_ecsw=(compute_ecsw and rep == 0),
                        **kwargs)
                    wall = time.perf_counter() - wall0
                finally:
                    sys.stdout = real
                if rep or args.repeats == 1:
                    times.append(elapsed)
                s = read_summary(prefix, mu1, mu2)
                rec = {
                    "row": label, "mu1": mu1, "mu2": mu2,
                    "err_percent": err, "wall_s": wall,
                    "gn_iterations": int(num(s, "gn_iterations_total") or 0),
                    "num_modes": s.get("num_modes_retained",
                                       s.get("retained_modes_per_cluster", "")),
                    "n_e": int(num(s, "num_nonzero_weights") or 0),
                    "jac_s": num(s, "jacobian_time_seconds"),
                    "res_s": num(s, "residual_time_seconds"),
                    "linsolve_s": num(s, "linear_solve_time_seconds"),
                }
                # After a recompute the weights are saved; reuse them afterwards.
                compute_ecsw = False

            rec["online_s"] = statistics.median(times)
            records.append(rec)
            print("  mu=(%.2f,%.3f)  %7.2f s  err=%.6f%%  its=%d  Ne=%d"
                  % (mu1, mu2, rec["online_s"], rec["err_percent"],
                     rec["gn_iterations"], rec["n_e"]), flush=True)

        write_csv(out_csv, records)

    print("\n" + "=" * 78)
    print("%-24s %9s %9s %9s %7s" % ("row", "mean s", "avg err%", "max err%", "Ne"))
    print("=" * 78)
    for label in [r[0] for r in rows]:
        got = [r for r in records if r["row"] == label]
        if not got:
            continue
        n = float(len(got))
        print("%-24s %9.2f %9.3f %9.3f %7d"
              % (label, sum(r["online_s"] for r in got) / n,
                 sum(r["err_percent"] for r in got) / n,
                 max(r["err_percent"] for r in got), got[0]["n_e"]))
    print("\nCSV: %s" % out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
