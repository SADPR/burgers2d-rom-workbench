# Paper Runbook (Reproducible Commands)

This file records the exact commands and artifacts used for paper figures/tables.

Scope: current repository only (`burgers2d-rom-workbench`), not `Project_YvonMaday`.
Policy: this runbook includes only HPROM/HQPROM-family runs used in the paper (no PROM runs).

## Global Settings

- Test points:
  - `(mu1, mu2) = (4.56, 0.019)`
  - `(mu1, mu2) = (4.75, 0.020)`
  - `(mu1, mu2) = (5.19, 0.026)`
- Base path:
  - `/scratch/users/sadpr/Codes26Feb/burgers2d-rom-workbench`

---

## Method 1: Global HPROM (Linear POD)

- Status: ACTIVE (candidate)
- Method tag: `global_hprom_linear`
- Selected setting (candidate):
  - `pod_tol = 5e-4` (Stage-1 POD truncation)
  - `n_keep = 96` (from `POD/stage1_pod_summary.txt`)
  - HPROM uses `num_modes=96`

### A) Build POD basis with `pod_tol=5e-4` (Python launcher)

```bash
cd /scratch/users/sadpr/Codes26Feb/burgers2d-rom-workbench
export PYTHONPATH="$PWD:$PYTHONPATH"

python3 - <<'PY'
import runpy

stage1 = runpy.run_path("POD/stage1_build_pod_basis.py")["main"]
stage1(
    pod_method="svd",
    pod_tol=5e-4,
    num_modes=None,
    center=True,
    random_state=0,
)
PY
```

### B) Check stage1 summary (`pod_tol=5e-4`, expected `n_keep=96`)

```bash
grep -E "pod_tol|n_keep|energy_lost" POD/stage1_pod_summary.txt
```

### C) Run HPROM on 3 points (compute ECSW once, then reuse)

```bash
python3 - <<'PY'
import run_hprom

pts = [(4.56,0.019), (4.75,0.020), (5.19,0.026)]

# Build ECSW once on first point
run_hprom.main(
    mu1=pts[0][0], mu2=pts[0][1],
    compute_ecsw=True,
    num_modes=96,
    pod_dir="POD",
    results_dir="Results",
    ecsw_random_seed=42,
)

# Reuse same ECSW weights for remaining points
for mu1, mu2 in pts[1:]:
    run_hprom.main(
        mu1=mu1, mu2=mu2,
        compute_ecsw=False,
        num_modes=96,
        pod_dir="POD",
        results_dir="Results",
    )
PY
```

### D) Extract max relative error over the 3 points

```bash
python3 - <<'PY'
import re
from pathlib import Path

files = [
    "Results/hprom_summary_mu1_4.56_mu2_0.019.txt",
    "Results/hprom_summary_mu1_4.75_mu2_0.020.txt",
    "Results/hprom_summary_mu1_5.19_mu2_0.026.txt",
]

vals = []
for f in files:
    txt = Path(f).read_text()
    vals.append(float(re.search(r"relative_error_percent:\s*([0-9eE+\-.]+)", txt).group(1)))

print(f"HPROM max relative error (%) = {max(vals):.8f}")
PY
```

## Method 2: Global HQPROM (Quadratic)

- Status: ACTIVE
- Method tag: `global_hqprom_quadratic`
- Selected settings:
  - Stage1 quadratic: `pod_tol = 1e-4`, `zeta_qua = 1.5`, `ridge_alpha = 1`
  - Stage1 normalization: `q_normalization_mode = "std"`, `q_normalization_eps = 1e-12`
  - Expected reduced size: `n_final = 39`
  - Traditional model path: `Quadratic/qm_*.npy`, `Quadratic/qm_metadata.npz`
  - Traditional HQPROM weights path: `Results/hqprom_ecsw_weights.npy`

### A) Build quadratic manifold (normal/traditional path)

```bash
cd /scratch/users/sadpr/Codes26Feb/burgers2d-rom-workbench
export PYTHONPATH="$PWD:$PYTHONPATH"

python3 - <<'PY'
import runpy
stage1 = runpy.run_path("Quadratic/stage1_quadratic_offline.py")["main"]
n = stage1(
    pod_tol=1e-4,
    zeta_qua=1.5,
    ridge_alpha=1.0,
    center_mode="on",
    q_normalization_mode="std",
    q_normalization_eps=1e-12,
    save_sv_plot=True,
)
print("n_final:", n)
PY
```

### B) Verify stage1 summary (settings + size)

```bash
grep -E "pod_tol|zeta_qua|ridge_alpha|q_normalization_mode|q_normalization_eps|n_final" \
  Quadratic/stage1_quadratic_offline_summary.txt
```

### C) Optional stage2 projection check on the 3 paper points

```bash
python3 - <<'PY'
import runpy
stage2 = runpy.run_path("Quadratic/stage2_quadratic_projection.py")["main"]
for mu1, mu2 in [(4.56,0.019), (4.75,0.020), (5.19,0.026)]:
    stage2(mu1=mu1, mu2=mu2)
PY
```

### D) Run HQPROM on 3 points (compute ECSW once, then reuse)

```bash
python3 - <<'PY'
import run_hqprom

pts = [(4.56,0.019), (4.75,0.020), (5.19,0.026)]
wfile = "Results/hqprom_ecsw_weights.npy"

run_hqprom.main(
    mu1=pts[0][0], mu2=pts[0][1],
    qm_dir="Quadratic",
    weights_file=wfile,
    compute_ecsw=True,
    ecsw_random_seed=42,
)

for mu1, mu2 in pts[1:]:
    run_hqprom.main(
        mu1=mu1, mu2=mu2,
        qm_dir="Quadratic",
        weights_file=wfile,
        compute_ecsw=False,
    )
PY
```

### E) Extract HQPROM max relative error over the 3 points

```bash
python3 - <<'PY'
import re
from pathlib import Path

files = [
    "Results/hqprom_summary_mu1_4.56_mu2_0.019.txt",
    "Results/hqprom_summary_mu1_4.75_mu2_0.020.txt",
    "Results/hqprom_summary_mu1_5.19_mu2_0.026.txt",
]

vals = []
for f in files:
    txt = Path(f).read_text()
    vals.append(float(re.search(r"relative_error_percent:\s*([0-9eE+\-.]+)", txt).group(1)))

print(f"HQPROM max relative error (%) = {max(vals):.8f}")
PY
```

## Method 3: Global HPROM-GPR

- Status: ACTIVE
- Method tag: `global_hprom_gpr`
- Selected settings:
  - Stage1 POD: `pod_tol = 1e-4` (expected `n_keep = 151`)
  - Stage2 split: `total_modes = 151`, `primary_modes = 20`, `secondary = 131`
  - Stage3 GPR: `kernel_name = "matern15"`, `alpha = 1e-4`, `n_restarts_optimizer = 1`
  - Online: `uref_mode = "auto"`, `use_custom_predict = True`
  - HPROM-GPR weights: `POD-GPR/pod_gpr_model/ecsw_weights_gpr.npy`

### A) Stage1 POD basis

```bash
cd /scratch/users/sadpr/Codes26Feb/burgers2d-rom-workbench
export PYTHONPATH="$PWD:$PYTHONPATH"

python3 - <<'PY'
import runpy
s1 = runpy.run_path("POD-GPR/stage1_build_pod_basis.py")["main"]
s1(
    pod_method="svd",
    pod_tol=1e-4,
    num_modes=None,
    center=True,
)
PY
```

### B) Stage2 projection (`n_p=20`, `n_tot=151`)

```bash
python3 - <<'PY'
import runpy
s2 = runpy.run_path("POD-GPR/stage2_project_training_data.py")["main"]
s2(
    primary_modes=20,
    total_modes=151,
    uref_mode="auto",
    test_params=[[4.56,0.019],[4.75,0.020],[5.19,0.026]],
)
PY
```

### C) Stage3 train GPR

```bash
python3 - <<'PY'
import runpy
s3 = runpy.run_path("POD-GPR/stage3_train_gpr.py")["main"]
s3(
    kernel_name="matern15",
    alpha=1e-4,
    optimize_hyperparameters=True,
    n_restarts_optimizer=1,
    max_train_samples=None,
    validation_fraction=0.1,
    random_seed=42,
    uref_mode="auto",
)
PY
```

### D) Stage4 reconstruction check on 3 points

```bash
python3 - <<'PY'
import runpy
from pathlib import Path

s4 = runpy.run_path("POD-GPR/stage4_test_gpr.py")["main"]
pts = [(4.56,0.019),(4.75,0.020),(5.19,0.026)]

for mu1,mu2 in pts:
    out = Path("POD-GPR") / f"stage4_mu1_{mu1:.2f}_mu2_{mu2:.3f}"
    s4(
        target_mu=(mu1,mu2),
        model_dir="POD-GPR/pod_gpr_model",
        output_dir=str(out),
        uref_mode="auto",
        compare_pod=False,
        use_custom_predict=True,
    )
PY
```

### E) Run online HPROM-GPR (compute ECSW once, then reuse)

```bash
python3 - <<'PY'
import run_hprom_gpr
pts=[(4.56,0.019),(4.75,0.020),(5.19,0.026)]
wfile="POD-GPR/pod_gpr_model/ecsw_weights_gpr.npy"

run_hprom_gpr.main(
    mu1=pts[0][0], mu2=pts[0][1],
    model_dir="POD-GPR/pod_gpr_model",
    weights_file=wfile,
    compute_ecsw=True,
    uref_mode="auto",
    use_custom_predict=True,
    ecsw_random_seed=42,
)

for mu1,mu2 in pts[1:]:
    run_hprom_gpr.main(
        mu1=mu1, mu2=mu2,
        model_dir="POD-GPR/pod_gpr_model",
        weights_file=wfile,
        compute_ecsw=False,
        uref_mode="auto",
        use_custom_predict=True,
    )
PY
```

### F) Extract HPROM-GPR max relative error over the 3 points

```bash
python3 - <<'PY'
import re
from pathlib import Path

files = [
    "Results/hprom_gpr_summary_mu1_4.56_mu2_0.019.txt",
    "Results/hprom_gpr_summary_mu1_4.75_mu2_0.020.txt",
    "Results/hprom_gpr_summary_mu1_5.19_mu2_0.026.txt",
]

vals = []
for f in files:
    txt = Path(f).read_text()
    vals.append(float(re.search(r"relative_error_percent:\s*([0-9eE+\-.]+)", txt).group(1)))

print(f"HPROM-GPR max relative error (%) = {max(vals):.8f}")
PY
```

## Method 4: Local HPROM (Local POD)

- Status: ACTIVE
- Method tag: `local_hprom_linear`
- Selected settings (from `LocalPOD/stage1_local_pod_offline_summary.txt`):
  - `n_clusters = 10`
  - `clustering_method = "kmeans"`
  - `phi = 0.1`
  - `pod_tol = 1.5e-3`
  - `pod_method = "rsvd"`
  - Retained modes per cluster: `[14, 15, 20, 23, 14, 15, 18, 15, 22, 16]`
  - Traditional weights filename: `Results/local_hprom_ecsw_weights.npy`

### A) Ensure stage1 settings in `LocalPOD/stage1_local_pod_offline_cheap.py`

```bash
cd /scratch/users/sadpr/Codes26Feb/burgers2d-rom-workbench
export PYTHONPATH="$PWD:$PYTHONPATH"

python3 - <<'PY'
from pathlib import Path
import re

p = Path("LocalPOD/stage1_local_pod_offline_cheap.py")
s = p.read_text()
i = s.index("def main():")
head, tail = s[:i], s[i:]

tail = re.sub(r'^(\s*)n_clusters\s*=\s*.*$', r'\1n_clusters = 10', tail, count=1, flags=re.M)
tail = re.sub(r'^(\s*)pod_tol\s*=\s*.*$', r'\1pod_tol = 1.5e-3', tail, count=1, flags=re.M)
tail = re.sub(r'^(\s*)pod_method\s*=\s*.*$', r'\1pod_method = "rsvd"', tail, count=1, flags=re.M)
tail = re.sub(r'^(\s*)clustering_method\s*=\s*.*$', r'\1clustering_method = "kmeans"', tail, count=1, flags=re.M)
tail = re.sub(r'^(\s*)phi\s*=\s*.*$', r'\1phi = 0.1', tail, count=1, flags=re.M)

p.write_text(head + tail)
print("Updated LocalPOD stage1 settings.")
PY
```

### B) Train stage1 local POD model (traditional path)

```bash
python3 LocalPOD/stage1_local_pod_offline_cheap.py
```

### C) Check stage1 summary

```bash
grep -E "pod_tol|pod_method|n_clusters|clustering_method|phi_overlap|min_retained_modes|max_retained_modes|avg_retained_modes" \
  LocalPOD/stage1_local_pod_offline_summary.txt
```

### D) Run local HPROM on 3 points (compute ECSW once, then reuse)

```bash
python3 - <<'PY'
import run_local_hprom

pts=[(4.56,0.019),(4.75,0.020),(5.19,0.026)]
wfile="Results/local_hprom_ecsw_weights.npy"

run_local_hprom.main(
    mu1=pts[0][0], mu2=pts[0][1],
    local_model_file="LocalPOD/local_pod_data.npz",
    weights_file=wfile,
    compute_ecsw=True,
    ecsw_random_seed=42,
)

for mu1,mu2 in pts[1:]:
    run_local_hprom.main(
        mu1=mu1, mu2=mu2,
        local_model_file="LocalPOD/local_pod_data.npz",
        weights_file=wfile,
        compute_ecsw=False,
    )
PY
```

### E) Extract local HPROM max relative error over the 3 points

```bash
python3 - <<'PY'
import re
from pathlib import Path

files = [
    "Results/local_hprom_summary_mu1_4.56_mu2_0.019.txt",
    "Results/local_hprom_summary_mu1_4.75_mu2_0.020.txt",
    "Results/local_hprom_summary_mu1_5.19_mu2_0.026.txt",
]

vals = []
for f in files:
    txt = Path(f).read_text()
    vals.append(float(re.search(r"relative_error_percent:\s*([0-9eE+\-.]+)", txt).group(1)))

print(f"Local HPROM max relative error (%) = {max(vals):.8f}")
PY
```

## Method 5: Local HQPROM (Local Quadratic)

- Status: ACTIVE
- Method tag: `local_hqprom_quadratic`
- Selected settings:
  - `pod_tol = 1e-4`
  - `zeta_qua = 0.5`
  - `alpha_ridge = 0.3`
  - `q_normalization_mode = "std"`
  - `q_normalization_eps = 1e-12`
  - `n_clusters = 10`
  - `clustering_method = "kmeans"`
  - `phi = 0.1`
  - `pod_method = "svd"`
  - `selector_mode = "quadratic"`
- Traditional weights filename:
  - `Results/local_hqprom_ecsw_weights.npy`

### A) Optional clean (only local quadratic/local HQPROM artifacts)

```bash
cd /scratch/users/sadpr/Codes26Feb/burgers2d-rom-workbench
export PYTHONPATH="$PWD:$PYTHONPATH"

rm -f LocalQuadratic/local_qm_data.npz
rm -f LocalQuadratic/stage1_local_qm_offline_summary.txt
rm -f Results/local_hqprom_ecsw_weights.npy
rm -f Results/local_hqprom_summary_mu1_4.56_mu2_0.019.txt
rm -f Results/local_hqprom_summary_mu1_4.75_mu2_0.020.txt
rm -f Results/local_hqprom_summary_mu1_5.19_mu2_0.026.txt
```

### B) Ensure stage1 settings in `LocalQuadratic/stage1_local_qm_offline.py`

```bash
python3 - <<'PY'
from pathlib import Path
import re

p = Path("LocalQuadratic/stage1_local_qm_offline.py")
s = p.read_text()
i = s.index("def main():")
head, tail = s[:i], s[i:]

tail = re.sub(r'^(\s*)pod_tol\s*=\s*.*$', r'\1pod_tol = 1e-4', tail, count=1, flags=re.M)
tail = re.sub(r'^(\s*)zeta_qua\s*=\s*.*$', r'\1zeta_qua = 0.5', tail, count=1, flags=re.M)
tail = re.sub(r'^(\s*)alpha_ridge\s*=\s*.*$', r'\1alpha_ridge = 0.3', tail, count=1, flags=re.M)
tail = re.sub(r'^(\s*)q_normalization_mode\s*=\s*.*$', r'\1q_normalization_mode = "std"', tail, count=1, flags=re.M)
tail = re.sub(r'^(\s*)q_normalization_eps\s*=\s*.*$', r'\1q_normalization_eps = 1e-12', tail, count=1, flags=re.M)
tail = re.sub(r'^(\s*)n_clusters\s*=\s*.*$', r'\1n_clusters = 10', tail, count=1, flags=re.M)
tail = re.sub(r'^(\s*)clustering_method\s*=\s*.*$', r'\1clustering_method = "kmeans"', tail, count=1, flags=re.M)
tail = re.sub(r'^(\s*)phi\s*=\s*.*$', r'\1phi = 0.1', tail, count=1, flags=re.M)
tail = re.sub(r'^(\s*)pod_method\s*=\s*.*$', r'\1pod_method = "svd"', tail, count=1, flags=re.M)

p.write_text(head + tail)
print("Updated main() settings.")
PY
```

### C) Train stage1 local quadratic model (traditional path)

```bash
python3 LocalQuadratic/stage1_local_qm_offline.py
```

### C.1) Verify stage1 summary (target local quadratic config)

```bash
grep -E "pod_tol|zeta_qua|alpha_ridge|q_normalization_mode|q_normalization_eps|min_n_qm|max_n_qm|avg_n_qm" \
  LocalQuadratic/stage1_local_qm_offline_summary.txt
```

### D) Build ECSW once and run 3-point campaign with traditional weights path

```bash
python3 - <<'PY'
import run_local_hqprom as r

pts = [(4.56,0.019), (4.75,0.020), (5.19,0.026)]
wfile = "Results/local_hqprom_ecsw_weights.npy"

# Build ECSW once on first point
r.main(
    mu1=pts[0][0], mu2=pts[0][1],
    local_model_file="LocalQuadratic/local_qm_data.npz",
    weights_file=wfile,
    compute_ecsw=True,
    selector_mode="quadratic",
    ecsw_random_seed=42,
)

# Reuse same weights on remaining points
for mu1, mu2 in pts[1:]:
    r.main(
        mu1=mu1, mu2=mu2,
        local_model_file="LocalQuadratic/local_qm_data.npz",
        weights_file=wfile,
        compute_ecsw=False,
        selector_mode="quadratic",
    )
PY
```

### E) Extract Local HQPROM max relative error over the 3 points

```bash
python3 - <<'PY'
import re
from pathlib import Path

files = [
    "Results/local_hqprom_summary_mu1_4.56_mu2_0.019.txt",
    "Results/local_hqprom_summary_mu1_4.75_mu2_0.020.txt",
    "Results/local_hqprom_summary_mu1_5.19_mu2_0.026.txt",
]

vals = []
for f in files:
    txt = Path(f).read_text()
    vals.append(float(re.search(r"relative_error_percent:\s*([0-9eE+\-.]+)", txt).group(1)))

print(f"Local HQPROM max relative error (%) = {max(vals):.8f}")
PY
```

## Method 6: Local HPROM-GPR

- Status: ACTIVE
- Method tag: `local_hprom_gpr`
- Selected settings (from LocalPOD-GPR summaries):
  - Stage1 clustering: `n_clusters=10`, `clustering_method="kmeans"`, `phi=0.1`
  - Stage2 local POD per cluster: `eps2_pod=1e-4`, ranks `[24,30,42,47,25,25,38,27,46,34]`
  - Stage4 local GPR: `n_primary=6`, `kernel_candidates=["matern15"]`, `alpha_values=[1e-10]`
  - `optimize_hyperparameters=True`, `n_restarts_optimizer=3`, `random_seed=42`
  - Traditional weights filename: `Results/local_hprom_gpr_ecsw_weights.npy`

### A) Stage1 cluster snapshots

```bash
cd /scratch/users/sadpr/Codes26Feb/burgers2d-rom-workbench
export PYTHONPATH="$PWD:$PYTHONPATH"
python3 LocalPOD-GPR/stage1_cluster_snapshots_u.py
```

### B) Stage2 local POD per cluster

```bash
python3 LocalPOD-GPR/stage2_local_pod_per_cluster.py
```

### C) Stage3 project to local reduced coordinates

```bash
python3 LocalPOD-GPR/stage3_local_project_to_q.py
```

### D) Stage4 train local GPR models (fixed paper settings)

```bash
python3 - <<'PY'
import runpy
s4 = runpy.run_path("LocalPOD-GPR/stage4_local_pod_gpr_training.py")["main"]
s4(
    n_primary=6,
    kernel_candidates=("matern15",),
    alpha_values=(1e-10,),
    optimize_hyperparameters=True,
    n_restarts_optimizer=3,
    random_seed=42,
)
PY
```

### E) Run local HPROM-GPR on 3 points (compute ECSW once, then reuse)

```bash
python3 - <<'PY'
import run_local_hprom_gpr

pts=[(4.56,0.019),(4.75,0.020),(5.19,0.026)]
wfile="Results/local_hprom_gpr_ecsw_weights.npy"

run_local_hprom_gpr.main(
    mu1=pts[0][0], mu2=pts[0][1],
    local_model_file="LocalPOD-GPR/local_pod_gpr_all_offline.npz",
    weights_file=wfile,
    compute_ecsw=True,
    ecsw_random_seed=42,
    use_custom_predict=True,
    jacobian_mode="auto",
    selector_mode="nonlinear",
)

for mu1,mu2 in pts[1:]:
    run_local_hprom_gpr.main(
        mu1=mu1, mu2=mu2,
        local_model_file="LocalPOD-GPR/local_pod_gpr_all_offline.npz",
        weights_file=wfile,
        compute_ecsw=False,
        use_custom_predict=True,
        jacobian_mode="auto",
        selector_mode="nonlinear",
    )
PY
```

### F) Extract local HPROM-GPR max relative error over the 3 points

```bash
python3 - <<'PY'
import re
from pathlib import Path

files = [
    "Results/local_hprom_gpr_summary_mu1_4.56_mu2_0.019.txt",
    "Results/local_hprom_gpr_summary_mu1_4.75_mu2_0.020.txt",
    "Results/local_hprom_gpr_summary_mu1_5.19_mu2_0.026.txt",
]

vals = []
for f in files:
    txt = Path(f).read_text()
    vals.append(float(re.search(r"relative_error_percent:\s*([0-9eE+\-.]+)", txt).group(1)))

print(f"Local HPROM-GPR max relative error (%) = {max(vals):.8f}")
PY
```

---

## Final Paper Table (Fill As You Lock Each Method)

| Method | Config key | Max relative error % |
|---|---|---:|
| Global HPROM | TODO |  |
| Global HQPROM | TODO |  |
| Global HPROM-GPR | TODO |  |
| Local HPROM | TODO |  |
| Local HQPROM | `pod_tol=1e-4,zeta=0.4,alpha=1.0,q_norm=std` |  |
| Local HPROM-GPR | TODO |  |
