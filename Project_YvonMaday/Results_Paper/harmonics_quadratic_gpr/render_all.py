import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent

for script in (
    "linear_manifold.py",
    "piecewise_linear_manifold.py",
    "quadratic_manifold.py",
    "nonlinear_closure_manifold.py",
    "generic_decoder_tangent.py",
    "pod_ae_manifold.py",
    "case1_state_closure.py",
    "case2_parameter_time_closure.py",
    "case3_hybrid_closure.py",
):
    subprocess.run([sys.executable, str(HERE / script)], check=True)
