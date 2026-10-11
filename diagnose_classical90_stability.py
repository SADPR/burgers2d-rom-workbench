#!/usr/bin/env python3
"""Evaluate the saved 90-pair classical rule for Appendix Table 5.

Reuses the production solver and measurement protocol of the existing
stability comparison. No cubature fitting is performed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from diagnose_bm_stability import production_run, save_json, PARAMETERS
from diagnose_bm_ecsw import RULES, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=10)
    parser.add_argument('--output-dir', default='Results/BM_ECSW_stability_classical90_20261009')
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    name = 'classical_90_offset3'
    weight_path = Path('POD')/RULES[name]
    basis = np.load('POD/basis.npy')
    reference = np.load('POD/u_ref.npy')
    weights = np.load(weight_path)
    with threadpool_limits(limits=args.threads):
        save_json(out/'provenance.json', dict(parameters=PARAMETERS, dt=.05,
            steps=500, modes=basis.shape[1], threads=threadpool_info(),
            weights={name:dict(path=str(weight_path),sha256=digest(weight_path),
                cells=int(np.count_nonzero(weights>0)),training_pairs=90)},
            sources={path:digest(path) for path in ('POD/basis.npy','POD/u_ref.npy',
                'burgers/gauss_newton.py','burgers/linear_manifold.py',
                'diagnose_bm_stability.py',__file__)},
            online='Saved classical ECSW weights; production GN; max20; relative1e-5; plateau.01; no retraining.',
            matrix_sampling='First correction matrix per time step supplies Table 5; all solved corrections are saved.',
            purpose='Additional classical 90-pair rule in the stability comparison.'))
        runs = []
        for mu in PARAMETERS:
            save_json(out/'status.json', dict(stage='online',rule=name,mu=mu,
                completed=len(runs),total=3))
            result = production_run(name,weights,mu,basis,reference,out,500)
            runs.append(result)
            save_json(out/'runs.json',runs)
            print(json.dumps(result),flush=True)
        save_json(out/'status.json',dict(stage='complete',online_completed=3))


if __name__ == '__main__':
    main()
