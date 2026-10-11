#!/usr/bin/env python3
"""Produce the exact-Jacobian gradient test table from saved diagnostics."""
import argparse
import json
from pathlib import Path


def format_ratio(value):
    if value<1e-10:
        mantissa,exponent=f'{value:.2e}'.split('e')
        return rf'${mantissa}\times10^{{{int(exponent)}}}$'
    return f'{value:.3f}'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',default='Results/BM_ECSW_gradient_structure_20261009')
    parser.add_argument('--document',default='Project_BM-ECSW')
    args=parser.parse_args()
    out=Path(args.results)
    if json.loads((out/'status.json').read_text())['stage']!='complete':
        raise RuntimeError('Gradient diagnostics are incomplete.')
    data=json.loads((out/'summary.json').read_text())
    online={r['rule']:r for r in data['trajectory_states']}
    rows=[]
    for name,model,count in (
        ('classical_90_offset3','HPROM (ECSW)',90),
        ('classical_225_offset1','HPROM (ECSW)',225),
        ('bm_90_offset1','HPROM (BM-ECSW)',90),
        ('bm_225_offset1','HPROM (BM-ECSW)',225)):
        assert online[name]['evaluations']==1500
        number=f"{online[name]['nonsymmetric_evaluations']:,}".replace(',',r'\,')
        rows.append(f"{model} & {count} & {format_ratio(online[name]['exact_asymmetry_median'])} & ${number}/1\\,500$\\\\")
    table=r'''\begin{tabular}{@{}lrrr@{}}
\toprule
Model & $N_c$ & $\|\matr{J}_n-\matr{J}_n^T\|_F/\|\matr{J}_n\|_F$ & Nonsymmetric\\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
'''
    (Path(args.document)/'tables/burgers_gradient_structure.tex').write_text(table)
    print(json.dumps(data,indent=2))


if __name__=='__main__':
    main()
