#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse


def parse_summary(p: Path):
    d = {}
    for line in p.read_text().splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            d[k.strip()] = v.strip()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Case2_oracle root folder')
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for s in sorted(root.glob('mu1_*/**/*_summary.txt')):
        d = parse_summary(s)
        name = s.name
        label = name.split('_mu1_')[0]
        mu1 = d.get('mu1', 'NA')
        mu2 = d.get('mu2', 'NA')
        rows.append((
            label,
            mu1,
            mu2,
            d.get('relative_error_percent_vs_hdm', 'NA'),
            d.get('relative_error_percent_vs_linear_prom', 'NA'),
            d.get('contamination_gain_primary_dq_over_dqbar', 'NA'),
            d.get('contamination_gain_state_du_over_dqbar', 'NA'),
        ))

    print('label,mu1,mu2,rel_hdm_pct,rel_linear_pct,gain_q,gain_u')
    for r in rows:
        print(','.join(map(str, r)))


if __name__ == '__main__':
    main()
