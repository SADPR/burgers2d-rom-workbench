#!/usr/bin/env python3
"""Generate parameter-domain figure with training and test points."""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TRAIN_MU1 = np.array([4.25, 4.875, 5.5])
TRAIN_MU2 = np.array([0.015, 0.0225, 0.03])
TEST_POINTS = np.array([
    [4.56, 0.019],
    [4.75, 0.020],
    [5.19, 0.026],
])

OUT = Path(__file__).resolve().parent / 'Figures' / 'parameter_domain_and_test_points.png'
OUT.parent.mkdir(parents=True, exist_ok=True)

# Cartesian product for training points
train_pts = np.array([(x, y) for x in TRAIN_MU1 for y in TRAIN_MU2], dtype=float)

fig, ax = plt.subplots(figsize=(7.8, 5.4))

# Domain rectangle
x0, x1 = TRAIN_MU1.min(), TRAIN_MU1.max()
y0, y1 = TRAIN_MU2.min(), TRAIN_MU2.max()
ax.fill([x0, x1, x1, x0], [y0, y0, y1, y1], color='#D3D3D3', alpha=0.25, zorder=0, label='Training domain')

# Train/test points
ax.scatter(train_pts[:, 0], train_pts[:, 1], s=70, c='black', marker='o', label='Training points', zorder=3)
ax.scatter(TEST_POINTS[:, 0], TEST_POINTS[:, 1], s=170, c='#d62728', marker='*', edgecolors='black', linewidths=0.6, label='Test points', zorder=4)

for i, (mx, my) in enumerate(TEST_POINTS, start=1):
    ax.annotate(rf'$\mu_{{{i}}}$', (mx, my), xytext=(6, 5), textcoords='offset points', fontsize=10)

ax.set_xlabel(r'$\mu_1$')
ax.set_ylabel(r'$\mu_2$')
ax.set_title('Parameter Domain and Test Points')
ax.grid(True, alpha=0.35, linewidth=0.6)
ax.legend(loc='best', frameon=True)

xpad = 0.06 * (x1 - x0)
ypad = 0.18 * (y1 - y0)
ax.set_xlim(x0 - xpad, x1 + xpad)
ax.set_ylim(y0 - ypad, y1 + ypad)

plt.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'Saved: {OUT}')
