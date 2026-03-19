"""
Define problem constants / 'magic numbers'
"""

import numpy as np
from .core import make_2D_grid

BATCH_SIZE = 16
TRAIN_FRAC = 0.9
SNAP_FOLDER = "param_snaps"

SEED = 1234557

## PROBLEM-WIDE CONSTANTS
#   These define the underlying hdm, so set them once and use the same values for all ROM
#   and neural network runs
DT = 0.05
NUM_STEPS = 500
NUM_CELLS = 100
XL, XU = 0, 100
U0 = np.ones((NUM_CELLS, NUM_CELLS))
V0 = U0.copy()
W0 = np.concatenate((U0.ravel(), V0.ravel()))
GRID_X, GRID_Y = make_2D_grid(XL, XU, XL, XU, NUM_CELLS, NUM_CELLS)
MU1_RANGE = 4.25, 5.5
MU2_RANGE = 0.015, 0.03
SAMPLES_PER_MU = 3



