# Findings from the local Euclidean tests

The useful change is an online residual-adaptive tail correction, not the previously tested fixed global B_r.

At each time step, build three tail patterns from the current residual and its Jacobian, freeze those patterns, and solve for their amplitudes together with the ten primary coordinates. The ANN architecture, weights and nine-parameter training set are unchanged.

The initial-state variant uses only the prescribed initial condition projected onto the existing 151-mode space. No reference trajectory is fed into an online solve.

## What helped on validation

- Case 2, original initialization: 3.8290% mean coefficient-trajectory error against the linear PROM.
- Case 2 + residual-adaptive r=1: 1.9884% mean coefficient-trajectory error against the linear PROM.
- Case 2 + residual-adaptive r=3: 1.0928% mean coefficient-trajectory error against the linear PROM.
- Case 2 + residual-adaptive r=3 + known initial state: 0.2308% mean coefficient-trajectory error against the linear PROM.
- Case 1, known initial state: 0.9992% mean coefficient-trajectory error against the linear PROM.
- Case 3, known initial state: 0.8069% mean coefficient-trajectory error against the linear PROM.

## What did not justify adoption

- The fixed SVD correction reduced validation error only slightly, even with ten extra amplitudes.
- ANN parameter/time derivative directions and constant affine primary-innovation feedback worsened the validation error.
- Increasing the adaptive rank from three to five did not improve either validation trajectory with the original initialization.
- Changing initialization alone did not improve mean validation error; the strong result requires the residual-adaptive correction.
- Removing the ANN after t=0 substantially degraded the adaptive method. The result is not explained by physics correction alone.

## Reporting errors against HDM

- PROM-ANN Case 2 (10): in-domain mean 1.6071%; extrapolation 2.6962%.
- Case 2 + residual-adaptive r=3: in-domain mean 0.6653%; extrapolation 1.0233%.
- Case 2 + residual-adaptive r=3 + known initial state: in-domain mean 0.4468%; extrapolation 0.8503%.
- Case 1, known initial state: in-domain mean 0.5656%; extrapolation 1.6729%.
- Case 3, known initial state: in-domain mean 0.4940%; extrapolation 1.5208%.
- Linear PROM (151): in-domain mean 0.4411%; extrapolation 0.8499%.

## Why the combination matters

Residual minimization advances the solution from the previous computed state. If the ANN supplies an inaccurate initial state, a small residual can still describe the evolution of that inaccurate initial condition. The known-initial-state treatment removes this source; the adaptive tail correction addresses the new injection errors at subsequent time steps. The ablation results are consistent with this explanation, rather than showing that a smaller residual automatically implies a smaller HDM error.

## Cost and interpretation

This changes the online PROM. It solves 13 amplitudes per time step, but direction construction uses all 141 tail modes and the full residual/Jacobian. It is not claimed to have the original Case-2 cost or to be hyper-reduced.

The first local timings place the adaptive method near the cost of Cases 1 and 3 and above unmodified Case 2. The timing table in results.md separates repeated measurements from single observations; CPU scheduling and frequency were not controlled.

The appropriate conclusion is a promising, validated accuracy improvement for this Euclidean campaign, not universal superiority. All four reporting parameters are pre-existing manuscript points, not a new independent generalization benchmark.

See results.md, the CSV tables, the saved coefficient trajectories and [the method notes](../../CASE2_LOCAL_CORRECTIONS.md) for the experiment definitions and provenance.
