# Online evaluation of the 1826-cell PSD checkpoint

Training residual: 3.823157812e-05; requested tolerance 1e-5 was not attained.
The weights are exported directly from the last saved full joint refit; no coefficients were changed.
96 POD modes, 500 time steps, unchanged production nonlinear solver and stopping settings.

| Parameter | Method | Cells | Augmented cells | Online (s) | Trajectory error (%) | Maximum snapshot error (%) |
|---|---|---:|---:|---:|---:|---:|
| (4.56, 0.019) | HPROM (ECSW) | 2 492 | 5 914 | 15.892 | 1.095919 | 3.795464 |
| (4.56, 0.019) | HPROM (BM-ECSW) | 300 | 868 | 1.165 | 1.101890 | 3.795464 |
| (4.56, 0.019) | HPROM (PSD, checkpoint) | 1 826 | 4 486 | 14.107 | 1.095890 | 3.795464 |
| (4.75, 0.02) | HPROM (ECSW) | 2 492 | 5 914 | 15.934 | 1.001054 | 3.795464 |
| (4.75, 0.02) | HPROM (BM-ECSW) | 300 | 868 | 1.221 | 1.006145 | 3.795464 |
| (4.75, 0.02) | HPROM (PSD, checkpoint) | 1 826 | 4 486 | 15.293 | 1.001033 | 3.795464 |
| (5.19, 0.026) | HPROM (ECSW) | 2 492 | 5 914 | 16.138 | 0.992762 | 3.795464 |
| (5.19, 0.026) | HPROM (BM-ECSW) | 300 | 868 | 1.188 | 1.006881 | 3.795464 |
| (5.19, 0.026) | HPROM (PSD, checkpoint) | 1 826 | 4 486 | 15.308 | 0.992738 | 3.795464 |

One fresh-process run per method and parameter; sequential execution with ten native threads.
Errors compare reconstructed trajectories to cached HDM trajectories. Timings include mesh setup, initial projection and integration, and exclude decoding and error evaluation.
Concurrent AERO-F activity may affect timings; these runs primarily assess the online behavior of the incomplete training rule.
Training-residual accuracy and online trajectory accuracy are different quantities.
