# Harmonic manifold benchmark

This folder is independent of the original manifold animations.

The controlled trajectory is

\[
\mathbf{u}(t)
=
\begin{bmatrix}
q_1(t)\\
q_2(t)\\
0.40\cos(2t)+0.06\sin(3t)
\end{bmatrix},
\]

with

\[
q_1(t)=\cos(t),
\qquad
q_2(t)=\sin(t).
\]

The trajectory is generated in harmonic coordinates, for which its third
component can be written directly as

\[
u_3
=
0.40\left(q_1^2-q_2^2\right)
+0.06\left(3q_2-4q_2^3\right).
\]

For the animations, the construction then follows the original manifold
examples:

\[
\mathbf{u}_{\mathrm{ref}}=\mathbf{u}(0),
\qquad
\mathbf{q}_{\mathrm{POD}}
=
\mathbf{V}^{\top}
\left(\mathbf{u}-\mathbf{u}_{\mathrm{ref}}\right),
\]

where `V` is obtained from a rank-two SVD of the shifted snapshots. This
makes the affine POD plane tilted rather than horizontal. Thus the `q`
shown in the animation labels denotes the POD coordinates; the `q_1` and
`q_2` used above denote the harmonic coordinates that generate the
benchmark.

The linear model misses the nonlinear coordinate. Piecewise linear
manifolds improve the local geometric approximation. The quadratic model
recovers the second harmonic, but it cannot reproduce the third harmonic.
A general nonlinear closure is not restricted to quadratic features and
can represent both contributions.

The notation used in the animations is:

- linear: `u_ref + V q`
- quadratic: `u_ref + V q + H h_2(q)`
- general nonlinear closure:
  `u_ref + V q + V_bar F(q, mu, t)`, with
  `F in {N, M, H}`

The general nonlinear animation retains two modes. It is method-neutral:
the learned map may be represented by ANN, RBF, or GPR.

The POD-AE animation trains a coefficient-space autoencoder with
\(n_{\mathrm{tot}}=3\) and \(n_z=2\). Its decoder defines the latent trial
manifold

\[
\widetilde{\mathbf u}
=
\mathbf u_{\mathrm{ref}}
+
\mathbf V_{\mathrm{tot}}\mathcal D_{\mathrm{AE}}(\mathbf z).
\]

Render it independently with:

```bash
python3 pod_ae_manifold.py
```

This produces `outputs/pod_ae_manifold.gif`.

The generic decoder visualization reuses the same controlled trajectory to
show a point on the nonlinear trial manifold, its local tangent plane, and the
two columns of the tangent basis. It does not introduce an encoder.

```bash
python3 generic_decoder_tangent.py
```

This produces:

- `outputs/generic_decoder_tangent.gif`
- `outputs/generic_decoder_tangent.png`
- `outputs/generic_decoder_tangent.pdf`
- `outputs/generic_decoder_tangent.svg`

## Aggressive compression and multiplicity

The Cases 1--3 animations retain only one POD mode:

\[
\dim(\mathbf V)=1,
\qquad
\dim(\overline{\mathbf V})=2.
\]

They compare:

\[
\begin{aligned}
\text{Case 1:}&\quad
\bar{\mathbf q}=\mathcal N(q),\\
\text{Case 2:}&\quad
\bar{\mathbf q}=\mathcal M(\boldsymbol\mu,t),\\
\text{Case 3:}&\quad
\bar{\mathbf q}=\mathcal H(q,\boldsymbol\mu,t).
\end{aligned}
\]

Case 1 exposes the multiplicity produced by the one-dimensional projection:
the same retained coordinate can correspond to different secondary
coordinates. Cases 2 and 3 add branch-identifying information. This argument
is independent of whether the deterministic regressor is ANN, RBF, or GPR.

Each slide can be rendered independently:

```bash
python3 case1_state_closure.py
python3 case2_parameter_time_closure.py
python3 case3_hybrid_closure.py
```

These commands produce:

- `outputs/case1_ann_rbf_gpr_state.gif`
- `outputs/case2_ann_rbf_gpr_parameter_time.gif`
- `outputs/case3_ann_rbf_gpr_hybrid.gif`

Render the complete series with:

```bash
cd /home/kratos/Documents/PhDThesis_Animations/Manifold_animations/harmonics_quadratic_gpr
python3 render_all.py
```

The GIF files are written to `outputs/`.
