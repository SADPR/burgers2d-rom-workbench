# WCCM 2026 conference slides

This document is organized as copy-ready input for a LaTeX-to-image service.
It follows the nomenclature and formatting conventions in
`Project_YvonMaday/Results_Paper/presentation_revision_after_slide_41.md`.

Use the same layout as the nonlinear-PROM harmonic-example slides:
formula image on the left, animation or static visual on the right, and the
same visual scale as slides 42--43.

## Nomenclature

- Retained reduced coordinates: `\mathbf q`.
- Discarded coordinates: `\overline{\mathbf q}`.
- Retained basis and complement: `\mathbf V`, `\overline{\mathbf V}`.
- Closure map:
  `\mathcal N\in\{\mathrm{ANN},\mathrm{RBF},\mathrm{GPR}\}`.
- Local-chart quantities use superscript `(i)`:
  `\mathbf u_{\mathrm{ref}}^{(i)}`,
  `\mathbf V^{(i)}`,
  `\overline{\mathbf V}^{(i)}`,
  `\mathbf q^{(i)}`,
  `\mathcal N_i`.

The WCCM-specific message is:

```text
global learned closure is useful -> local linear charts explain why locality helps
-> local nonlinear closure charts combine locality with learned out-of-plane closure
```

---

## Slide sequence starting at slide 44

### Slide 44 -- Nonlinear PROMs: Closure-based nonlinear manifold

```latex
\begin{aligned}
&\text{\bfseries Closure-based nonlinear manifold approximation}\\[0.6em]
&\hspace{1.2em}
\widetilde{\mathbf u}(\mathbf q;\boldsymbol\mu,t)
=
\mathbf u_{\mathrm{ref}}
+
\mathbf V\mathbf q
+
\overline{\mathbf V}
\mathcal N(\mathbf q,\boldsymbol\mu,t)
\\[1.3em]
&\text{\bfseries Learned closure of the discarded coordinates}\\[0.6em]
&\hspace{3.6em}
\overline{\mathbf q}
\approx
\mathcal N(\mathbf q,\boldsymbol\mu,t),
\qquad
\mathcal N
\in
\left\{
\mathrm{ANN},\mathrm{RBF},\mathrm{GPR}
\right\}
\\[1.3em]
&\text{\bfseries Tangent induced by the closure}\\[0.6em]
&\hspace{3.8em}
\mathbf T_q
=
\frac{\partial\widetilde{\mathbf u}}{\partial\mathbf q}
=
\mathbf V
+
\overline{\mathbf V}
\frac{\partial\mathcal N}{\partial\mathbf q}
\\[1.3em]
&\text{\bfseries Representation and projection hint}\\[0.6em]
&\hspace{1.0em}
\text{\itshape The learned map can reproduce both the second and third}
\\[0.2em]
&\hspace{1.5em}
\text{\itshape harmonics, and also changes the local directions used later.}
\end{aligned}
```

Visual suggestion: use
`../Results_Paper/harmonic_manifold_closure_animations/outputs/general_ann_rbf_gpr_closure.gif`.
Keep the exact visual scale used in slides 42--43. The intended visual
progression is:

```text
linear misses -> quadratic improves -> learned ANN/RBF/GPR closure captures
```

Speaker note: this is still the global closure story. Do not introduce local
charts yet; use this slide to establish the closure form and the induced
tangent basis.

### Slide 45 -- Working example: finite-deformation RVE training trajectories

```latex
\begin{aligned}
&\text{\bfseries Working example: RVE training trajectories}\\[0.6em]
&\hspace{1.0em}
\boldsymbol\mu
=
\left(E_{xx},E_{yy},G_{xy}\right)
\in
\mathcal P\subset\mathbb R^3
\\[1.1em]
&\text{\bfseries Ten training paths through the parameter domain}\\[0.6em]
&\hspace{1.0em}
\left\{
\boldsymbol\mu_j^{(k)}
\right\}_{j=0}^{N_k},
\qquad
k=1,\ldots,10
\\[1.1em]
&\text{\bfseries Full-order snapshots on each path}\\[0.6em]
&\hspace{1.0em}
\mathbf u_h\!\left(\boldsymbol\mu_j^{(k)}\right)
\quad\longrightarrow\quad
\text{training data for reduced coordinates and closures}
\\[1.1em]
&\text{\bfseries Why show this before locality?}\\[0.6em]
&\hspace{1.0em}
\text{\itshape The data are not isolated points: they are organized paths}\\[0.2em]
&\hspace{1.7em}
\text{\itshape in a three-dimensional parameter space, with strongly}\\[0.2em]
&\hspace{1.7em}
\text{\itshape nonlinear deformations along the RVE response.}
\end{aligned}
```

Visual suggestion: use
`/home/kratos/ML_assisted_CLs_clean/RVE_homogenization_NeoHookean_using_Kratos/WCCM2026_animations/training_trajectories_5_representative_rve_animation.gif`.
Place the GIF as the right-side visual, or use it full-width if the formula
becomes too dense. The animation walks through five representative trajectories slowly, while
keeping the full ten-trajectory training design visible in the background; it
then leaves a final rotating view of the completed parameter-domain paths.

Speaker note: this is the concrete working example before the local-nonlinear
motivation. Say: these ten trajectories define the training data, and the RVE
response is already geometrically rich before we start discussing global versus
local closure maps.

### Slide 46 -- PROM-ANN coordinate model: from macro strain paths to q

```latex
\begin{aligned}
&\text{\bfseries PROM-ANN coordinate model}\\[0.6em]
&\hspace{0.4em}
\mathbf u_j
=
\mathbf u_h(\boldsymbol\mu_j),
\qquad
\boldsymbol\mu_j=(E_{xx},E_{yy},G_{xy})_j
\\[1.0em]
&\text{\bfseries Retained coordinates aligned with the macro strains}\\[0.6em]
&\hspace{0.8em}
\min_{\mathbf T}
\left\|\mathbf Q\mathbf T^T-\mathbf M\right\|_F
\qquad\Longrightarrow\qquad
\mathbf A
\\[0.4em]
&\hspace{0.8em}
\mathbf M_j=(E_{xx},E_{yy},G_{xy})_j
\\[0.8em]
&\hspace{0.8em}
\mathbf q^0(\boldsymbol\mu)
=
[\boldsymbol\mu,1]\mathbf B_{\rm aff},
\qquad
\mathbf q\in\mathbb R^3
\\[1.0em]
&\text{\bfseries PROM-ANN closure of the discarded coordinates}\\[0.6em]
&\hspace{0.8em}
\overline{\mathbf q}
\approx
\mathcal N_\theta(\mathbf q),
\qquad
\mathcal N_\theta:\mathbb R^3\rightarrow\mathbb R^{36}
\\[1.0em]
&\text{\bfseries Decoder used online}\\[0.6em]
&\hspace{0.8em}
\widetilde{\mathbf u}(\mathbf q,\boldsymbol\mu)
\approx
\mathbf u_{\rm aff}(\boldsymbol\mu)
+
\mathbf V\mathbf A\mathbf q
+
\overline{\mathbf V}\mathcal N_\theta(\mathbf q)
\\[0.6em]
&\hspace{0.8em}
\mathbf u_j
\approx
\mathbf u_{\rm aff}(\boldsymbol\mu_j)
+
\mathbf V\mathbf A\mathbf q_j
+
\overline{\mathbf V}\overline{\mathbf q}_j,
\qquad
\overline{\mathbf q}_j\in\mathbb R^{36}
\end{aligned}
```

Visual suggestion: use
`/home/kratos/ML_assisted_CLs_clean/RVE_homogenization_NeoHookean_using_Kratos/WCCM2026_animations/mu_to_q_prom_ann_pipeline.gif`.
This is a didactic visual of the PROM-ANN coordinate model: first show the
truncated POD basis \(\mathbf V_{\rm tot}\in\mathbb R^{N\times39}\), split as
\(\mathbf V\in\mathbb R^{N\times3}\) and
\(\overline{\mathbf V}\in\mathbb R^{N\times36}\). The three primary modes
match the three-dimensional parameter space. Then show the first three POD
coordinates as a curved manifold, which is not an ideal regression input. Show
the objective: find a subset or linear combinations of
coordinates whose \(\mathbf q=(q_1,q_2,q_3)\) space looks like a structured
grid. Finally, \(\mathbf q\) is used as the three-dimensional input of
\(\mathcal N_\theta\), whose output is the real discarded-coordinate vector
\(\overline{\mathbf q}\in\mathbb R^{36}\).

Speaker note: keep this high-level. The audience only needs the pipeline:
macro strain paths -> retained coordinates \(\mathbf q\) -> ANN closure
\(\overline{\mathbf q}=\mathcal N_\theta(\mathbf q)\) -> decoder with
\(\mathbf u_{\rm aff}\), \(\mathbf V\), and \(\overline{\mathbf V}\).

### Slide 47 -- Nonlinear PROMs: Global ANN/RBF/GPR closure on a complex manifold

```latex
\begin{aligned}
&\text{\bfseries Same closure form on a more demanding trajectory}\\[0.6em]
&\hspace{1.2em}
\widetilde{\mathbf u}_{\mathrm{glob}}(\mathbf q;\boldsymbol\mu,t)
=
\mathbf u_{\mathrm{ref}}
+
\mathbf V\mathbf q
+
\overline{\mathbf V}
\mathcal N_{\mathrm{glob}}(\mathbf q,\boldsymbol\mu,t)
\\[1.2em]
&\text{\bfseries Global learned closure}\\[0.6em]
&\hspace{3.0em}
\overline{\mathbf q}
\approx
\mathcal N_{\mathrm{glob}}(\mathbf q,\boldsymbol\mu,t),
\qquad
\mathcal N_{\mathrm{glob}}
\in
\left\{
\mathrm{ANN},\mathrm{RBF},\mathrm{GPR}
\right\}
\\[1.2em]
&\text{\bfseries Trial manifold}\\[0.6em]
&\hspace{2.2em}
\mathcal M_{\mathrm{glob}}
=
\left\{
\mathbf u_{\mathrm{ref}}
+
\mathbf V\mathbf q
+
\overline{\mathbf V}
\mathcal N_{\mathrm{glob}}(\mathbf q,\boldsymbol\mu,t)
\right\}
\\[1.2em]
&\text{\bfseries What the global closure gives}\\[0.6em]
&\hspace{1.0em}
\text{\itshape A single learned map captures most of the nonlinear geometry,}
\\[0.2em]
&\hspace{1.7em}
\text{\itshape but very sharp local features can still be smoothed globally.}
\end{aligned}
```

Visual suggestion: use
`../Results_Paper/harmonic_manifold_closure_animations/outputs/local_prom_ann_two_bases_global_ann_rbf_gpr.gif`.
This animation uses a GPR-like smooth interpolant only as one concrete drawing
device; the slide label and formula are intentionally method-neutral:
ANN/RBF/GPR.

Speaker note: keep the tone fair. The global learned closure is good; the
motivation is not that it fails, but that locality may reduce the burden on a
single global map.

### Slide 48 -- Nonlinear PROMs: Local linear POD charts

```latex
\begin{aligned}
&\text{\bfseries Local linear manifold approximation}\\[0.6em]
&\hspace{1.7em}
\widetilde{\mathbf u}^{(i)}(\mathbf q^{(i)})
=
\mathbf u_{\mathrm{ref}}^{(i)}
+
\mathbf V^{(i)}\mathbf q^{(i)},
\qquad
i=1,\ldots,K
\\[1.2em]
&\text{\bfseries Piecewise affine trial manifold}\\[0.6em]
&\hspace{2.0em}
\mathcal M_{\mathrm{loc}}^{\mathrm{lin}}
=
\bigcup_{i=1}^{K}
\left\{
\mathbf u_{\mathrm{ref}}^{(i)}
+
\mathbf V^{(i)}\mathbf q^{(i)}
\right\},
\qquad
\dim\mathbf V^{(i)}=2
\\[1.2em]
&\text{\bfseries Chart selection}\\[0.6em]
&\hspace{3.3em}
i=i(\mathbf q,\boldsymbol\mu,t)
\quad
\text{or}
\quad
i=i(t)
\\[1.2em]
&\text{\bfseries Accuracy--organization tradeoff}\\[0.6em]
&\hspace{1.0em}
\text{\itshape Local POD charts follow the geometry accurately, but the}
\\[0.2em]
&\hspace{1.7em}
\text{\itshape representation needs several charts and a selection rule.}
\end{aligned}
```

Visual suggestion: use
`../Results_Paper/harmonic_manifold_closure_animations/outputs/local_prom_ann_two_bases_local_linear.gif`.
The current animation uses `K=6` square local rank-two affine charts. Keep the
same formula-left/GIF-right layout. This slide is a geometric bridge, not an
attack on local linear POD.

Speaker note: this slide motivates locality. It should say: local linear is
honest and accurate, but it pays with chart count and chart management.

---

## 2D Burgers results: global and local HPROMs ($N_c=3$)

The following slides use only the three-chart local campaign reported in
`Results_Paper/main.tex`. They exclude POD--AE, POD--NN--ROM, and POD--DL--ROM.

### Slide 49 -- 2D Burgers: Problem and discretization

```latex
\begin{aligned}
&\text{\bfseries Two-dimensional parametric inviscid Burgers problem}\\[0.5em]
&\hspace{0.8em}
\frac{\partial\mathbf U}{\partial t}
+\frac{\partial\mathbf F(\mathbf U)}{\partial x}
+\frac{\partial\mathbf G(\mathbf U)}{\partial y}
=\mathbf S(x;\boldsymbol\mu),
\qquad \mathbf U=\begin{bmatrix}u_x\\u_y\end{bmatrix}
\\[1.0em]
&\text{\bfseries Parameter-dependent forcing and boundary condition}\\[0.5em]
&\hspace{0.8em}
\mathbf S(x;\boldsymbol\mu)=
\begin{bmatrix}0.02\exp(\mu_2x)\\0\end{bmatrix},
\qquad u_x(0,y,t;\boldsymbol\mu)=\mu_1
\\[1.0em]
&\text{\bfseries Domain and HDM}\\[0.5em]
&\hspace{0.8em}
\Omega=[0,100]^2,
\quad t\in[0,25],
\quad \boldsymbol\mu\in[4.25,5.50]\times[0.015,0.030]
\\[0.4em]
&\hspace{0.8em}
250\times250\text{ finite-volume cells},\quad N=125{,}000,
\quad \Delta t=0.05,\quad 501\text{ stored states.}
\end{aligned}
```

Visual: place `burgers_global_local_animations/outputs/hdm_centerline_cuts_mu1.gif`
on the right. It shows the HDM surface and the two centerline cuts at the first
test parameter. Keep the formula-left/GIF-right layout used in slides 44--48.

Speaker note: define the two cuts once here; all subsequent global--local
comparisons use those same cuts and the same vertical scale.

### Slide 50 -- 2D Burgers: Baseline training and evaluation

```latex
\begin{aligned}
&\text{\bfseries HDM training parameters}\\[0.5em]
&\hspace{1.8em}
\mathcal D_{\mathrm{train}}=\{3\times3\text{ structured parameter grid}\}
\\[1.0em]
&\text{\bfseries Snapshot set used for all global and local bases}\\[0.5em]
&\hspace{1.8em}N_{\mathrm{snap}}=9\times501=4509
\\[1.0em]
&\text{\bfseries Fixed in-domain test parameters}\\[0.5em]
&\hspace{1.8em}\boldsymbol\mu^{(1)}=(4.56,0.019),
\qquad \boldsymbol\mu^{(2)}=(4.75,0.020),\\[0.25em]
&\hspace{1.8em}\boldsymbol\mu^{(3)}=(5.19,0.026)
\\[1.0em]
&\text{\bfseries Important distinction}\\[0.5em]
&\hspace{1.0em}\text{\itshape All three tests lie inside the training box,
but none is a training-grid point.}\\[-0.1em]
&\hspace{1.0em}\text{\itshape This campaign has neither a separate
verification point nor an extrapolation point.}
\end{aligned}
```

Visual: use `burgers_global_local_animations/outputs/parameter_domain_test_points.png`
on the right. It retains the black-grid/red-star Paris-conference style, but
shows only the three test points of this study.

### Slide 51 -- 2D Burgers: Global versus local linear HPROM

```latex
\begin{aligned}
&\text{\bfseries One global linear trial manifold}\\[0.5em]
&\hspace{1.7em}\widetilde{\mathbf u}_{\mathrm{glob}}(\mathbf q)
=\mathbf u_{\mathrm{ref}}+\mathbf V\mathbf q,
\qquad n_q=96
\\[1.1em]
&\text{\bfseries Three local linear trial manifolds}\\[0.5em]
&\hspace{1.7em}\widetilde{\mathbf u}^{(i)}_{\mathrm{loc}}(\mathbf q^{(i)})
=\mathbf u_{\mathrm{ref}}^{(i)}+\mathbf V^{(i)}\mathbf q^{(i)},
\qquad i=1,2,3
\\[1.1em]
&\text{\bfseries Comparison question}\\[0.5em]
&\hspace{1.0em}\text{\itshape Does localization improve the HDM trajectory}
\\[-0.1em]
&\hspace{1.0em}\text{\itshape while reducing the online cost?}
\end{aligned}
```

Visual: use `burgers_global_local_animations/outputs/global_vs_local_hprom.gif`
full width or as the right-side visual. It shows the HDM, global HPROM, and
local HPROM at all three test parameters with synchronized time.

Speaker note: this is the cleanest first comparison: same linear structure,
but three smaller charts and a chart-selection step.

### Slide 52 -- 2D Burgers: Global versus local HQPROM

```latex
\begin{aligned}
&\text{\bfseries Global quadratic manifold}\\[0.5em]
&\hspace{1.0em}\widetilde{\mathbf u}_{\mathrm{glob}}(\mathbf q)
=\mathbf u_{\mathrm{ref}}+\mathbf V\mathbf q
+\mathbf H(\mathbf q\otimes\mathbf q),\qquad n_q=39
\\[1.1em]
&\text{\bfseries Local quadratic manifolds}\\[0.5em]
&\hspace{1.0em}\widetilde{\mathbf u}^{(i)}_{\mathrm{loc}}(\mathbf q^{(i)})
=\mathbf u_{\mathrm{ref}}^{(i)}+\mathbf V^{(i)}\mathbf q^{(i)}
+\mathbf H^{(i)}(\mathbf q^{(i)}\otimes\mathbf q^{(i)}),
\\[-0.2em]
&\hspace{6.2em}i=1,2,3,\qquad n_q^{(i)}=11\text{--}15
\\[1.1em]
&\text{\bfseries What changes}\\[0.5em]
&\hspace{1.0em}\text{\itshape Locality reduces the primary coordinate}
\\[-0.1em]
&\hspace{1.0em}\text{\itshape dimension before the quadratic lifting is formed.}
\end{aligned}
```

Visual: use `burgers_global_local_animations/outputs/global_vs_local_hqprom.gif`.
It retains the same layout as slide 51, so the audience compares the effect of
quadratic lifting rather than learning a new plot format.

### Slide 53 -- 2D Burgers: Global versus local HPROM--GPR

```latex
\begin{aligned}
&\text{\bfseries Global learned closure}\\[0.5em]
&\hspace{1.0em}\widetilde{\mathbf u}_{\mathrm{glob}}(\mathbf q)
=\mathbf u_{\mathrm{ref}}+\mathbf V\mathbf q
+\overline{\mathbf V}\,\mathcal N_{\mathrm{GPR}}(\mathbf q),
\qquad n_q=20,\quad\bar n=131
\\[1.1em]
&\text{\bfseries Local learned closures}\\[0.5em]
&\hspace{1.0em}\widetilde{\mathbf u}_{\mathrm{loc}}^{(i)}(\mathbf q^{(i)})
=\mathbf u_{\mathrm{ref}}^{(i)}+\mathbf V^{(i)}\mathbf q^{(i)}
+\overline{\mathbf V}^{(i)}\mathcal N_{\mathrm{GPR},i}(\mathbf q^{(i)}),
\\[-0.2em]
&\hspace{6.2em}i=1,2,3,\qquad n_q=10,\quad\bar n=60\text{--}97
\\[1.1em]
&\text{\bfseries Point of the comparison}\\[0.5em]
&\hspace{1.0em}\text{\itshape The closure acts only inside the active chart,}
\\[-0.1em]
&\hspace{1.0em}\text{\itshape reducing the burden on one global nonlinear map.}
\end{aligned}
```

Visual: use `burgers_global_local_animations/outputs/global_vs_local_hprom_gpr.gif`.
This is the nonlinear-closure counterpart of slides 51--52; it uses GPR only,
because that is the learned closure exposed by `main.tex`.

### Slide 54 -- 2D Burgers: Local model family (optional)

```latex
\begin{aligned}
&\text{\bfseries Same three-chart organization, three manifold choices}\\[0.6em]
&\hspace{0.7em}\mathbf u_{\mathrm{lin}}^{(i)}
=\mathbf u_{\mathrm{ref}}^{(i)}+\mathbf V^{(i)}\mathbf q^{(i)}
\\[0.7em]
&\hspace{0.7em}\mathbf u_{\mathrm{quad}}^{(i)}
=\mathbf u_{\mathrm{ref}}^{(i)}+\mathbf V^{(i)}\mathbf q^{(i)}
+\mathbf H^{(i)}(\mathbf q^{(i)}\otimes\mathbf q^{(i)})
\\[0.7em]
&\hspace{0.7em}\mathbf u_{\mathrm{GPR}}^{(i)}
=\mathbf u_{\mathrm{ref}}^{(i)}+\mathbf V^{(i)}\mathbf q^{(i)}
+\overline{\mathbf V}^{(i)}\mathcal N_{\mathrm{GPR},i}(\mathbf q^{(i)})
\\[1.1em]
&\text{\bfseries Controlled comparison}\\[0.5em]
&\hspace{1.0em}\text{\itshape Keep the local partition fixed; change only}
\\[-0.1em]
&\hspace{1.0em}\text{\itshape the manifold inside each chart.}
\end{aligned}
```

Visual: use `burgers_global_local_animations/outputs/local_hprom_hqprom_gpr_mu2.gif`.
It compares the three local methods against the HDM at
\(\boldsymbol\mu^{(2)}\). Hide this slide if time is tight: slides 51--53
already establish the main global-to-local story.

### Slide 55 -- 2D Burgers: Accuracy--cost and configuration summary

```latex
\begin{aligned}
&\text{\bfseries Comparison protocol}\\[0.5em]
&\hspace{1.0em}\text{mean and maximum relative state error over }
\boldsymbol\mu^{(1)},\boldsymbol\mu^{(2)},\boldsymbol\mu^{(3)}
\\[0.9em]
&\text{\bfseries Cost measures}\\[0.5em]
&\hspace{1.0em}n_q,\quad \bar n\ \text{or quadratic features},\quad N_e,
\quad \text{speedup relative to the HDM}
\\[0.9em]
&\text{\bfseries Takeaway}\\[0.5em]
&\hspace{1.0em}\text{\itshape Global and local models are compared at equal HDM}
\\[-0.1em]
&\hspace{1.0em}\text{\itshape resolution; locality changes both the trial manifold and ECSW mesh.}
\end{aligned}
```

Visual: use the single slide-ready image
`burgers_global_local_animations/outputs/global_local_accuracy_cost_summary_merged.png`
full width. It combines the accuracy--speedup plot with the LaTeX-rendered
configuration/performance table from `Results_Paper/main.tex`. All displayed local
results use $N_c=3$.

---

### Slide 56 -- Conclusions and future work

```latex
\begin{aligned}
&\text{\bfseries Conclusions}\\[0.4em]
&\hspace{0.6em}\text{\bfseries RVE:}\quad
\text{Global HPROM--ANN is already an effective nonlinear model.}
\\[0.55em]
&\hspace{0.6em}\text{\bfseries Locality:}\quad
\text{a complementary alternative, not a requirement.}
\\[0.55em]
&\hspace{0.6em}\text{\bfseries Burgers:}\quad
\text{local charts offer competitive accuracy--cost tradeoffs.}
\\[0.9em]
&\text{\bfseries Next}\\[0.4em]
&\hspace{0.6em}\text{Plastic / hysteretic RVEs.}\\[0.35em]
&\hspace{0.6em}\text{Hypersonic local nonlinear ROMs.}\\[0.35em]
&\hspace{0.6em}\text{Systematic PANN comparison: ICKAN / ICNN.}
\end{aligned}
```

Visual: place rve_icnn_dhprom_animations/outputs/rve_fom_dhprom_ann_icnn_stage10.gif
on the right half. Keep the compact aligned block on the left.

Speaker note: the right-hand animation is a same-path preliminary ICNN
comparison. D-HPROM--ANN remains more accurate in global stress, while the
ICNN already reproduces the stress response reasonably and motivates the
systematic physics-aware comparison proposed here.

---

## Appendix

### Appendix A -- Global ECSW meshes

Use `burgers_global_local_animations/outputs/appendix_global_ecsw_meshes.png`.
It places the HPROM, HQPROM, and HPROM--GPR meshes on one slide and reports
their respective $N_e$ values.

### Appendix B -- Local ECSW meshes ($N_c=3$)

Use `burgers_global_local_animations/outputs/appendix_local_ecsw_meshes.png`.
It places the Local HPROM, Local HQPROM, and Local HPROM--GPR meshes on one
slide and reports their respective $N_e$ values.
