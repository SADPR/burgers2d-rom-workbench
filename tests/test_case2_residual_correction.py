import numpy as np
import pytest

from burgers.case2_residual_correction import residual_krylov_space, solve_affine_step


def problem():
    rng = np.random.default_rng(31)
    v, _ = np.linalg.qr(rng.normal(size=(32, 10)))
    j = np.eye(32) + 0.15 * rng.normal(size=(32, 32))
    return j, v[:, :3], v[:, 3:], rng.normal(size=32)


def test_full_rank_recovers_full_linearized_lspg():
    j, vp, vs, f = problem()
    b = residual_krylov_space(j, vp, vs, f, vs.shape[1])
    np.testing.assert_allclose(b.T @ b, np.eye(7), atol=1e-12)
    v = np.column_stack((vp, vs))
    t = np.column_stack((vp, vs @ b))
    direct = v @ np.linalg.lstsq(j @ v, -f, rcond=None)[0]
    krylov = t @ np.linalg.lstsq(j @ t, -f, rcond=None)[0]
    np.testing.assert_allclose(krylov, direct, atol=1e-11)


def test_rank_one_is_schur_steepest_descent_and_reduces_residual():
    j, vp, vs, f = problem()
    a = j @ vp
    q = np.linalg.qr(a)[0]
    c = j @ vs - q @ (q.T @ (j @ vs))
    rhs = f - q @ (q.T @ f)
    g = c.T @ rhs
    expected = -g * (g @ g) / np.linalg.norm(c @ g)**2
    b = residual_krylov_space(j, vp, vs, f, 1)
    t = np.column_stack((vp, vs @ b))
    z = np.linalg.lstsq(j @ t, -f, rcond=None)[0]
    np.testing.assert_allclose(b @ z[3:], expected, atol=1e-12)
    assert np.linalg.norm(f + j @ t @ z) < np.linalg.norm(rhs)


def test_identity_breakdown_and_zero_rank():
    j, vp, vs, f = problem()
    assert residual_krylov_space(j, vp, vs, f, 0).shape == (7, 0)
    assert residual_krylov_space(np.eye(32), vp, vs, f, 7).shape == (7, 1)
    assert residual_krylov_space(j, vp, vs, np.zeros(32), 3).shape == (7, 0)
    with pytest.raises(ValueError):
        residual_krylov_space(j, vp, vs, f, -1)


def test_affine_step_recovers_linear_least_squares():
    j, vp, _, f = problem()
    z, _, _, _ = solve_affine_step(np.zeros(32), vp, np.zeros(3),
                                   lambda w: j @ w + f, lambda w: j)
    np.testing.assert_allclose(z, np.linalg.lstsq(j @ vp, -f, rcond=None)[0], atol=1e-12)


def test_rank_zero_matches_production_case2():
    import torch
    from burgers.core import make_2D_grid, get_ops, inviscid_burgers_res2D, inviscid_burgers_exact_jac2D
    from burgers.pod_ann_manifold import inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2

    class Tail(torch.nn.Module):
        def forward(self, x):
            return x[..., -1:] * torch.tensor([0.1, -0.2, 0.3])

    rng = np.random.default_rng(7)
    gridx, gridy = make_2D_grid(0, 1, 0, 1, 4, 4)
    v = np.linalg.qr(rng.normal(size=(32, 6)))[0]
    vp, vs = v[:, :3], v[:, 3:]
    uref, initial = np.ones(32), np.ones(32)
    dt, mu = 0.001, [1.5, 0.02]
    model = Tail()
    native, _ = inviscid_burgers_implicit2D_LSPG_pod_ann_2D_case2(
        gridx, gridy, initial, dt, 4, mu, model, None, vp, vs, u_ref=uref, min_delta=1e-2)
    dx, dy, jdx, jdy, eye = get_ops(gridx, gridy)
    q = np.zeros(3)
    states = [initial]
    for k in range(1, 5):
        tail = model(torch.tensor([[*mu, dt*k]], dtype=torch.float32)).numpy().reshape(-1)
        previous = states[-1]
        q, state, _, _ = solve_affine_step(
            uref + vs @ tail, vp, q,
            lambda w: inviscid_burgers_res2D(w, gridx, gridy, dt, previous, mu, dx, dy),
            lambda w: inviscid_burgers_exact_jac2D(w, dt, jdx, jdy, eye))
        states.append(state)
    np.testing.assert_allclose(np.column_stack(states), native, atol=2e-14)


def test_burgers_jacobian_action_matches_finite_difference():
    from burgers.core import make_2D_grid, get_ops, inviscid_burgers_res2D, inviscid_burgers_exact_jac2D
    rng = np.random.default_rng(11)
    gridx, gridy = make_2D_grid(0, 1, 0, 1, 4, 4)
    dx, dy, jdx, jdy, eye = get_ops(gridx, gridy)
    w = 1 + rng.random(32)
    d = rng.normal(size=32)
    residual = lambda x: inviscid_burgers_res2D(x, gridx, gridy, 0.05, w, [1.5, .02], dx, dy)
    fd = (residual(w + 1e-6*d) - residual(w - 1e-6*d)) / 2e-6
    action = inviscid_burgers_exact_jac2D(w, 0.05, jdx, jdy, eye) @ d
    np.testing.assert_allclose(fd, action, rtol=2e-8, atol=2e-9)


def test_affine_feedback_fit_and_tangent():
    from Project_YvonMaday.run_case2_affine_feedback import fit_feedback
    rng = np.random.default_rng(23)
    ep = rng.normal(size=(3, 40))
    gain = rng.normal(size=(7, 3))
    es = gain @ ep
    np.testing.assert_allclose(fit_feedback(ep, es, 0), gain, atol=1e-12)
    fitted = fit_feedback(ep, es, 0.01)
    lam = 0.01 * np.trace(ep @ ep.T) / 3
    np.testing.assert_allclose((fitted @ ep - es) @ ep.T + lam*fitted, 0, atol=1e-12)
    j, vp, vs, _ = problem()
    q, mp, ms = rng.normal(size=3), rng.normal(size=3), rng.normal(size=7)
    direct = vp @ q + vs @ (ms + gain @ (q - mp))
    affine = vs @ (ms - gain @ mp) + (vp + vs @ gain) @ q
    np.testing.assert_allclose(direct, affine, atol=1e-12)
    with pytest.raises(ValueError):
        fit_feedback(ep, es, -1)


def test_known_initial_state_is_independent_of_ann_tail(monkeypatch):
    import torch
    from burgers.core import make_2D_grid
    import Project_YvonMaday.run_case2_local_corrections as experiment
    rng = np.random.default_rng(83)
    v = np.linalg.qr(rng.normal(size=(32, 6)))[0]
    uref = np.ones(32)
    w0 = uref + v @ rng.normal(scale=0.05, size=6)
    gx, gy = make_2D_grid(0, 1, 0, 1, 4, 4)
    monkeypatch.setattr(experiment, "GRID_X", gx)
    monkeypatch.setattr(experiment, "GRID_Y", gy)
    monkeypatch.setattr(experiment, "W0", w0)
    monkeypatch.setattr(experiment, "DT", 0.001)

    class Prediction(torch.nn.Module):
        def forward(self, x):
            return torch.ones((x.shape[0], 6), dtype=torch.float32) * .2

    q, _ = experiment.rollout(Prediction(), v, uref, [1.5, .02], "baseline", 2, 3,
                              initialization="linear")
    np.testing.assert_allclose(uref + v @ q[:, 0], w0, atol=1e-13)


@pytest.mark.parametrize("case", [1, 3])
def test_nonlinear_known_initial_control_uses_correct_tangent(monkeypatch, case):
    import torch
    from burgers.core import make_2D_grid, get_ops, inviscid_burgers_res2D, inviscid_burgers_exact_jac2D
    import Project_YvonMaday.run_case2_local_reference_benchmarks as experiment
    rng = np.random.default_rng(93)
    v = np.linalg.qr(rng.normal(size=(200, 151)))[0]
    uref = np.ones(200)
    w0 = uref + v @ rng.normal(scale=0.002, size=151)
    gx, gy = make_2D_grid(0, 1, 0, 1, 10, 10)
    monkeypatch.setattr(experiment, "GRID_X", gx)
    monkeypatch.setattr(experiment, "GRID_Y", gy)
    monkeypatch.setattr(experiment, "W0", w0)
    monkeypatch.setattr(experiment, "DT", 0.001)
    gain = rng.normal(scale=.002, size=(141, 10)).astype(np.float32)

    class Closure(torch.nn.Module):
        def forward(self, x):
            return torch.tensor(gain) @ x[:10]

    q, _ = experiment.known_initial_nonlinear_rollout(Closure(), v, uref, [1.5, .02], case, steps=2)
    np.testing.assert_allclose(uref + v @ q[:, 0], w0, atol=1e-13)
    dx, dy, jdx, jdy, eye = get_ops(gx, gy)
    tangent = v[:, :10] + v[:, 10:] @ gain
    previous = w0
    primary = q[:10, 0]
    for k in range(1, 3):
        primary, previous, _, _ = solve_affine_step(
            uref, tangent, primary,
            lambda w: inviscid_burgers_res2D(w, gx, gy, .001, previous, [1.5, .02], dx, dy),
            lambda w: inviscid_burgers_exact_jac2D(w, .001, jdx, jdy, eye))
        np.testing.assert_allclose(q[:10, k], primary, atol=1e-6, rtol=1e-5)
