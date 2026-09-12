import numpy as np
import pytest

from burgers.case2_hyperreduction import SampledBurgers, sampled_affine_step, cell_moments
from burgers.case2_residual_correction import residual_krylov_space, solve_affine_step
from burgers.core import (make_2D_grid, get_ops, inviscid_burgers_res2D,
                          inviscid_burgers_exact_jac2D)


def problem():
    rng = np.random.default_rng(43)
    gx, gy = make_2D_grid(0, 2, 0, 3, 6, 6)
    v = np.linalg.qr(rng.normal(size=(72, 10)))[0]
    state, previous = 1 + rng.random(72), 1 + rng.random(72)
    return rng, gx, gy, v, state, previous


def test_sampled_residual_jacobian_and_adjoint_are_exact_restrictions():
    rng, gx, gy, v, state, previous = problem()
    weights = np.zeros(36)
    weights[[0, 1, 7, 8, 13, 18, 23, 27, 29]] = rng.uniform(.1, 8, 9)
    mesh = SampledBurgers(gx, gy, .02, weights)
    f, j = mesh.step_operators(previous[mesh.state_indices], [1.5, .02])
    dx, dy, jdx, jdy, eye = get_ops(gx, gy)
    full_f = inviscid_burgers_res2D(state, gx, gy, .02, previous, [1.5, .02], dx, dy)
    full_j = inviscid_burgers_exact_jac2D(state, .02, jdx, jdy, eye)
    local = state[mesh.state_indices]
    np.testing.assert_allclose(f(local), mesh.sqrt_weights * full_f[mesh.residual_indices], atol=1e-14)
    action = mesh.sqrt_weights[:, None] * (full_j @ v)[mesh.residual_indices]
    np.testing.assert_allclose(j(local) @ v[mesh.state_indices], action, atol=1e-14)
    direction = rng.normal(size=local.size)
    fd = (f(local + 1e-6*direction) - f(local - 1e-6*direction)) / 2e-6
    np.testing.assert_allclose(fd, j(local) @ direction, rtol=1e-7, atol=2e-9)
    x = rng.normal(size=action.shape[0])
    np.testing.assert_allclose(v[mesh.state_indices].T @ (j(local).T @ x), action.T @ x, atol=1e-14)


@pytest.mark.parametrize("rank", [0, 1, 3, 7])
def test_full_mesh_recovers_prom_step(rank):
    _, gx, gy, v, state, previous = problem()
    mesh = SampledBurgers(gx, gy, .02, np.ones(36))
    vp, vs = v[:, :3], v[:, 3:]
    q = np.array([.1, .2, -.1])
    offset = state - vp @ q
    actual, b, sampled_state, _, _ = sampled_affine_step(
        mesh, vp, vs, offset, previous, [1.5, .02], q, rank)
    dx, dy, jdx, jdy, eye = get_ops(gx, gy)
    f = lambda w: inviscid_burgers_res2D(w, gx, gy, .02, previous, [1.5, .02], dx, dy)
    j = lambda w: inviscid_burgers_exact_jac2D(w, .02, jdx, jdy, eye)
    full_b = residual_krylov_space(j(state), vp, vs, f(state), rank)
    tangent = np.column_stack((vp, vs @ full_b))
    expected, full_state, _, _ = solve_affine_step(offset, tangent, np.r_[q, np.zeros(full_b.shape[1])], f, j)
    np.testing.assert_allclose(sampled_state, full_state, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(vp @ actual[:3] + vs @ b @ actual[3:], tangent @ expected, atol=1e-11)


def test_weighted_krylov_recovers_full_weighted_linearized_solve():
    rng, gx, gy, v, state, previous = problem()
    weights = rng.uniform(.1, 10, 36)
    weights[[2, 4, 6, 11]] = 0
    mesh = SampledBurgers(gx, gy, .02, weights)
    f, j = mesh.step_operators(previous[mesh.state_indices], [1.5, .02])
    local_v, local = v[mesh.state_indices], state[mesh.state_indices]
    a = j(local) @ local_v
    b = residual_krylov_space(j(local), local_v[:, :3], local_v[:, 3:], f(local), 7)
    transform = np.zeros((10, 3 + b.shape[1]))
    transform[:3, :3] = np.eye(3)
    transform[3:, 3:] = b
    direct = np.linalg.lstsq(a, -f(local), rcond=None)[0]
    reduced = transform @ np.linalg.lstsq(a @ transform, -f(local), rcond=None)[0]
    np.testing.assert_allclose(reduced, direct, atol=1e-11)


def test_cell_moments_preserve_gradient_gram_and_energy():
    rng, _, _, v, state, _ = problem()
    transform = rng.normal(size=(10, 4))
    blocks = cell_moments(v, state, transform)
    assert [b.shape for b in blocks] == [(36, 10), (36, 10), (36, 1)]
    for block in blocks:
        np.testing.assert_allclose(np.linalg.norm(block), 1., atol=1e-14)
    gradient = v[:36] * state[:36, None] + v[36:] * state[36:, None]
    np.testing.assert_allclose(blocks[0].sum(axis=0) * np.linalg.norm(gradient), v.T @ state)
    jt = v @ transform
    i, j = np.triu_indices(4)
    raw_gram = jt[:36, i]*jt[:36, j] + jt[36:, i]*jt[36:, j]
    raw_gram[:, i != j] *= np.sqrt(2.)
    expected = (jt.T @ jt)[i, j]
    expected[i != j] *= np.sqrt(2.)
    np.testing.assert_allclose(blocks[1].sum(axis=0)*np.linalg.norm(raw_gram), expected)
    raw_energy = state[:36]**2 + state[36:]**2
    np.testing.assert_allclose(blocks[2].sum()*np.linalg.norm(raw_energy), state @ state)


@pytest.mark.parametrize("weights", [np.zeros(36), -np.ones(36), np.full(36, np.nan), np.ones(29)])
def test_invalid_rules_rejected(weights):
    _, gx, gy, _, _, _ = problem()
    with pytest.raises(ValueError):
        SampledBurgers(gx, gy, .02, weights)


@pytest.mark.parametrize("rank", [0, 3])
def test_full_mesh_trajectory_recovers_own_predecessor_prom(monkeypatch, rank):
    import torch
    import Project_YvonMaday.run_case2_hyperreduction as hyper
    import Project_YvonMaday.run_case2_local_corrections as full

    rng = np.random.default_rng(112)
    gx, gy = make_2D_grid(0, 1, 0, 1, 10, 10)
    v = np.linalg.qr(rng.normal(size=(200, 151)))[0]
    uref = np.ones(200)
    initial = uref + v @ rng.normal(scale=.002, size=151)
    for module in (hyper, full):
        monkeypatch.setattr(module, "W0", initial)
        monkeypatch.setattr(module, "DT", .001)
        monkeypatch.setattr(module, "GRID_X", gx)
        monkeypatch.setattr(module, "GRID_Y", gy)

    class Predictor(torch.nn.Module):
        def forward(self, x):
            return .01 * x[:, -1:] * torch.arange(151, dtype=x.dtype)[None, :]

    model = Predictor()
    mesh = SampledBurgers(gx, gy, .001, np.ones(100))
    qfull, _ = full.rollout(model, v, uref, [1.5, .02], "baseline" if rank == 0 else "krylov3",
                            4, 10, initialization="linear")
    qhyper, _ = hyper.hyper_rollout(model, v, uref, [1.5, .02], mesh, rank, 4)
    np.testing.assert_allclose(qhyper, qfull, atol=2e-11, rtol=1e-9)
