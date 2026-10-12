"""Positive semidefinite residual-block weights for LSPG hyperreduction.

For each cell e, A_e is a fixed 2 x 2 PSD matrix in residual space:
    Phi(q) = 1/2 sum_e r_e(q).T A_e r_e(q).
Its gradient is sum W_e.T A_e r_e. This is not modal BM weighting.

Training selects cells by PSD directional correlations and jointly refits all
three symmetric entries of every selected matrix with PSD cone constraints.
This greedy method is not a global minimum-cardinality solver.
"""
import time
import numpy as np


def residual_metric_factors(matrices):
    """Return B_e with B_e.T B_e = A_e; permit rank-deficient PSD blocks."""
    matrices = np.asarray(matrices, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (2, 2):
        raise ValueError('Expected (cells, 2, 2) residual metric matrices.')
    if not np.all(np.isfinite(matrices)):
        raise ValueError('Residual metric matrices must be finite.')
    scale = np.maximum(1., np.max(np.abs(matrices), axis=(1, 2)))
    if np.any(np.max(np.abs(matrices - matrices.transpose(0, 2, 1)), axis=(1, 2)) > 1e-12 * scale):
        raise ValueError('Residual metric matrices must be symmetric.')
    eigenvalues, vectors = np.linalg.eigh(matrices)
    if np.any(eigenvalues[:, 0] < -1e-12 * scale):
        raise ValueError('Residual metric matrices must be positive semidefinite.')
    return np.sqrt(np.maximum(eigenvalues, 0.))[:, :, None] * vectors.transpose(0, 2, 1)


def apply_residual_metric(factors, values):
    """Apply B to vectors or Jacobian rows ordered [all u, all v]."""
    values = np.asarray(values)
    n = factors.shape[0]
    if values.ndim not in (1, 2) or values.shape[0] != 2 * n:
        raise ValueError('Residual/Jacobian dimensions do not match the metric.')
    shape = (n,) + (1,) * (values.ndim - 1)
    return np.concatenate((
        factors[:, 0, 0].reshape(shape) * values[:n] + factors[:, 0, 1].reshape(shape) * values[n:],
        factors[:, 1, 0].reshape(shape) * values[:n] + factors[:, 1, 1].reshape(shape) * values[n:],
    ), axis=0)


def metric_training_prediction(blocks, matrices):
    """blocks = (G11, G12, G22), where G12 includes both cross products."""
    return (blocks[0] @ matrices[:, 0, 0]
            + blocks[1] @ matrices[:, 0, 1]
            + blocks[2] @ matrices[:, 1, 1])


def project_soc_blocks(values):
    """Euclidean projection onto a product of 3D second-order cones.

    A = [[t+z, b], [b, t-z]] is PSD exactly when t >= hypot(z, b).
    These three coordinates leave every symmetric matrix entry free.
    """
    values = np.asarray(values, dtype=float).reshape(-1, 3)
    result = values.copy()
    radius = np.linalg.norm(values[:, 1:], axis=1)
    zero = radius <= -values[:, 0]
    result[zero] = 0.
    middle = ~(zero | (radius <= values[:, 0]))
    t = .5 * (radius[middle] + values[middle, 0])
    result[middle, 0] = t
    result[middle, 1:] *= (t / radius[middle])[:, None]
    return result


def refit_psd_matrices(training, target, initial, gram=None, threads=10):
    """Joint convex least-squares refit with one full 2x2 PSD block per cell.

    The three coordinates (t, z, b) represent A_e=[[t+z,b],[b,t-z]].
    All selected entries are free under their PSD constraints. A correction
    x=center+delta is only a change of variables in this same problem.
    Numerical retries change the center or the KKT factorization, retaining
    the objective, feasible matrices and solver tolerances.
    """
    import clarabel
    from scipy import sparse
    training = np.asarray(training, dtype=float)
    target = np.asarray(target, dtype=float)
    initial = np.asarray(initial, dtype=float).reshape(-1)
    if training.ndim != 2 or training.shape[1] % 3 or initial.size != training.shape[1]:
        raise ValueError('Require three training coefficients per selected cell.')
    if target.shape != (training.shape[0],):
        raise ValueError('Target and training rows do not match.')
    if not np.all(np.isfinite(initial)) or not np.allclose(
            initial.reshape(-1, 3), project_soc_blocks(initial), rtol=1e-12, atol=1e-13):
        raise ValueError('Initial residual metrics must be PSD.')
    if threads < 1:
        raise ValueError('Require a positive thread count.')
    if gram is None:
        gram = training.T @ training
    if gram.shape != (initial.size, initial.size):
        raise ValueError('Gram matrix and training columns do not match.')
    initial_residual = target - training @ initial
    error_limit = (np.linalg.norm(initial_residual) * (1 + 1e-8)
                   + 1e-12 * np.linalg.norm(target))
    n = initial.size
    upper = sparse.triu(sparse.csc_matrix(gram), format='csc')
    constraint = -sparse.eye(n, format='csc')
    cones = [clarabel.SecondOrderConeT(3) for _ in range(n // 3)]
    center = initial.copy()
    center_error = np.linalg.norm(initial_residual)
    attempts = []
    for backend in ('faer', 'faer', 'qdldl'):
        current_residual = target - training @ center
        linear = -(training.T @ current_residual)
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_threads = int(threads)
        settings.direct_solve_method = backend
        settings.max_iter = 300
        settings.tol_gap_abs = 1e-13
        settings.tol_gap_rel = 1e-11
        settings.tol_feas = 1e-11
        settings.reduced_tol_gap_abs = 1e-11
        settings.reduced_tol_gap_rel = 1e-9
        settings.reduced_tol_feas = 1e-9
        # Standard KKT stabilization is retained; no objective ridge is added.
        solver = clarabel.DefaultSolver(upper, linear, constraint, center, cones, settings)
        solution = solver.solve()
        status = str(solution.status)
        attempt = dict(status=status, iterations=int(solution.iterations),
            solve_seconds=float(solution.solve_time),
            primal_residual=float(solution.r_prim), dual_residual=float(solution.r_dual),
            duality_gap=float(solution.obj_val - solution.obj_val_dual),
            direct_solver=str(solver.get_info().linsolver.name),
            solver_threads=int(solver.get_info().linsolver.threads))
        attempts.append(attempt)
        delta = np.asarray(solution.x)
        if delta.shape != center.shape or not np.all(np.isfinite(delta)):
            attempt['rejection'] = 'nonfinite or invalid correction'
            continue
        raw = (center + delta).reshape(-1, 3)
        feasible = project_soc_blocks(raw)
        correction_norm = np.linalg.norm(feasible - raw)
        attempt['psd_projection_roundoff'] = float(correction_norm)
        if correction_norm > 1e-8 * max(1., np.linalg.norm(raw)):
            attempt['rejection'] = 'significant PSD infeasibility'
            continue
        x = feasible.reshape(-1)
        residual = target - training @ x
        error = np.linalg.norm(residual)
        if not np.isfinite(error) or error > error_limit:
            attempt['rejection'] = 'original least-squares error increased'
            continue
        if status in ('Solved', 'AlmostSolved'):
            gradient = -(training.T @ residual)
            spectral_scale = max(1., float(np.max(np.sum(np.abs(gram), axis=1))))
            mapped = x - project_soc_blocks(x - gradient / spectral_scale).reshape(-1)
            report = dict(attempt,
                solve_seconds=sum(a['solve_seconds'] for a in attempts),
                projected_gradient_norm=float(np.linalg.norm(mapped) * spectral_scale),
                attempts=attempts, retry_count=len(attempts) - 1)
            return x, residual, report
        # A failed iterate is only a feasible center for another solve. Its
        # status is never accepted as a successful fit. Every retry solves
        # the same original least-squares problem with unchanged tolerances.
        if error < center_error:
            center = x
            center_error = error
    detail = '; '.join(f"{a['direct_solver']}: {a['status']}"
                       f" (primal={a['primal_residual']:.3e}, dual={a['dual_residual']:.3e})"
                       for a in attempts)
    raise RuntimeError(f'Full PSD refit did not solve after equivalent retries: {detail}')


def greedy_psd_metric_weights(blocks, target, relative_tolerance=1e-5,
                              progress=None, threads=10, checkpoint=None):
    """Greedy cell selection with joint full-PSD refits of selected matrices.

    Candidate scores are the largest eigenvalues of the residual correlation
    matrices, as in the rank-one prototype. A selected cell contributes all
    three symmetric coefficients immediately. Every refit frees all of them,
    subject only to PSD. Unselected cell matrices remain zero.

    No SVD truncation or cell budget is used. The relative stopping test is
    always evaluated with original training rows. Small cone-solver errors
    are checked and recorded separately from the training tolerance.
    """
    if len(blocks) != 3 or any(g.shape != blocks[0].shape for g in blocks):
        raise ValueError('Three equally sized training matrices are required.')
    m, ne = blocks[0].shape
    target = np.asarray(target,dtype=float)
    if target.shape != (m,) or not 0 < relative_tolerance < 1:
        raise ValueError('Invalid target shape or tolerance.')
    norm_b = np.linalg.norm(target)
    if not np.isfinite(norm_b) or norm_b == 0:
        raise ValueError('Target must have finite nonzero norm.')
    b = target/norm_b
    residual = b.copy()
    selected, scales = [], []
    available = np.ones(ne,dtype=bool)
    coefficients = np.empty(0)
    capacity = 3 * min(64,ne)
    active = np.empty((m,capacity),order='F')
    gram = np.empty((capacity,capacity),order='F')
    history = []
    started = time.perf_counter()
    relative_error = 1.
    reason = 'tolerance'
    cumulative_refit = 0.
    max_refit_projected_gradient = 0.
    while len(selected) < ne and relative_error > relative_tolerance:
        c11,c12,c22 = (g.T @ residual for g in blocks)
        scores = .5*(c11+c22+np.hypot(c11-c22,c12))
        scores[~available] = -np.inf
        cell = int(np.argmax(scores))
        if not np.isfinite(scores[cell]) or scores[cell] <= 0:
            reason = 'no_positive_unselected_correlation'
            break
        available[cell] = False
        raw = np.column_stack((blocks[0][:,cell]+blocks[2][:,cell],
                               blocks[0][:,cell]-blocks[2][:,cell],
                               blocks[1][:,cell]))
        scale = np.linalg.norm(raw)
        if scale == 0:
            continue
        columns = raw/scale
        n = len(coefficients)
        if n+3 > capacity:
            new_capacity = min(3*ne,capacity+3*64)
            new_active = np.empty((m,new_capacity),order='F')
            new_gram = np.empty((new_capacity,new_capacity),order='F')
            new_active[:,:n],new_gram[:n,:n] = active[:,:n],gram[:n,:n]
            active,gram = new_active,new_gram
            capacity = new_capacity
        cross = active[:,:n].T @ columns
        active[:,n:n+3] = columns
        gram[:n,n:n+3],gram[n:n+3,:n] = cross,cross.T
        gram[n:n+3,n:n+3] = columns.T @ columns
        selected.append(cell);scales.append(scale)
        initial = np.concatenate((coefficients,np.zeros(3)))
        coefficients,residual,refit = refit_psd_matrices(active[:,:n+3],b,initial,
            gram=gram[:n+3,:n+3],threads=threads)
        relative_error = float(np.linalg.norm(residual))
        cumulative_refit += refit['solve_seconds']
        max_refit_projected_gradient = max(max_refit_projected_gradient,refit['projected_gradient_norm'])
        if checkpoint is not None:
            checkpoint(np.asarray(selected), np.asarray(scales), coefficients.copy(), norm_b)
        row = dict(iteration=len(selected), cells=len(selected),
            free_coefficients=len(coefficients), relative_training_error=relative_error,
            selection_seconds=time.perf_counter()-started,
            conic_refit_seconds=cumulative_refit, joint_refits=len(selected),
            last_refit=refit)
        history.append(row)
        if progress is not None:
            progress(row)
        if len(selected)%10 == 0 or relative_error<=relative_tolerance:
            print(f'[Full PSD metric] cells={len(selected)}, free_coefficients={len(coefficients)}, '
                  f'relative_error={relative_error:.9e}, refit={refit["status"]}',flush=True)
    matrices = np.zeros((ne,2,2))
    values = coefficients.reshape(-1,3)*(norm_b/np.asarray(scales))[:,None]
    for cell,(t,z,cross) in zip(selected,values):
        matrices[cell] = [[t+z,cross],[cross,t-z]]
    predicted = metric_training_prediction(blocks,matrices)
    final_error = float(np.linalg.norm(target-predicted)/norm_b)
    residual_metric_factors(matrices)
    if reason == 'tolerance' and final_error > relative_tolerance:
        raise RuntimeError('Saved full matrices did not attain the training tolerance.')
    report = dict(relative_training_error=final_error, tolerance=relative_tolerance,
        tolerance_met=bool(final_error<=relative_tolerance), stop_reason=reason,
        iterations=len(selected), free_coefficients=len(coefficients),
        cells=int(np.count_nonzero(np.trace(matrices,axis1=1,axis2=2)>0)),
        selected_cells=len(selected), selection_seconds=time.perf_counter()-started,
        conic_refit_seconds=cumulative_refit, joint_refits=len(selected),
        max_refit_projected_gradient=max_refit_projected_gradient,history=history)
    return matrices,report
