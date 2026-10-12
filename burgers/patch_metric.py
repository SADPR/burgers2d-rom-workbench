"""PSD residual metrics on fixed cell groups, fitted by adaptive-rank factors.

Training matches projected residuals. The objective is convex in the metrics,
but not in their factors. Numerical KKT checks always concern the original
convex problem on the supplied fixed groups, not mesh cardinality.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import minimize


def symmetric_indices(size):
    return np.triu_indices(size)


def svec(matrix):
    """Frobenius-isometric upper-triangle coordinates (sqrt(2) off diagonal)."""
    i, j = symmetric_indices(matrix.shape[-1])
    return matrix[..., i, j] * np.where(i == j, 1., np.sqrt(2.))


def smat(vector, size):
    i, j = symmetric_indices(size)
    matrix = np.zeros(vector.shape[:-1] + (size, size))
    entries = vector / np.where(i == j, 1., np.sqrt(2.))
    matrix[..., i, j] = entries
    matrix[..., j, i] = entries
    return matrix


def geometric_cell_groups(selected_cells, grid_shape, group_size=4):
    """Deterministic disjoint groups: smallest index followed by nearest cells."""
    cells = np.sort(np.asarray(selected_cells, dtype=int))
    if group_size < 1 or len(np.unique(cells)) != len(cells):
        raise ValueError('Require unique cells and a positive group size.')
    if cells.size == 0 or cells[0] < 0 or cells[-1] >= np.prod(grid_shape):
        raise ValueError('Cell indices outside the supplied mesh.')
    locations = np.column_stack(np.unravel_index(cells, grid_shape))
    remaining = np.ones(cells.size, dtype=bool)
    groups = []
    while remaining.any():
        seed = np.flatnonzero(remaining)[0]
        candidates = np.flatnonzero(remaining)
        distances = np.sum((locations[candidates] - locations[seed])**2, axis=1)
        order = np.lexsort((cells[candidates], distances))
        chosen = candidates[order[:group_size]]
        groups.append(cells[chosen])
        remaining[chosen] = False
    return groups


def regular_mesh_groups(grid_shape, side=2):
    """Partition the original mesh into disjoint square candidate patches."""
    if side < 1:
        raise ValueError('Patch side must be positive.')
    mesh = np.arange(np.prod(grid_shape)).reshape(grid_shape)
    return [mesh[y:y+side, x:x+side].ravel()
            for y in range(0, grid_shape[0], side)
            for x in range(0, grid_shape[1], side)]


def patch_correlations(residuals, tangents, fitting_residual, groups):
    """Exact directional correlations for every original-mesh candidate.

    S_P = sum_c sym((W_P h_c) R_P.T). The largest eigenvalue gives
    the largest initial decrease among unit-trace PSD additions.
    Equal-sized blocks permit batched contractions without a global design.
    """
    if len({len(group) for group in groups}) != 1:
        raise ValueError('The batched candidate scan requires equal-sized patches.')
    rows = tangents.shape[0]
    modes = tangents.shape[-1]
    h = np.asarray(fitting_residual).reshape(rows, modes)
    local = np.matmul(tangents.reshape(rows, -1, modes), h[:, :, None])[..., 0]
    local = local.reshape(residuals.shape)
    indices = np.asarray(groups)
    R = residuals[:, indices, :].reshape(rows, len(groups), -1)
    U = local[:, indices, :].reshape(rows, len(groups), -1)
    products = np.matmul(U.transpose(1, 2, 0), R.transpose(1, 0, 2))
    return .5*(products+products.transpose(0, 2, 1))


def greedy_patch_metrics(residuals, tangents, target, candidate_groups,
                         relative_tolerance=1e-5, initial_rank=2,
                         kkt_tolerance=1e-8, max_seconds=1800.,
                         progress=None, checkpoint=None):
    """Select patches from zero and refit all active PSD factors.

    Every candidate is scored; no existing cubature support or weights enter
    the fit. Selection resumes only after the active fit meets numerical KKT
    conditions. A runtime limit produces an explicitly incomplete checkpoint.
    """
    started = time.perf_counter()
    target = np.asarray(target, dtype=float)
    norm_b = float(np.linalg.norm(target))
    all_cells = np.arange(residuals.shape[1])
    groups, factors, dimensions, history = [], [], [], []
    available = np.ones(len(candidate_groups), dtype=bool)
    capacity = 256
    active = np.empty((target.size, capacity), order='F')
    gram = np.empty((capacity, capacity), order='F')
    width = 0
    residual = target.copy()
    last_fit = None
    stop_reason = 'candidate set exhausted'
    while available.any():
        if time.perf_counter()-started >= max_seconds:
            stop_reason = 'time budget reached; training tolerance not attained'
            break
        scan_started = time.perf_counter()
        if progress:
            progress(dict(stage='candidate_scan', iteration=len(groups),
                cells=sum(len(g) for g in groups), relative_training_error=float(np.linalg.norm(residual)/norm_b),
                training_seconds=time.perf_counter()-started))
        correlations = patch_correlations(residuals, tangents, residual, candidate_groups)
        values, vectors = np.linalg.eigh(correlations)
        scores = values[:, -1].copy()
        scores[~available] = -np.inf
        chosen = int(np.argmax(scores))
        if scores[chosen] <= 0:
            stop_reason = 'no PSD descent candidate; training tolerance not attained'
            break
        group = candidate_groups[chosen]
        columns, dims = assemble_group_design(residuals, tangents, all_cells, [group])
        new_width = width+columns.shape[1]
        if new_width > capacity:
            capacity = max(new_width, 2*capacity)
            new_active = np.empty((target.size, capacity), order='F')
            new_gram = np.empty((capacity, capacity), order='F')
            new_active[:, :width] = active[:, :width]
            new_gram[:width, :width] = gram[:width, :width]
            active, gram = new_active, new_gram
        active[:, width:new_width] = columns
        cross = active[:, :width].T @ columns
        gram[:width, width:new_width] = cross
        gram[width:new_width, :width] = cross.T
        gram[width:new_width, width:new_width] = columns.T @ columns
        count = min(initial_rank, max(1, int(np.count_nonzero(values[chosen] > 0))))
        amplitudes = values[chosen, -count:]
        L = vectors[chosen, :, -count:] * np.sqrt(amplitudes)[None, :]
        prediction = columns @ svec(L @ L.T)
        step = max(0., float(prediction @ residual)/float(prediction @ prediction))
        factors.append(np.sqrt(step)*L)
        groups.append(group.copy())
        dimensions.extend(dims)
        available[chosen] = False
        width = new_width
        def inner_progress(row):
            if progress:
                progress(dict(**row, selection_iteration=len(groups),
                    cells=sum(len(g) for g in groups), groups=len(groups),
                    training_seconds=time.perf_counter()-started))
        def inner_checkpoint(fitted, row):
            if checkpoint:
                checkpoint(groups, fitted, row)
        # Tiny Gram-vector products were faster with one BLAS thread in
        # matched refit measurements. Candidate scans and large refits keep
        # the process thread limit. The objective and tolerances are identical.
        from threadpoolctl import threadpool_limits
        refit_limit = 1 if width <= 1024 else None
        with threadpool_limits(limits=refit_limit):
            factors, last_fit = fit_adaptive_rank(active[:, :width], target, dimensions,
                relative_tolerance=relative_tolerance, initial_rank=initial_rank,
                kkt_tolerance=kkt_tolerance, stage_iterations=750,
                max_seconds=max(.01, max_seconds-(time.perf_counter()-started)),
                progress=inner_progress, checkpoint=inner_checkpoint,
                initial_factors=factors, gram=gram[:width, :width])
        last_fit['refit_blas_threads'] = refit_limit or 'process limit'
        coefficients = np.concatenate([svec(L @ L.T) for L in factors])
        residual = target-active[:, :width] @ coefficients
        error = float(np.linalg.norm(residual)/norm_b)
        if not np.isclose(error, last_fit['relative_training_error'], rtol=1e-8, atol=1e-12):
            raise RuntimeError('Scaled and original-row training errors differ.')
        row = dict(stage='selection', iteration=len(groups), cells=sum(len(g) for g in groups),
            groups=len(groups), relative_training_error=error,
            training_seconds=time.perf_counter()-started,
            candidate_scan_seconds=time.perf_counter()-scan_started-last_fit['fit_seconds'],
            selected_candidate=chosen, maximum_psd_correlation=float(scores[chosen]),
            last_refit={k:v for k, v in last_fit.items() if k != 'history'})
        history.append(row)
        if progress:
            progress(row)
        if checkpoint:
            checkpoint(groups, factors, row)
        if error <= relative_tolerance:
            stop_reason = 'training tolerance attained'
            break
        if not last_fit['numerical_kkt_satisfied']:
            stop_reason = last_fit['stop_reason']+'; further selection not performed'
            break
    if not groups:
        raise RuntimeError('No PSD patch selected.')
    selected = np.sort(np.concatenate(groups))
    report = dict(cells=selected.size, groups=len(groups),
        candidate_groups=len(candidate_groups), relative_training_error=float(np.linalg.norm(residual)/norm_b),
        tolerance=relative_tolerance, tolerance_met=bool(np.linalg.norm(residual)/norm_b <= relative_tolerance),
        training_seconds=time.perf_counter()-started, stop_reason=stop_reason,
        last_refit=last_fit, history=history,
        initialization='empty patch set; no previously trained support or weights',
        candidate_rule='maximum eigenvalue of the exact residual-space correlation matrix; all candidates scanned')
    return PatchResidualMetric(selected, groups, factors), report


def assemble_group_design(residuals, tangents, selected_cells, groups):
    """Rows stack configuration then reduced coordinate; blocks use svec(A).

    residuals: (configurations, selected cells, residual components).
    tangents:  (configurations, selected cells, residual components, modes).
    """
    residuals = np.asarray(residuals)
    tangents = np.asarray(tangents)
    cells = np.asarray(selected_cells)
    if residuals.shape != tangents.shape[:-1] or residuals.shape[1] != cells.size:
        raise ValueError('Incompatible local residual and test-basis arrays.')
    if not np.array_equal(cells, np.sort(cells)):
        raise ValueError('Selected cells must be sorted.')
    designs, dimensions = [], []
    for group in groups:
        positions = np.searchsorted(cells, group)
        if np.any(positions >= cells.size) or not np.array_equal(cells[positions], group):
            raise ValueError('Group contains an unselected cell.')
        r = residuals[:, positions].reshape(residuals.shape[0], -1)
        W = tangents[:, positions].reshape(tangents.shape[0], -1, tangents.shape[-1])
        d = r.shape[1]
        i, j = symmetric_indices(d)
        columns = (W[:, i, :] * r[:, j, None] + W[:, j, :] * r[:, i, None])
        columns /= np.where(i == j, 2., np.sqrt(2.))[None, :, None]
        designs.append(columns.transpose(0, 2, 1).reshape(-1, len(i)))
        dimensions.append(d)
    return np.ascontiguousarray(np.concatenate(designs, axis=1)), dimensions


class FactorFitProblem:
    """Block-scaled convex objective and its factorized analytic gradient."""
    def __init__(self, design, target, dimensions, gram=None):
        self.dimensions = list(map(int, dimensions))
        widths = [d*(d+1)//2 for d in self.dimensions]
        self.offsets = np.cumsum([0] + widths)
        design = np.asarray(design, dtype=float)
        self.target_norm = float(np.linalg.norm(target))
        if (design.shape != (len(target), self.offsets[-1])
                or self.target_norm <= 0 or not np.isfinite(design).all()):
            raise ValueError('Invalid training design or target.')
        self.scales = np.array([max(np.linalg.norm(design[:, a:b])/np.sqrt(b-a),
                                   np.finfo(float).tiny)
                                for a, b in zip(self.offsets[:-1], self.offsets[1:])])
        self.design = np.array(design, order='C', copy=True)
        for scale, a, b in zip(self.scales, self.offsets[:-1], self.offsets[1:]):
            self.design[:, a:b] /= scale
        self.target = np.asarray(target, dtype=float) / self.target_norm
        self.gram = None
        if gram is not None:
            column_scales = np.repeat(self.scales, widths)
            self.gram = np.asarray(gram)/column_scales[:, None]/column_scales[None, :]
        self.linear = self.design.T @ self.target
        self.evaluations = 0
        self.last_objective = np.inf

    def coefficients(self, factors):
        return np.concatenate([svec(L @ L.T) for L in factors])

    def unpack(self, vector, ranks):
        factors, offset = [], 0
        for d, rank in zip(self.dimensions, ranks):
            factors.append(vector[offset:offset+d*rank].reshape(d, rank))
            offset += d*rank
        if offset != vector.size:
            raise ValueError('Factor vector size mismatch.')
        return factors

    def evaluate_matrices(self, coefficients):
        residual = self.design @ coefficients - self.target
        gradient = self.design.T @ residual
        matrices = [smat(gradient[a:b], d) for d, a, b in
                    zip(self.dimensions, self.offsets[:-1], self.offsets[1:])]
        return residual, matrices

    def enable_qr(self):
        """Equivalent least-squares coordinates; no singular values discarded."""
        from scipy.linalg import qr
        from threadpoolctl import threadpool_limits
        started=time.perf_counter()
        # The large dense QR benefits from the process's reserved ten threads;
        # factor-vector iterations keep the active refit's existing limit.
        with threadpool_limits(limits=10):
            Q,self.qr_design=qr(self.design,mode='economic',check_finite=False)
            self.qr_target=Q.T @ self.target
            self.qr_orthogonal_residual=self.target-Q @ self.qr_target
        self.qr_seconds=time.perf_counter()-started

    def center_objective(self, factors):
        # Recenter the same quadratic at a feasible iterate. Removing its
        # constant prevents large residual errors from hiding tiny final
        # improvements in an ill-conditioned active-group refit.
        self.center_coefficients = self.coefficients(factors)
        residual = self.design @ self.center_coefficients-self.target
        self.center_gradient = self.design.T @ residual
        self.center_value = .5*float(residual @ residual)
        if hasattr(self,'qr_design'):
            self.qr_center_residual=self.qr_design @ self.center_coefficients-self.qr_target

    def objective_gradient(self, vector, ranks):
        factors = self.unpack(vector, ranks)
        coefficients = self.coefficients(factors)
        if hasattr(self,'qr_design'):
            delta=coefficients-self.center_coefficients
            change=self.qr_design @ delta
            gradient=self.qr_design.T @ (self.qr_center_residual+change)
            objective=float(self.qr_center_residual @ change+.5*(change @ change))
            gradients=[smat(gradient[a:b],d) for d,a,b in
                       zip(self.dimensions,self.offsets[:-1],self.offsets[1:])]
        elif self.gram is None:
            residual, gradients = self.evaluate_matrices(coefficients)
            objective = .5 * float(residual @ residual)
        else:
            center = getattr(self, 'center_coefficients', np.zeros_like(coefficients))
            delta = coefficients-center
            product = self.gram @ delta
            center_gradient = getattr(self, 'center_gradient', -self.linear)
            gradient = product+center_gradient
            objective = .5 * float(delta @ product)+float(delta @ center_gradient)
            gradients = [smat(gradient[a:b], d) for d, a, b in
                         zip(self.dimensions, self.offsets[:-1], self.offsets[1:])]
        self.evaluations += 1
        gradient = np.concatenate([(2*S @ L).ravel() for S, L in zip(gradients, factors)])
        self.last_objective = objective+getattr(self, 'center_value', 0.) if self.gram is not None or hasattr(self,'qr_design') else objective
        return objective, gradient

    def initial_factors(self, rank):
        _, gradients = self.evaluate_matrices(np.zeros(self.design.shape[1]))
        factors = []
        for d, S in zip(self.dimensions, gradients):
            values, vectors = np.linalg.eigh(-S)
            count = min(rank, d)
            directions = vectors[:, -count:]
            amplitudes = np.maximum(values[-count:], 1e-8 * max(1., np.max(abs(values))))
            factors.append(directions * np.sqrt(amplitudes)[None, :])
        prediction = self.design @ self.coefficients(factors)
        scale = max(1e-8, float(prediction @ self.target) / max(float(prediction @ prediction), 1e-300))
        return [np.sqrt(scale) * L for L in factors]

    def diagnostics(self, factors, kkt_tolerance):
        residual, gradients = self.evaluate_matrices(self.coefficients(factors))
        eigensystems = [np.linalg.eigh(S) for S in gradients]
        min_eigenvalue = float(min(e[0][0] for e in eigensystems))
        gradient_scale = max(1., np.sqrt(sum(np.linalg.norm(S)**2 for S in gradients)))
        matrix_scale = max(1., np.sqrt(sum(np.linalg.norm(L @ L.T)**2 for L in factors)))
        dual_violation = max(0., -min_eigenvalue)/gradient_scale
        complementarity = np.sqrt(sum(np.linalg.norm(S @ (L @ L.T))**2
                                     for S, L in zip(gradients, factors)))/matrix_scale/gradient_scale
        factor_gradient = np.sqrt(sum(np.linalg.norm(2*S @ L)**2
                                      for S, L in zip(gradients, factors)))
        report = dict(relative_training_error=float(np.linalg.norm(residual)),
            minimum_dual_eigenvalue=min_eigenvalue,
            dual_psd_violation=float(dual_violation),
            complementarity_residual=float(complementarity),
            factor_gradient_norm=float(factor_gradient),
            numerical_kkt_satisfied=bool(dual_violation <= kkt_tolerance
                                        and complementarity <= kkt_tolerance),
            kkt_tolerance=kkt_tolerance, ranks=[L.shape[1] for L in factors],
            factor_parameters=sum(L.size for L in factors),
            evaluations=self.evaluations)
        return report, residual, eigensystems

    def expand_ranks(self, factors, residual, eigensystems, kkt_tolerance):
        """Add negative-gradient eigenvectors; use an exact LS line minimizer."""
        directions, added = [], []
        for group, (d, L, (values, vectors)) in enumerate(
                zip(self.dimensions, factors, eigensystems)):
            if values[0] < -kkt_tolerance:
                v = vectors[:, 0]
                amplitude = -float(values[0])
                directions.append(svec(amplitude * np.outer(v, v)))
                added.append((group, amplitude, v))
            else:
                directions.append(np.zeros(d*(d+1)//2))
        if not added:
            return None, []
        prediction = self.design @ np.concatenate(directions)
        derivative = float(residual @ prediction)
        denominator = float(prediction @ prediction)
        if derivative >= 0 or denominator <= 0:
            raise RuntimeError('Rank expansion did not give a descent direction.')
        step = -derivative/denominator
        expanded = [L.copy() for L in factors]
        for group, amplitude, v in added:
            extended = np.column_stack((expanded[group], np.sqrt(step*amplitude)*v))
            if extended.shape[1] > extended.shape[0]:
                # A square factor can still have deficient numerical rank.
                # A PSD descent addition remains valid at full column count.
                # Thin SVD reexpresses the same LL^T with at most d columns;
                # every singular value is retained (no rank truncation).
                left, singular, _ = np.linalg.svd(extended, full_matrices=False)
                expanded[group] = left*singular[None, :]
            else:
                expanded[group] = extended
        return expanded, [item[0] for item in added]

    def physical_factors(self, factors):
        return [np.sqrt(self.target_norm/scale)*L for L, scale in zip(factors, self.scales)]


def fit_adaptive_rank(design, target, dimensions, relative_tolerance=1e-5,
                      initial_rank=2, kkt_tolerance=1e-8, stage_iterations=750,
                      max_seconds=600., progress=None, checkpoint=None,
                      initial_factors=None, gram=None):
    """Fit on fixed groups; report attained tolerance and numerical KKT separately.

    A time limit returns the best PSD feasible factors, explicitly marked as
    incomplete. Ranks can grow to full dimension. An L-BFGS termination alone
    is never reported as a certificate for the original convex problem.
    """
    started = time.perf_counter()
    problem = FactorFitProblem(design, target, dimensions, gram=gram)
    if initial_factors is None:
        factors = problem.initial_factors(initial_rank)
    else:
        factors = [np.sqrt(scale/problem.target_norm)*L
                   for L, scale in zip(initial_factors, problem.scales)]
        if len(factors) != len(dimensions):
            raise ValueError('Initial factors do not match all active groups.')
    history, total_iterations = [], 0
    class BudgetReached(Exception):
        pass
    best_factors = [L.copy() for L in factors]
    best_error = np.inf
    stop_reason = 'time budget reached; optimality not certified'
    stage=0
    while time.perf_counter()-started < max_seconds:
        stage += 1
        # If the first factor stage needs further refinement, QR avoids
        # squaring the conditioning of the same least-squares operator.
        # This applies to rectangular and square factors alike.
        if stage >= 2 and not hasattr(problem,'qr_design'):
            if progress:
                progress(dict(stage='qr_stabilization',rank_stage=stage,
                    iteration=total_iterations,fit_seconds=time.perf_counter()-started,
                    coefficient_columns=design.shape[1]))
            problem.enable_qr()
        ranks = [L.shape[1] for L in factors]
        vector = np.concatenate([L.ravel() for L in factors])
        problem.center_objective(factors)
        stage_started = time.perf_counter()
        last_reported = stage_started-10.
        accepted = 0
        def callback(x):
            nonlocal best_error, best_factors, last_reported, accepted
            accepted += 1
            current = problem.unpack(x, ranks)
            estimated_error = np.sqrt(max(0., 2*problem.last_objective))
            if estimated_error < best_error:
                best_error = estimated_error
                best_factors = [L.copy() for L in current]
            now = time.perf_counter()
            if now-last_reported >= 10.:
                value, _, _ = problem.diagnostics(current, kkt_tolerance)
                row = dict(stage='factor_fit', rank_stage=stage,
                    iteration=total_iterations+accepted, fit_seconds=now-started, **value)
                if progress:
                    progress(row)
                if checkpoint:
                    checkpoint(problem.physical_factors(best_factors), row)
                last_reported = now
            if now-started >= max_seconds:
                raise BudgetReached
        try:
            result = minimize(problem.objective_gradient, vector, args=(ranks,), jac=True,
                method='L-BFGS-B', callback=callback,
                options=dict(maxiter=stage_iterations, ftol=0., gtol=1e-11,
                             maxls=40, maxcor=30))
            factors = problem.unpack(result.x, ranks)
            optimizer_status = str(result.message)
            total_iterations += int(result.nit)
        except BudgetReached:
            total_iterations += accepted
            factors = best_factors
            optimizer_status = 'time budget reached'
        value, residual, eigensystems = problem.diagnostics(factors, kkt_tolerance)
        if value['relative_training_error'] < best_error:
            best_error = value['relative_training_error']
            best_factors = [L.copy() for L in factors]
        row = dict(stage='rank_check', rank_stage=stage, iteration=total_iterations,
                   fit_seconds=time.perf_counter()-started, optimizer_status=optimizer_status,
                   stage_seconds=time.perf_counter()-stage_started, **value)
        history.append(row)
        if progress:
            progress(row)
        if checkpoint:
            checkpoint(problem.physical_factors(best_factors), row)
        if value['relative_training_error'] <= relative_tolerance:
            stop_reason = 'training tolerance attained'
            break
        if value['numerical_kkt_satisfied']:
            stop_reason = 'numerical fixed-group optimum above training tolerance'
            break
        if time.perf_counter()-started >= max_seconds:
            stop_reason = 'time budget reached; optimality not certified'
            break
        expanded, changed = problem.expand_ranks(factors, residual, eigensystems, kkt_tolerance)
        if expanded is not None:
            factors = expanded
            row['expanded_groups'] = changed
        elif optimizer_status.startswith('CONVERGENCE') and stage > 1:
            # Retry from a newly centered quadratic with the same factors,
            # objective and KKT tolerances; no matrix directions are frozen.
            continue
    # Use the last accepted optimizer result. Its true residual was checked
    # above; normal-product cancellation is never used for tolerance attainment.
    value, _, _ = problem.diagnostics(factors, kkt_tolerance)
    report = dict(**value, stop_reason=stop_reason,
        tolerance=relative_tolerance, tolerance_met=value['relative_training_error'] <= relative_tolerance,
        fit_seconds=time.perf_counter()-started, iterations=total_iterations,
        group_dimensions=list(dimensions), history=history,
        evaluation_kernel='QR' if hasattr(problem,'qr_design') else 'Gram',
        qr_seconds=getattr(problem,'qr_seconds',0.),
        rank_limit='full residual-block dimension; no fixed rank truncation',
        optimality_scope='Numerical KKT checks of the convex fit on these fixed groups only.')
    return problem.physical_factors(factors), report


@dataclass
class PatchResidualMetric:
    """Apply group factors to residual/JV rows ordered [all u cells, all v]."""
    selected_cells: np.ndarray
    groups: list
    factors: list

    def __post_init__(self):
        self.selected_cells = np.sort(np.asarray(self.selected_cells, dtype=int))
        if len(self.groups) != len(self.factors):
            raise ValueError('Every group needs a factor.')
        joined = np.concatenate(self.groups)
        if not np.array_equal(np.sort(joined), self.selected_cells):
            raise ValueError('Groups must partition the selected cells exactly.')
        n = self.selected_cells.size
        self.input_rows = 2*n
        self.output_rows = sum(L.shape[1] for L in self.factors)
        self._rows = []
        for cells, L in zip(self.groups, self.factors):
            positions = np.searchsorted(self.selected_cells, cells)
            rows = np.column_stack((positions, n+positions)).reshape(-1)
            if L.shape[0] != rows.size or not np.isfinite(L).all():
                raise ValueError('Invalid group factor.')
            self._rows.append(rows)

    def apply(self, values):
        values = np.asarray(values)
        if values.ndim not in (1, 2) or values.shape[0] != self.input_rows:
            raise ValueError('Sampled residual rows do not match the patch metric.')
        return np.concatenate([L.T @ values[rows] for rows, L in zip(self._rows, self.factors)], axis=0)

    def save(self, path):
        np.savez(path, selected_cells=self.selected_cells, group_count=len(self.groups),
                 **{f'group_{i}':g for i, g in enumerate(self.groups)},
                 **{f'factor_{i}':L for i, L in enumerate(self.factors)})

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            count = int(data['group_count'])
            return cls(data['selected_cells'].copy(),
                       [data[f'group_{i}'].copy() for i in range(count)],
                       [data[f'factor_{i}'].copy() for i in range(count)])
