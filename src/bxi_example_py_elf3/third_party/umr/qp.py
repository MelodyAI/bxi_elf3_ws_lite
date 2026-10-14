"""Solve UMR's bounded quadratic step using the existing SciPy runtime."""
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize


def solve_qp_step(J, residual, damping, lower, upper, *, ineq_A=None,
                  ineq_b=None, ineq_soft_costs=None, l2_step_limits=None):
    n = J.shape[1]
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    hessian = J.T @ J + float(damping) * np.eye(n)
    linear = J.T @ residual
    original_constraints = []
    linear_rows, linear_bounds = [], []
    if ineq_A is not None and len(ineq_A):
        if ineq_soft_costs is not None and np.any(ineq_soft_costs > 0):
            raise ValueError("The bundled ELF3 profile requires hard ground constraints")
        ineq_A = np.asarray(ineq_A, dtype=np.float64)
        ineq_b = np.asarray(ineq_b, dtype=np.float64)
        original_constraints.append(dict(type="ineq", fun=lambda x: ineq_b - ineq_A @ x,
                                         jac=lambda x: -ineq_A))
        linear_rows.append(ineq_A)
        linear_bounds.append(ineq_b)
    for ids, radius, _label in l2_step_limits or []:
        ids = np.asarray(ids, dtype=np.int64)
        def value(x, ids=ids, radius=radius):
            return radius * radius - x[ids] @ x[ids]
        def jacobian(x, ids=ids):
            jac = np.zeros(n)
            jac[ids] = -2 * x[ids]
            return jac
        original_constraints.append(dict(type="ineq", fun=value, jac=jacobian))
    # Scale the objective for SLSQP; costs span several orders of magnitude.
    scale = max(float(np.max(np.diag(hessian))), 1.0)
    hessian /= scale
    linear /= scale
    factor = cholesky(hessian, lower=True)
    whitened_linear = solve_triangular(factor, linear, lower=True)
    unconstrained = solve_triangular(factor.T, -whitened_linear, lower=False)

    def feasible(x, tolerance=2e-6):
        return (np.isfinite(x).all() and np.all(x >= lower - tolerance) and
                np.all(x <= upper + tolerance) and
                all(np.min(c["fun"](x)) >= -tolerance for c in original_constraints))

    if feasible(unconstrained, tolerance=1e-12):
        return np.clip(unconstrained, lower, upper)

    # z = L.T @ x makes the quadratic Hessian identity. In the common case of
    # active ground constraints, SLSQP then solves the same QP in 2 iterations
    # rather than spending dozens learning the original ill-conditioned metric.
    transform = solve_triangular(factor.T, np.eye(n), lower=False)
    upper_mask, lower_mask = np.isfinite(upper), np.isfinite(lower)
    if np.any(upper_mask):
        linear_rows.append(np.eye(n)[upper_mask])
        linear_bounds.append(np.asarray(upper)[upper_mask])
    if np.any(lower_mask):
        linear_rows.append(-np.eye(n)[lower_mask])
        linear_bounds.append(-np.asarray(lower)[lower_mask])
    constraints = []
    if linear_rows:
        matrix = np.concatenate(linear_rows) @ transform
        bounds = np.concatenate(linear_bounds)
        constraints.append(dict(type="ineq", fun=lambda z: bounds - matrix @ z,
                                jac=lambda z: -matrix))
    for ids, radius, _label in l2_step_limits or []:
        selected = transform[np.asarray(ids, dtype=np.int64)]
        def value(z, selected=selected, radius=radius):
            x = selected @ z
            return radius * radius - x @ x
        def jacobian(z, selected=selected):
            return -2 * selected.T @ (selected @ z)
        constraints.append(dict(type="ineq", fun=value, jac=jacobian))
    result = minimize(lambda z: .5 * z @ z + whitened_linear @ z, -whitened_linear,
                      jac=lambda z: z + whitened_linear, constraints=constraints,
                      method="SLSQP", options=dict(ftol=1e-11, maxiter=150))
    step = transform @ result.x
    if result.success and feasible(step):
        return np.clip(step, lower, upper)

    # Retain the original coordinate solve as a fallback for difficult cone steps.
    result = minimize(lambda x: .5 * x @ hessian @ x + linear @ x,
                      np.clip(unconstrained, lower, upper), jac=lambda x: hessian @ x + linear,
                      bounds=list(zip(lower, upper)), constraints=original_constraints,
                      method="SLSQP", options=dict(ftol=1e-9, maxiter=150))
    if not result.success or not feasible(result.x):
        raise RuntimeError(f"UMR QP failed: {result.message}")
    return np.clip(result.x, lower, upper)
