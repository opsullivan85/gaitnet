"""A batched QP solver for thousands of robots at once.

Solves, independently for each of N robots,

    minimise  1/2 x' P x + q' x        subject to   l <= (I_B (x) C) x <= u

with the operator-splitting (ADMM) algorithm of OSQP [1], which is what the CPU
controller's `mpc_osqp.cc` calls into. Three things make it worth rewriting here rather
than calling a solver per robot:

- Every robot's problem has the same sparsity and the same size, so one batched
  factorisation and a fixed number of batched triangular solves cover the whole batch.
  Nothing branches on a robot's data, so the GPU never diverges.
- The constraint matrix is block diagonal with the same small block shape repeated once
  per (horizon step, leg) - a friction pyramid on that foot's three force components.
  Storing the blocks instead of the assembled matrix turns the constraint work from the
  largest read in the iteration into a rounding error, which matters because the
  iteration is memory bound, not compute bound.
- The scaling and step-size heuristics below are per robot but branch-free, so the batch
  gets OSQP's robustness without its per-problem control flow.

That robustness is not optional here. The MPC's cost leaves four of its thirteen states
unweighted, so P has a wide near-null space and is regularised only by a tiny penalty on
the forces themselves; the minimiser is well defined but the problem is badly
conditioned, and plain ADMM on the raw data converges towards it far too slowly to be
useful. Equilibration is what makes a few tens of iterations enough.

Accuracy is still traded for throughput deliberately: a fixed iteration budget with a
warm start from the previous solve, rather than an exact active-set solve that would
need an unpredictable number of iterations and a per-robot branch. `QpSolution` carries
the residuals so a caller can check.

[1] Stellato, Banjac, Goulart, Bemporad and Boyd, "OSQP: an operator splitting solver
    for quadratic programs", Mathematical Programming Computation 12(4), 2020.
    https://osqp.org/docs/solver/index.html - this follows section 3 (the reduced KKT
    form and the relaxed iteration), the modified Ruiz equilibration of section 5.1,
    and the step-size update of section 5.2, but not the polishing step, which needs a
    per-problem decision about which constraints are active.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_TINY = 1e-12


@dataclass(frozen=True)
class AdmmSettings:
    """Tuning for `solve`. The defaults follow OSQP's except for the iteration budget."""

    iterations: int = 80
    """Iterations per solve. Fixed, so the batch never diverges. Warm started from the
    previous solve, the worst step of a scripted episode lands within about half a
    percent of the CPU controller's torques; 100 iterations roughly halves that again
    for about a tenth more time, and 60 is three times worse, mostly in a transient
    after a foot changes state."""
    rho: float = 0.1
    """Initial step size on the inequality constraints."""
    equality_rho_scale: float = 1e3
    """How much larger the step size is on a constraint whose bounds have closed up.
    Every swing leg contributes twelve of those (its forces are pinned to zero), so this
    is not an edge case here - it is most of the constraint set."""
    sigma: float = 1e-6
    """Proximal regularisation on the primal step, which keeps the factorisation
    positive definite whatever P does."""
    relaxation: float = 1.6
    """Over-relaxation factor. 1.0 is the plain iteration."""
    scaling_iterations: int = 3
    """Ruiz equilibration passes before the first iteration. Zero disables scaling,
    which on this problem costs about three orders of magnitude of accuracy; past three
    passes nothing more is bought, and each one costs a pass over the Hessian."""
    rho_update_interval: int | None = 20
    """Re-balance the step size against the residuals this often, re-inverting when it
    does. None keeps the initial step size, which on this problem costs two orders of
    magnitude of accuracy - equilibration alone is not enough. Unlike OSQP this fires on
    a fixed schedule rather than when a residual ratio is exceeded, so the batch stays
    in lock step. Each update costs one more inversion, so the interval and the
    iteration budget should be chosen together."""
    equality_tolerance: float = 1e-9
    """Bound gap at or below which a constraint counts as an equality."""
    convergence_tolerance: float = 1e-3
    """Residual, in force units (N), that `check_every` tests against."""
    check_every: int | None = None
    """Test convergence this often and stop the whole batch early once every robot has
    converged. Each test synchronises with the host, so the default is not to: inside a
    rollout the fixed budget is both faster and reproducible."""


@dataclass
class QpSolution:
    """What `solve` found, in the units it was handed."""

    primal: torch.Tensor
    """(N, n) decision variables."""
    dual: torch.Tensor
    """(N, B, m) dual variables, for warm starting the next solve."""
    primal_residual: torch.Tensor
    """(N,) worst constraint violation."""
    dual_residual: torch.Tensor
    """(N,) worst violation of stationarity."""
    iterations: int
    """Iterations actually run."""


def _apply(cone: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """(N, B, m) from (N, B, m, k) blocks and an (N, B * k) vector."""
    blocked = vector.view(cone.shape[0], cone.shape[1], cone.shape[3])
    return torch.einsum("nbmk,nbk->nbm", cone, blocked)


def _apply_transpose(cone: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """(N, B * k) from (N, B, m, k) blocks and an (N, B, m) vector."""
    product = torch.einsum("nbmk,nbm->nbk", cone, vector)
    return product.reshape(cone.shape[0], -1)


def _equilibrate(
    hessian: torch.Tensor, gradient: torch.Tensor, cone: torch.Tensor, passes: int
) -> tuple[torch.Tensor, ...]:
    """Modified Ruiz equilibration of the KKT data, OSQP section 5.1.

    Repeatedly divides each variable and each constraint by the square root of the
    largest entry it takes part in, which drives the rows and columns of
    [[P, A'], [A, 0]] towards a common magnitude, then rescales the whole cost so its
    gradient is of order one.

    Returns the scaled P, q and A, the variable scaling, the constraint scaling and the
    cost scaling. The variable scaling maps scaled variables back to real ones.
    """
    num_robots, blocks, rows, width = cone.shape
    size = blocks * width
    scaled_hessian = hessian.clone()
    scaled_gradient = gradient.clone()
    scaled_cone = cone.clone()
    variable = torch.ones(num_robots, blocks, width, device=cone.device, dtype=cone.dtype)
    constraint = torch.ones(num_robots, blocks, rows, device=cone.device, dtype=cone.dtype)
    cost = torch.ones(num_robots, 1, device=cone.device, dtype=cone.dtype)

    for _ in range(passes):
        column = torch.maximum(
            scaled_hessian.abs().amax(dim=1).view(num_robots, blocks, width),
            scaled_cone.abs().amax(dim=2),
        )
        row = scaled_cone.abs().amax(dim=3)
        column_scale = column.clamp_min(_TINY).rsqrt()
        row_scale = row.clamp_min(_TINY).rsqrt()

        flat = column_scale.view(num_robots, size)
        scaled_hessian = scaled_hessian * flat.unsqueeze(-1) * flat.unsqueeze(-2)
        scaled_gradient = scaled_gradient * flat
        scaled_cone = scaled_cone * row_scale.unsqueeze(-1) * column_scale.unsqueeze(-2)

        # keep the cost from drifting off in absolute terms as the variables shrink
        mean_column = scaled_hessian.abs().amax(dim=1).mean(dim=1, keepdim=True)
        largest = torch.maximum(mean_column, scaled_gradient.abs().amax(dim=1, keepdim=True))
        gamma = 1.0 / largest.clamp_min(_TINY)
        scaled_hessian = scaled_hessian * gamma.unsqueeze(-1)
        scaled_gradient = scaled_gradient * gamma
        cost = cost * gamma
        variable = variable * column_scale
        constraint = constraint * row_scale

    return scaled_hessian, scaled_gradient, scaled_cone, variable, constraint, cost


def _invert(
    hessian: torch.Tensor, cone: torch.Tensor, step: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Inverse of P + sigma I + A' diag(rho) A, whose second term is block diagonal and
    so only touches the diagonal blocks.

    The inverse is formed explicitly rather than kept as a factorisation, because the
    iteration applies it to a single right-hand side at a time: a batched matrix-vector
    product runs about four times faster than a batched pair of triangular solves with
    one column, and the one-off cost is repaid within ten iterations. On the
    equilibrated system the matrix is well conditioned and close to unit-diagonal, and
    ADMM recomputes its residual every iteration, so the accuracy an explicit inverse
    gives up here does not accumulate.
    """
    num_robots, blocks, _, width = cone.shape
    kkt = hessian.clone()
    kkt.diagonal(dim1=-2, dim2=-1).add_(sigma)
    weighted = torch.einsum("nbmi,nbm,nbmj->nbij", cone, step, cone)
    kkt.view(num_robots, blocks, width, blocks, width).diagonal(dim1=1, dim2=3).add_(
        weighted.permute(0, 2, 3, 1)
    )
    # positive definite by construction, so the info flag cannot fire for a well-formed
    # problem; cholesky_ex is used only because it will not synchronise to raise
    factor, _ = torch.linalg.cholesky_ex(kkt)
    return torch.cholesky_inverse(factor)


def solve(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    cone: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    primal: torch.Tensor,
    dual: torch.Tensor,
    settings: AdmmSettings,
) -> QpSolution:
    """Solve one QP per robot.

    Args:
        hessian: (N, n, n) P, symmetric positive semi-definite.
        gradient: (N, n) q.
        cone: (N, B, m, k) the constraint block for each of the B groups of k variables.
        lower: (N, B, m) l.
        upper: (N, B, m) u.
        primal: (N, n) warm start for x, zeros on the first solve.
        dual: (N, B, m) warm start for y, zeros on the first solve.
        settings: iteration budget, step sizes and scaling.

    Returns:
        The solution, its duals and its residuals, all in the units passed in. `primal`
        and `dual` are returned rather than written in place; hand them back next time
        to warm start.
    """
    num_robots, blocks, rows, width = cone.shape
    size = blocks * width

    scaled_hessian, scaled_gradient, scaled_cone, variable, constraint, cost = _equilibrate(
        hessian, gradient, cone, settings.scaling_iterations
    )
    flat_variable = variable.view(num_robots, size)
    scaled_lower = lower * constraint
    scaled_upper = upper * constraint

    # the equality rows are the swing legs, whose forces are pinned to zero; they
    # converge much faster with a larger step, and equilibration does not change which
    # rows they are
    step = torch.full_like(lower, settings.rho)
    step = torch.where(
        upper - lower <= settings.equality_tolerance, step * settings.equality_rho_scale, step
    )
    inverse = _invert(scaled_hessian, scaled_cone, step, settings.sigma)

    x = primal / flat_variable
    y = dual * cost.unsqueeze(-1) / constraint
    z = _apply(scaled_cone, x)
    relaxation = settings.relaxation
    iterations = settings.iterations

    for iteration in range(1, settings.iterations + 1):
        rhs = settings.sigma * x - scaled_gradient + _apply_transpose(scaled_cone, step * z - y)
        candidate = torch.bmm(inverse, rhs.unsqueeze(-1)).squeeze(-1)
        projected = _apply(scaled_cone, candidate)

        x = relaxation * candidate + (1.0 - relaxation) * x
        relaxed = relaxation * projected + (1.0 - relaxation) * z
        combined = relaxed + y / step
        z = torch.clamp(combined, scaled_lower, scaled_upper)
        y = step * (combined - z)

        rebalance = (
            settings.rho_update_interval is not None
            and iteration % settings.rho_update_interval == 0
            and iteration < settings.iterations
        )
        if rebalance:
            step = _rebalance(scaled_hessian, scaled_gradient, scaled_cone, step, x, y, z)
            inverse = _invert(scaled_hessian, scaled_cone, step, settings.sigma)

        if settings.check_every is not None and iteration % settings.check_every == 0:
            residuals = _residuals(
                scaled_hessian,
                scaled_gradient,
                scaled_cone,
                flat_variable,
                constraint,
                cost,
                x,
                y,
                z,
            )
            worst = torch.maximum(residuals[0].amax(), residuals[1].amax())
            if bool(worst <= settings.convergence_tolerance):
                iterations = iteration
                break

    primal_residual, dual_residual = _residuals(
        scaled_hessian, scaled_gradient, scaled_cone, flat_variable, constraint, cost, x, y, z
    )
    return QpSolution(
        primal=x * flat_variable,
        dual=y * constraint / cost.unsqueeze(-1),
        primal_residual=primal_residual,
        dual_residual=dual_residual,
        iterations=iterations,
    )


def _rebalance(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    cone: torch.Tensor,
    step: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """Move the step size towards whichever residual is lagging, OSQP section 5.2.

    A step size that is too small leaves the constraints violated, too large leaves
    stationarity violated; the ratio of the two relative residuals says which way to go,
    and its square root is the correction. Applied to every robot's whole constraint
    set, which keeps the equality rows' head start intact.
    """
    primal = (_apply(cone, x) - z).abs().amax(dim=(1, 2))
    stationarity = torch.einsum("nij,nj->ni", hessian, x) + gradient + _apply_transpose(cone, y)
    dual = stationarity.abs().amax(dim=1)

    primal_scale = torch.maximum(
        _apply(cone, x).abs().amax(dim=(1, 2)), z.abs().amax(dim=(1, 2))
    )
    dual_scale = torch.maximum(
        torch.einsum("nij,nj->ni", hessian, x).abs().amax(dim=1),
        torch.maximum(gradient.abs().amax(dim=1), _apply_transpose(cone, y).abs().amax(dim=1)),
    )
    relative_primal = primal / primal_scale.clamp_min(_TINY)
    relative_dual = (dual / dual_scale.clamp_min(_TINY)).clamp_min(_TINY)
    ratio = (relative_primal / relative_dual).clamp(1e-6, 1e6)
    return step * ratio.sqrt().view(-1, 1, 1)


def _residuals(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    cone: torch.Tensor,
    variable: torch.Tensor,
    constraint: torch.Tensor,
    cost: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Primal and dual infinity-norm residuals, (N,) each, in unscaled units.

    Both are computed on the scaled problem and then undone: the primal residual is a
    constraint value, so it picks up the constraint scaling, and the dual one is a
    gradient with respect to x, so it picks up the variable and cost scalings.
    """
    primal = ((_apply(cone, x) - z) / constraint).abs().amax(dim=(1, 2))
    stationarity = torch.einsum("nij,nj->ni", hessian, x) + gradient + _apply_transpose(cone, y)
    dual = (stationarity / (variable * cost)).abs().amax(dim=1)
    return primal, dual
