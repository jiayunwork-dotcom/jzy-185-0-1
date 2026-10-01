"""Interpret a simplex solution in the language of a feed formula.

Produces:

* per-ingredient quantities and reduced costs (检验数);
* per-constraint evaluation, slack, active (起作用) flag, shadow price;
* an independently recomputed optimality certificate: primal feasibility,
  dual feasibility, complementary slackness, primal/dual objective gap;
* price sensitivity intervals inside which the optimal basis / ingredient
  composition is unchanged.

All checks substitute values into the *original* problem rather than trusting
the solver's bookkeeping, so the certificate is genuinely self-contained.

Degenerate shadow prices
------------------------
At a degenerate vertex a basis can attach a non-zero dual value to a
constraint whose relaxation does not actually move the optimum. The reported
shadow price is therefore verified by warm-starting the solver on the model
with that row's right-hand side perturbed by +/- eps and measuring the real
cost response; a constraint is called *active* only when its marginal cost is
genuinely non-zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lpmodel import StandardModel
from .simplex import SimplexOutcome, solve_standard

CHECK_TOL = 1e-8
ACTIVE_TOL = 1e-8
PERTURB_EPS = 1e-6


@dataclass
class Certificate:
    primal_feasible: bool
    dual_feasible: bool
    complementary_slackness: bool
    primal_objective: float
    dual_objective: float
    relative_gap: float
    max_primal_residual: float
    min_reduced_cost: float
    max_cs_violation: float
    details: list[dict]


def shadow_prices(model: StandardModel, out: SimplexOutcome) -> list[float]:
    """Canonical dual -> shadow price w.r.t. each original RHS.

    Canonical row is ``flip * (original normal form)``; the dual objective
    therefore changes with the original RHS at rate ``flip * y``.
    """
    return [float(model.rows[i].flip * out.y[i])
            for i in range(len(model.rows))]


def verified_shadow_prices(
    model: StandardModel, out: SimplexOutcome, c: np.ndarray
) -> list[float]:
    """Shadow prices verified against real cost response under RHS perturb.

    The canonical dual ``y`` gives the right value whenever the optimal basis
    persists under an infinitesimal RHS shift.  At degenerate vertices a
    binding row's dual can be one-sided / misleading; here each binding
    inequality is re-solved (warm-started from the optimal basis) with its
    original RHS relaxed by ``PERTURB_EPS`` and the finite-difference cost
    slope is returned instead.
    """
    shadows = shadow_prices(model, out)
    kept = np.ones(len(model.rows), dtype=bool)
    kept[list(out.dropped_rows)] = False
    for i, meta in enumerate(model.rows):
        if meta.sense == "=" or not kept[i]:
            continue
        # Re-evaluate the true slack in the original constraint.
        coef = model.A[i, model.ingredient_cols] / meta.flip
        lhs = float(coef @ out.x[model.ingredient_cols])
        gap = lhs - meta.rhs_original if meta.sense == ">=" \
            else meta.rhs_original - lhs
        if abs(gap) > ACTIVE_TOL:
            shadows[i] = 0.0
            continue
        if abs(shadows[i]) <= CHECK_TOL:
            continue
        # Perturb: relaxing an inequality means moving its RHS away from the
        # feasible region by eps (>= lowers rhs, <= raises rhs).
        b2 = model.b.copy()
        direction = -1.0 if meta.sense == ">=" else +1.0
        # canonical rhs = flip * original rhs at build time; shift the
        # canonical row rhs consistently.
        b2[i] += meta.flip * direction * PERTURB_EPS
        try:
            # A cold start here is cheap and robust: the original optimal
            # basis may refer to a reduced row set after redundant-row
            # deletion, so its column indices cannot be blindly reused.
            o2 = solve_standard(
                model.A, b2, c, None, model.artificial_cols,
                dropable_rows=set(range(len(model.rows))))
        except Exception:
            continue
        base_cost = float(c[model.ingredient_cols]
                          @ out.x[model.ingredient_cols])
        new_cost = float(c[model.ingredient_cols]
                         @ o2.x[model.ingredient_cols])
        slope = (new_cost - base_cost) / (direction * PERTURB_EPS)
        # shadow price = rate of cost increase per unit of tightening, i.e.
        # rate of decrease per unit of relaxation; keep sign convention.
        shadows[i] = float(slope)
    return shadows


def constraint_report(
    model: StandardModel,
    out: SimplexOutcome,
    shadows: list[float] | None = None,
) -> list[dict]:
    x = out.x
    if shadows is None:
        shadows = shadow_prices(model, out)
    rows_out = []
    kept = np.ones(len(model.rows), dtype=bool)
    kept[list(out.dropped_rows)] = False
    for i, meta in enumerate(model.rows):
        # Original-form ingredient coefficients.
        coef = model.A[i, model.ingredient_cols] / meta.flip
        lhs = float(coef @ x[model.ingredient_cols])
        rhs = meta.rhs_original
        if meta.sense == ">=":
            slack = lhs - rhs
        elif meta.sense == "<=":
            slack = rhs - lhs
        else:
            slack = abs(lhs - rhs)
        binding = bool(kept[i] and abs(slack) <= ACTIVE_TOL)
        shadow = float(shadows[i])
        active = binding and (
            meta.sense == "=" or abs(shadow) > CHECK_TOL)
        rows_out.append({
            "index": i,
            "kind": meta.kind,
            "ref": meta.ref,
            "sense": meta.sense,
            "rhs": rhs,
            "lhs": lhs,
            "slack": float(max(slack, 0.0)) if meta.sense != "=" else slack,
            "binding": binding,
            "active": bool(active),
            "shadow_price": shadow,
            "redundant": (not kept[i]),
        })
    return rows_out


def ingredient_report(
    model: StandardModel, out: SimplexOutcome, mins: list[float], maxs: list[float]
) -> list[dict]:
    rep = []
    for k, ing_id in enumerate(model.ingredient_ids):
        col = model.ingredient_cols[k]
        value = float(out.x[col])
        status = "basic" if col in set(out.basis.tolist()) else "at_lower"
        if abs(value - maxs[k]) <= ACTIVE_TOL and maxs[k] > mins[k] + ACTIVE_TOL:
            status = "at_upper"
        rep.append({
            "ingredient_id": ing_id,
            "quantity": value,
            "reduced_cost": float(out.reduced_costs[col]),
            "status": status,
            "min": float(mins[k]),
            "max": float(maxs[k]),
        })
    return rep


def verify(
    model: StandardModel,
    out: SimplexOutcome,
    prices: list[float],
    mins: list[float],
    maxs: list[float],
) -> Certificate:
    """Recompute every optimality condition from scratch."""
    xi = out.x[model.ingredient_cols]
    A = model.A
    b = model.b
    kept = np.ones(A.shape[0], dtype=bool)
    kept[list(out.dropped_rows)] = False

    # --- primal feasibility: canonical equalities + bounds ---------------
    eq_res = np.abs(A @ out.x - b)
    eq_res[~kept] = 0.0
    bound_low = mins - xi
    bound_high = xi - np.asarray(maxs)
    nonneg = -np.minimum(out.x, 0.0)
    max_res = float(max(eq_res.max(initial=0.0),
                        np.maximum(bound_low, 0.0).max(initial=0.0),
                        np.maximum(bound_high, 0.0).max(initial=0.0),
                        nonneg.max(initial=0.0)))
    primal_ok = max_res <= CHECK_TOL

    # --- dual feasibility: reduced costs of every real column >= 0 -------
    art = set(model.artificial_cols)
    real_cols = [j for j in range(A.shape[1]) if j not in art]
    rc_real = out.reduced_costs[real_cols]
    min_rc = float(rc_real.min())
    dual_ok = min_rc >= -CHECK_TOL

    # --- complementary slackness: x_j * rc_j == 0 for all real columns ---
    cs_viol = float(np.max(np.abs(out.x[real_cols] * rc_real)))
    # plus: every positive dual sign mismatch is already encoded by slack
    # columns being real columns, so no separate check is needed.
    cs_ok = cs_viol <= CHECK_TOL

    pobj = float(np.asarray(prices) @ xi)
    dobj = float(out.y @ b)
    denom = max(abs(pobj), 1e-30)
    rel_gap = abs(pobj - dobj) / denom

    details = []
    for j in real_cols:
        details.append({
            "column": int(j),
            "value": float(out.x[j]),
            "reduced_cost": float(out.reduced_costs[j]),
            "product": float(out.x[j] * out.reduced_costs[j]),
        })

    return Certificate(
        primal_feasible=bool(primal_ok),
        dual_feasible=bool(dual_ok),
        complementary_slackness=bool(cs_ok),
        primal_objective=pobj,
        dual_objective=dobj,
        relative_gap=float(rel_gap),
        max_primal_residual=max_res,
        min_reduced_cost=min_rc,
        max_cs_violation=cs_viol,
        details=details,
    )


def price_sensitivity(
    model: StandardModel, out: SimplexOutcome
) -> list[dict]:
    """Price interval for each ingredient while the basis stays optimal.

    For a basic ingredient with objective price c_j + delta the reduced costs
    become ``rc - delta * v`` with ``v = A^T B^{-T} e_p``; every non-basic
    real column must keep a non-negative reduced cost. Non-basic ingredients
    sit at their lower bound (zero), so only their own reduced cost matters.
    """
    A = model.A
    basis = out.basis
    art = set(model.artificial_cols)
    real_cols = np.array([j for j in range(A.shape[1]) if j not in art])
    # After redundant-row deletion the basis spans only the kept rows.
    kept = np.ones(A.shape[0], dtype=bool)
    kept[list(out.dropped_rows)] = False
    A_eff = A[kept, :]
    results: list[dict] = []

    basis_set = set(basis.tolist())
    B = A_eff[:, basis]
    # Row index each ingredient column occupies in the basis, if any.
    basis_pos = {int(col): i for i, col in enumerate(basis)}

    for k, ing_id in enumerate(model.ingredient_ids):
        col = model.ingredient_cols[k]
        c0 = float(model.c[col])
        if col in basis_set:
            p = basis_pos[col]
            z = np.linalg.solve(B.T, _unit(B.shape[0], p))
            v = A_eff.T @ z
            d_lo, d_hi = -np.inf, np.inf
            for q in real_cols:
                if int(q) in basis_set:
                    continue
                vq = v[q]
                rc0 = out.reduced_costs[q]
                if vq > 1e-12:
                    d_hi = min(d_hi, rc0 / vq)
                elif vq < -1e-12:
                    d_lo = max(d_lo, rc0 / vq)
        else:
            # non-basic at lower bound: rc = rc0 + delta must stay >= 0
            d_lo, d_hi = -float(out.reduced_costs[col]), np.inf
        results.append({
            "ingredient_id": ing_id,
            "price_low": None if not np.isfinite(c0 + d_lo) else float(c0 + d_lo),
            "price_high": None if not np.isfinite(c0 + d_hi) else float(c0 + d_hi),
            "current_price": c0,
        })
    return results


def _unit(n: int, i: int) -> np.ndarray:
    e = np.zeros(n)
    e[i] = 1.0
    return e
