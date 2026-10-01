"""High level optimisation service: spec + ingredient library -> result."""
from __future__ import annotations

import math
from typing import Any

from ..errors import InfeasibleError
from ..solver import analyze
from ..solver.diagnose import diagnose
from ..solver.lpmodel import build_model
from ..solver.simplex import InfeasibleLP, solve_standard


def _finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def solve_formula(
    *,
    ingredient_ids: list[str],
    ingredients_payload: list[dict[str, Any]],
    nutrient_codes: list[str],
    nutrient_bounds: dict[str, dict[str, float | None]],
    ingredient_bounds: dict[str, dict[str, float]],
    ratios: list[dict[str, Any]],
    warm_basis: tuple[tuple, list[int]] | None = None,
) -> dict[str, Any]:
    """Solve one feed formula.

    ``ingredients_payload`` is the ingredient-library snapshot rows; the
    nutrient matrix is built from their ``nutrients`` mapping.

    Returns a JSON-serialisable result dict including the optimality
    certificate, shadow prices, reduced costs, sensitivity ranges and the
    optimal basis (for warm starts).
    """
    ids = list(ingredient_ids)
    by_id = {row["ingredient_id"]: row for row in ingredients_payload}
    prices = [float(by_id[i]["price"]) for i in ids]
    dm = [float(by_id[i].get("dry_matter", 0.0) or 0.0) for i in ids]
    nutrients: dict[str, list[float]] = {}
    for code in nutrient_codes:
        nutrients[code] = [
            float((by_id[i].get("nutrients") or {}).get(code, 0.0) or 0.0)
            for i in ids
        ]
    mins = [float(ingredient_bounds[i]["min"]) for i in ids]
    maxs = [float(ingredient_bounds[i]["max"]) for i in ids]

    nb: dict[str, tuple[float | None, float | None]] = {}
    for code, lo_hi in nutrient_bounds.items():
        nb[code] = (lo_hi.get("min"), lo_hi.get("max"))

    model = build_model(ids, prices, nutrients, dm, mins, maxs, nb, ratios)

    warm = None
    if warm_basis is not None:
        key, basis_cols = warm_basis
        if key == model.structure_key() and len(basis_cols) == model.A.shape[0]:
            warm = basis_cols

    try:
        out = solve_standard(
            model.A, model.b, model.c, warm,
            model.artificial_cols,
            dropable_rows=set(range(len(model.rows))),
        )
    except InfeasibleLP:
        conflicts = diagnose(ids, prices, nutrients, mins, maxs, nb, ratios)
        raise InfeasibleError(conflicts)

    cert = analyze.verify(model, out, prices, mins, maxs)
    if not (cert.primal_feasible and cert.dual_feasible
            and cert.complementary_slackness and cert.relative_gap < 1e-9):
        raise RuntimeError(
            "internal solver failed its own optimality certificate: "
            f"residual={cert.max_primal_residual:.2e}, "
            f"min_rc={cert.min_reduced_cost:.2e}, "
            f"cs={cert.max_cs_violation:.2e}, gap={cert.relative_gap:.2e}")

    # Degenerate-vertex-safe shadow prices (re-solve each binding row with a
    # tiny RHS perturbation, warm-started from this optimal basis).
    shadows = analyze.verified_shadow_prices(model, out, model.c)
    ingredients = analyze.ingredient_report(model, out, mins, maxs)
    constraints = analyze.constraint_report(model, out, shadows)

    return {
        "cost": cert.primal_objective,
        "dual_objective": cert.dual_objective,
        "relative_gap": cert.relative_gap,
        "iterations": out.iterations,
        "phase1_iterations": out.phase1_iterations,
        "warm_start_accepted": warm is not None,
        "ingredients": ingredients,
        "constraints": constraints,
        "reduced_costs": [
            {"ingredient_id": r["ingredient_id"],
             "reduced_cost": r["reduced_cost"],
             "status": r["status"]}
            for r in ingredients
        ],
        "certificate": {
            "primal_feasible": cert.primal_feasible,
            "dual_feasible": cert.dual_feasible,
            "complementary_slackness": cert.complementary_slackness,
            "primal_objective": cert.primal_objective,
            "dual_objective": cert.dual_objective,
            "relative_gap": cert.relative_gap,
            "max_primal_residual": cert.max_primal_residual,
            "min_reduced_cost": cert.min_reduced_cost,
            "max_cs_violation": cert.max_cs_violation,
            "columns": cert.details,
        },
        "sensitivity": analyze.price_sensitivity(model, out),
        "structure_key": list(model.structure_key()),
        "basis": out.basis.tolist(),
    }
