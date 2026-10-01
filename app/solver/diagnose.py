"""Explain *why* a feed formula is infeasible.

A bare "no solution" answer is not acceptable.  Diagnosis has two layers:

1. Fast structural checks that additionally produce a human explanation and
   the witness ingredients:
     - the sum of maximum inclusions is below 1 kg (or mins exceed 1);
     - a nutrient lower bound exceeds the best attainable value under the
       inclusion limits (and symmetrically for upper bounds);
     - a ratio constraint is infeasible by extrema of its linearised form;
2. Fallback: an irreducible infeasible subset found by a deletion filter.
   Macro-constraints are removed one at a time while the rest stays
   infeasible; the remaining set is a minimal conflicting explanation.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .lpmodel import build_model
from .simplex import InfeasibleLP, solve_standard


def _fmt(v: float) -> str:
    """60.0 -> '60', 1.5 -> '1.5'."""
    return f"{float(v):g}"


def _extrema(coef: np.ndarray, mins: np.ndarray, maxs: np.ndarray,
             total: float | None = None) -> tuple[float, float]:
    """Min/max of sum coef_i x_i over the box [mins, maxs].

    With ``total`` given, additionally require sum x_i = total; this is the
    classic clip-and-distribute greedy computation.
    """
    if total is None:
        lo = float(np.sum(np.where(coef >= 0, coef * mins, coef * maxs)))
        hi = float(np.sum(np.where(coef >= 0, coef * maxs, coef * mins)))
        return lo, hi

    x = mins.copy()
    remaining = total - float(x.sum())
    if remaining < -1e-9:
        return np.inf, -np.inf  # mins already exceed total
    order = np.argsort(-coef)
    cap = maxs - mins
    for i in order:
        add = min(cap[i], remaining)
        x[i] += add
        remaining -= add
    if remaining > 1e-8:
        return np.inf, -np.inf  # cannot reach total
    hi = float(coef @ x)

    x = mins.copy()
    remaining = total - float(x.sum())
    order = np.argsort(coef)
    for i in order:
        add = min(cap[i], remaining)
        x[i] += add
        remaining -= add
    lo = float(coef @ x)
    return lo, hi


def structural_checks(
    ingredient_ids: list[str],
    nutrients: dict[str, list[float]],
    mins: list[float],
    maxs: list[float],
    nutrient_bounds: dict[str, tuple[float | None, float | None]],
    ratios: list[dict],
) -> list[dict[str, Any]]:
    """Return a list of conflict descriptions (empty => nothing obvious)."""
    mins_a = np.asarray(mins, dtype=float)
    maxs_a = np.asarray(maxs, dtype=float)
    conflicts: list[dict[str, Any]] = []

    # ---- total mass -------------------------------------------------------
    if float(maxs_a.sum()) < 1.0 - 1e-9:
        top = sorted(
            ({"ingredient_id": ingredient_ids[i], "max": float(maxs_a[i])}
             for i in range(len(ingredient_ids))),
            key=lambda d: -d["max"],
        )
        conflicts.append({
            "code": "max_inclusion_below_total",
            "message": ("各原料最大添加比例之和为 "
                        f"{float(maxs_a.sum()):.6f}，小于总量 1"),
            "constraints": ["mass=1"],
            "witness": top,
        })
    if float(mins_a.sum()) > 1.0 + 1e-9:
        conflicts.append({
            "code": "min_inclusion_above_total",
            "message": ("各原料最小添加比例之和为 "
                        f"{float(mins_a.sum()):.6f}，大于总量 1"),
            "constraints": ["mass=1"],
            "witness": [{"ingredient_id": ingredient_ids[i],
                         "min": float(mins_a[i])}
                        for i in np.argsort(-mins_a)],
        })

    # ---- nutrient bounds --------------------------------------------------
    for code, (lb, ub) in nutrient_bounds.items():
        coef = np.asarray(nutrients[code], dtype=float)
        if lb is not None:
            _, attainable = _extrema(coef, mins_a, maxs_a, total=1.0)
            if attainable < lb - 1e-8:
                # ingredients that contribute most at their upper bound
                contrib = sorted(
                    ((ingredient_ids[i], coef[i], float(maxs_a[i]))
                     for i in range(len(ingredient_ids))),
                    key=lambda t: -t[1] * t[2],
                )
                conflicts.append({
                    "code": "nutrient_lower_unattainable",
                    "message": (f"营养 {code} 下限 {lb:g} 高于各原料在添加上限内"
                                f"可达到的最大值 {attainable:.6g}"),
                    "constraints": [f"{code}>={_fmt(lb)}"],
                    "witness": [
                        {"ingredient_id": iid, "content": float(v),
                         "max_inclusion": hi,
                         "best_contribution": float(v) * hi}
                        for iid, v, hi in contrib[:5]
                    ],
                    "attainable_max": attainable,
                    "required": float(lb),
                })
        if ub is not None:
            attainable, _ = _extrema(coef, mins_a, maxs_a, total=1.0)
            if attainable > ub + 1e-8:
                conflicts.append({
                    "code": "nutrient_upper_unavoidable",
                    "message": (f"营养 {code} 上限 {ub:g} 低于各原料在添加下限下"
                                f"不可避免的最小值 {attainable:.6g}"),
                    "constraints": [f"{code}<={ub}"],
                    "witness": [],
                    "attainable_min": attainable,
                    "required": float(ub),
                })

    # ---- ratios -----------------------------------------------------------
    for r in ratios:
        num = np.asarray(nutrients[r["numerator"]], dtype=float)
        den = np.asarray(nutrients[r["denominator"]], dtype=float)
        if r.get("min") is not None:
            coef = num - float(r["min"]) * den
            lo, _ = _extrema(coef, mins_a, maxs_a, total=1.0)
            if lo < -1e-8:
                conflicts.append({
                    "code": "ratio_lower_infeasible",
                    "message": (f"比值 {r['numerator']}/{r['denominator']} "
                                f">= {r['min']:g} 无法满足"),
                    "constraints": [f"ratio:{r['id']}"],
                    "witness": [],
                })
        if r.get("max") is not None:
            coef = float(r["max"]) * den - num
            lo, _ = _extrema(coef, mins_a, maxs_a, total=1.0)
            if lo < -1e-8:
                conflicts.append({
                    "code": "ratio_upper_infeasible",
                    "message": (f"比值 {r['numerator']}/{r['denominator']} "
                                f"<= {r['max']:g} 无法满足"),
                    "constraints": [f"ratio:{r['id']}"],
                    "witness": [],
                })

    return conflicts


# ---- deletion filter -------------------------------------------------------

def _macro_list(
    nutrient_bounds: dict[str, tuple[float | None, float | None]],
    ratios: list[dict],
    ingredient_ids: list[str],
) -> list[dict[str, Any]]:
    macros: list[dict[str, Any]] = [{"kind": "mass"}]
    for code, (lb, ub) in nutrient_bounds.items():
        if lb is not None and lb > 0:
            macros.append({"kind": "nutrient", "code": code,
                           "sense": ">=", "rhs": float(lb)})
        if ub is not None:
            macros.append({"kind": "nutrient", "code": code,
                           "sense": "<=", "rhs": float(ub)})
    for r in ratios:
        if r.get("min") is not None:
            macros.append({"kind": "ratio", "ratio": r, "side": "low"})
        if r.get("max") is not None:
            macros.append({"kind": "ratio", "ratio": r, "side": "high"})
    for iid in ingredient_ids:
        macros.append({"kind": "min", "ingredient": iid})
        macros.append({"kind": "max", "ingredient": iid})
    return macros


def _feasible_with(
    macros: list[dict[str, Any]],
    ingredient_ids: list[str],
    prices: list[float],
    nutrients: dict[str, list[float]],
    mins: list[float],
    maxs: list[float],
    ratios: list[dict],
) -> bool:
    kept_ratios: dict[str, dict] = {}
    ratio_sides: dict[str, set[str]] = {}
    bounds: dict[str, list[float | None]] = {}
    keep_mins = [0.0] * len(ingredient_ids)
    keep_maxs = [1.0] * len(ingredient_ids)
    mass_kept = False
    idx = {iid: i for i, iid in enumerate(ingredient_ids)}
    for m in macros:
        if m["kind"] == "mass":
            mass_kept = True
        elif m["kind"] == "nutrient":
            vals = bounds.setdefault(m["code"], [None, None])
            if m["sense"] == ">=":
                vals[0] = m["rhs"]
            else:
                vals[1] = m["rhs"]
        elif m["kind"] == "ratio":
            r = m["ratio"]
            kept_ratios.setdefault(r["id"], {
                "id": r["id"], "numerator": r["numerator"],
                "denominator": r["denominator"]})
            ratio_sides.setdefault(r["id"], set()).add(m["side"])
        elif m["kind"] == "min":
            keep_mins[idx[m["ingredient"]]] = mins[idx[m["ingredient"]]]
        elif m["kind"] == "max":
            keep_maxs[idx[m["ingredient"]]] = maxs[idx[m["ingredient"]]]

    ratio_out = []
    for rid, r in kept_ratios.items():
        orig = next(q for q in ratios if q["id"] == rid)
        if "low" in ratio_sides[rid]:
            r["min"] = orig["min"]
        if "high" in ratio_sides[rid]:
            r["max"] = orig["max"]
        ratio_out.append(r)

    nutrient_bounds = {code: (vals[0], vals[1]) for code, vals in bounds.items()}

    if not mass_kept:
        # Feasibility over the box only (no fixed total).
        for code, (lb, ub) in nutrient_bounds.items():
            coef = np.asarray(nutrients[code], dtype=float)
            lo, hi = _extrema(coef, np.asarray(keep_mins),
                              np.asarray(keep_maxs))
            if lb is not None and hi < lb - 1e-8:
                return False
            if ub is not None and lo > ub + 1e-8:
                return False
        for r in ratio_out:
            num = np.asarray(nutrients[r["numerator"]], dtype=float)
            den = np.asarray(nutrients[r["denominator"]], dtype=float)
            if r.get("min") is not None:
                lo, _ = _extrema(num - float(r["min"]) * den,
                                 np.asarray(keep_mins), np.asarray(keep_maxs))
                if lo < -1e-8:
                    return False
            if r.get("max") is not None:
                lo, _ = _extrema(float(r["max"]) * den - num,
                                 np.asarray(keep_mins), np.asarray(keep_maxs))
                if lo < -1e-8:
                    return False
        return True

    model = build_model(
        ingredient_ids, prices, nutrients, [0.0] * len(ingredient_ids),
        keep_mins, keep_maxs, nutrient_bounds, ratio_out,
    )
    try:
        solve_standard(model.A, model.b, model.c, None,
                       model.artificial_cols,
                       dropable_rows=set(range(len(model.rows))))
    except InfeasibleLP:
        return False
    return True


def diagnose(
    ingredient_ids: list[str],
    prices: list[float],
    nutrients: dict[str, list[float]],
    mins: list[float],
    maxs: list[float],
    nutrient_bounds: dict[str, tuple[float | None, float | None]],
    ratios: list[dict],
) -> list[dict[str, Any]]:
    """Return at least one conflict explanation for an infeasible formula."""
    conflicts = structural_checks(
        ingredient_ids, nutrients, mins, maxs, nutrient_bounds, ratios)
    if conflicts:
        return conflicts

    macros = _macro_list(nutrient_bounds, ratios, ingredient_ids)
    # deletion filter
    remaining = list(macros)
    for m in macros:
        trial = [q for q in remaining if q is not m]
        try:
            if not _feasible_with(trial, ingredient_ids, prices, nutrients,
                                  mins, maxs, ratios):
                remaining = trial  # still infeasible without it: drop
        except Exception:
            pass
    labels = []
    for m in remaining:
        if m["kind"] == "mass":
            labels.append("mass=1")
        elif m["kind"] == "nutrient":
            labels.append(f"{m['code']}{m['sense']}{_fmt(m['rhs'])}")
        elif m["kind"] == "ratio":
            labels.append(f"ratio:{m['ratio']['id']}:{m['side']}")
        else:
            labels.append(f"{m['kind']}({m['ingredient']})")
    return [{
        "code": "irreducible_infeasible_subset",
        "message": "以下约束集合本身已互相矛盾（最小冲突集）：" + "、".join(labels),
        "constraints": labels,
        "witness": [],
    }]
