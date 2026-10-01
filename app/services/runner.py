"""Resolve versioned inputs, run the solver, persist an optimisation row."""
from __future__ import annotations

from typing import Any

from ..errors import InfeasibleError, NotFoundError
from ..storage import Database
from ..validation import validate_formula_spec
from .optimize import solve_formula


class OptimizationRunner:
    def __init__(self, db: Database):
        self.db = db

    # -- resolution ----------------------------------------------------------
    def _resolve(self, formula_id: str, formula_version: int | None,
                 library_version: int | None) -> tuple[dict, list[dict]]:
        spec_row = self.db.get_formula_spec(formula_id, formula_version)
        if spec_row is None:
            raise NotFoundError("formula_version",
                                f"{formula_id}@{formula_version}")
        if library_version is None:
            library_version = self.db.latest_library_version()
            if library_version is None:
                raise NotFoundError("library_version", "latest")
        if not self.db.library_version_exists(library_version):
            raise NotFoundError("library_version", library_version)
        snapshot = self.db.list_snapshot_ingredients(library_version)
        return spec_row, snapshot

    # -- core ----------------------------------------------------------------
    def run_optimization(
        self,
        *,
        formula_id: str,
        formula_version: int | None = None,
        library_version: int | None = None,
        trigger: str = "manual",
        job_id: str | None = None,
        warm_from_id: str | None = None,
        force_cold_start: bool = False,
    ) -> dict[str, Any]:
        spec_row, snapshot = self._resolve(
            formula_id, formula_version, library_version)
        fv = spec_row["version"]
        lv = library_version if library_version is not None \
            else self.db.latest_library_version()
        spec = spec_row["spec"]

        snap_by_id = {r["ingredient_id"]: r for r in snapshot}
        known_codes = set(self.db.list_nutrient_codes())
        # also include every code present in this snapshot
        for row in snapshot:
            known_codes.update((row.get("nutrients") or {}).keys())

        ids = [item["ingredient_id"] for item in spec["ingredients"]]
        # Ratio denominator-zero validation only sees ingredients the formula
        # actually references (and that exist in this snapshot).
        formula_snapshot = {
            iid: dict((snap_by_id.get(iid) or {}).get("nutrients") or {})
            for iid in ids
        }
        validate_formula_spec(
            spec,
            known_ingredient_ids=set(snap_by_id),
            known_nutrient_codes=known_codes,
            snapshot_nutrients=formula_snapshot,
        )

        bounds = {item["ingredient_id"]: {
            "min": float(item.get("min", 0.0)),
            "max": float(item.get("max", 1.0)),
        } for item in spec["ingredients"]}
        nutrient_codes = sorted(set(spec.get("nutrient_bounds", {}).keys()) | {
            c for r in spec.get("ratios", [])
            for c in (r["numerator"], r["denominator"])})

        warm_basis = None
        warm_id = None
        if not force_cold_start:
            prev = self.db.get_optimization(warm_from_id) if warm_from_id \
                else self.db.latest_successful_optimization(formula_id)
            if prev and prev["status"] == "ok" and prev["result"]:
                prev_res = prev["result"]
                stored = prev_res.get("structure_key", [])
                # JSON turns tuples into lists; convert back recursively.
                def _as_tuple(v):
                    return tuple(_as_tuple(x) if isinstance(x, list) else x
                                 for x in v)
                key = tuple(_as_tuple(x) if isinstance(x, list) else x
                            for x in stored)
                warm_basis = (key, prev_res["basis"])
                warm_id = prev["optimization_id"]

        try:
            result = solve_formula(
                ingredient_ids=ids,
                ingredients_payload=[snap_by_id[i] for i in ids],
                nutrient_codes=nutrient_codes,
                nutrient_bounds=spec.get("nutrient_bounds", {}),
                ingredient_bounds=bounds,
                ratios=spec.get("ratios", []),
                warm_basis=warm_basis,
            )
        except InfeasibleError as e:
            oid = self.db.insert_optimization(
                formula_id=formula_id, formula_version=fv,
                library_version=lv, trigger=trigger, job_id=job_id,
                status="infeasible", conflicts=e.conflicts)
            # Surface conflicts to the caller; HTTP layer maps this to 409.
            # The row is already persisted for history/comparison.
            setattr(e, "optimization_id", oid)
            setattr(e, "payload", {"optimization_id": oid,
                                   "status": "infeasible",
                                   "formula_version": fv,
                                   "library_version": lv,
                                   "conflicts": e.conflicts})
            raise

        oid = self.db.insert_optimization(
            formula_id=formula_id, formula_version=fv,
            library_version=lv, trigger=trigger, job_id=job_id,
            status="ok", result=result, warm_from_id=warm_id)
        return {"optimization_id": oid, "status": "ok",
                "formula_version": fv, "library_version": lv,
                "warm_from_id": warm_id,
                "warm_start_accepted": result["warm_start_accepted"],
                **result}

    # -- reads ---------------------------------------------------------------
    def get(self, optimization_id: str) -> dict[str, Any]:
        row = self.db.get_optimization(optimization_id)
        if row is None:
            raise NotFoundError("optimization", optimization_id)
        return row

    def history(self, formula_id: str, limit: int = 100) -> list[dict[str, Any]]:
        if self.db.get_formula(formula_id) is None:
            raise NotFoundError("formula", formula_id)
        return self.db.list_optimizations(formula_id, limit=limit)

    def compare(self, formula_id: str,
                optimization_a: str, optimization_b: str) -> dict[str, Any]:
        a = self.db.get_optimization(optimization_a)
        b = self.db.get_optimization(optimization_b)
        if a is None:
            raise NotFoundError("optimization", optimization_a)
        if b is None:
            raise NotFoundError("optimization", optimization_b)
        if a["formula_id"] != formula_id or b["formula_id"] != formula_id:
            raise NotFoundError("optimization",
                                f"{optimization_a}/{optimization_b}")

        def qty(row: dict[str, Any]) -> dict[str, float] | None:
            if row["status"] != "ok" or not row["result"]:
                return None
            return {ing["ingredient_id"]: ing["quantity"]
                    for ing in row["result"]["ingredients"]}

        qa, qb = qty(a), qty(b)
        diffs = []
        for iid in sorted(set(qa or {}) | set(qb or {})):
            diffs.append({
                "ingredient_id": iid,
                "quantity_a": (qa or {}).get(iid, 0.0),
                "quantity_b": (qb or {}).get(iid, 0.0),
                "delta": (qb or {}).get(iid, 0.0) - (qa or {}).get(iid, 0.0),
            })
        cost_a = a["result"]["cost"] if qa is not None else None
        cost_b = b["result"]["cost"] if qb is not None else None
        return {
            "formula_id": formula_id,
            "a": self._summary(a),
            "b": self._summary(b),
            "cost_delta": (cost_b - cost_a)
                          if cost_a is not None and cost_b is not None else None,
            "ingredient_deltas": diffs,
        }

    @staticmethod
    def _summary(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "optimization_id": row["optimization_id"],
            "status": row["status"],
            "formula_version": row["formula_version"],
            "library_version": row["library_version"],
            "cost": row["result"]["cost"] if row["result"] else None,
            "created_at": row["created_at"],
        }
