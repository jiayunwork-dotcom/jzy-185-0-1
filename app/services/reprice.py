"""Assemble and submit weekly re-pricing batch jobs."""
from __future__ import annotations

from typing import Any

from ..errors import NotFoundError, ValidationError
from ..storage import Database
from .scheduler import JobScheduler


class RepriceService:
    def __init__(self, db: Database, scheduler: JobScheduler):
        self.db = db
        self.scheduler = scheduler

    def affected_ingredients(self, library_version: int) -> list[str]:
        """Ingredients changed between the previous version and this one."""
        if library_version <= 1:
            prev = self.db.list_snapshot_ingredients(library_version)
            return sorted(r["ingredient_id"] for r in prev)
        if not self.db.library_version_exists(library_version):
            raise NotFoundError("library_version", library_version)
        diff = self.db.diff_library_versions(library_version - 1,
                                             library_version)
        return sorted(set(diff["added"]) | {c["ingredient_id"]
                                            for c in diff["changed"]})

    def preview_targets(self, library_version: int,
                        ingredient_ids: list[str] | None = None
                        ) -> list[dict[str, Any]]:
        affected = set(ingredient_ids or
                       self.affected_ingredients(library_version))
        targets = []
        for f in self.db.list_formulas():
            spec_row = self.db.get_formula_spec(f["formula_id"])
            if spec_row is None:
                continue
            used = {item["ingredient_id"]
                    for item in spec_row["spec"].get("ingredients", [])}
            if used & affected:
                targets.append({
                    "formula_id": f["formula_id"],
                    "name": f["name"],
                    "formula_version": spec_row["version"],
                })
        return targets

    def submit_reprice(
        self,
        *,
        library_version: int | None = None,
        ingredient_ids: list[str] | None = None,
        formula_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        if library_version is None:
            library_version = self.db.latest_library_version()
        if library_version is None or not self.db.library_version_exists(
                library_version):
            raise NotFoundError("library_version", library_version)

        targets = self.preview_targets(library_version, ingredient_ids)
        if formula_ids is not None:
            wanted = set(formula_ids)
            targets = [t for t in targets if t["formula_id"] in wanted]
        if not targets:
            raise ValidationError(
                "no affected formulas",
                {"formula_ids": "没有引用受影响原料的在产配方"})

        # Versions are frozen here: the job items carry concrete formula
        # versions and the job carries one concrete library version.
        job_id = self.scheduler.submit(
            library_version=library_version,
            scope={
                "reason": "reprice",
                "ingredient_ids": sorted(
                    set(ingredient_ids or
                        self.affected_ingredients(library_version))),
            },
            formula_targets=[(t["formula_id"], t["formula_version"])
                             for t in targets],
        )
        return {"job_id": job_id, "library_version": library_version,
                "target_count": len(targets), "targets": targets}
