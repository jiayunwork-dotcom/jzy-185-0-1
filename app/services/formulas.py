"""Formula specification service: versioned create / update."""
from __future__ import annotations

from typing import Any

from ..errors import NotFoundError, ValidationError
from ..storage import Database
from ..validation import validate_formula_spec


class FormulaService:
    def __init__(self, db: Database):
        self.db = db

    def _draft_ingredient_ids(self) -> set[str]:
        return {r["ingredient_id"] for r in self.db.list_draft_ingredients()}

    def create(self, name: str, spec: dict[str, Any],
               note: str | None = None) -> dict[str, Any]:
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("name required", {"name": "名称不能为空"})
        # Structural validation + ingredient reference validation against the
        # current draft. Nutrient references are validated lazily against the
        # library snapshot at optimisation time, because a formula may
        # outlive several library versions.
        validate_formula_spec(
            spec, known_ingredient_ids=self._draft_ingredient_ids())
        fid, version = self.db.create_formula(name, spec, note)
        return self.get_spec(fid, version)

    def update_spec(self, formula_id: str, spec: dict[str, Any],
                    note: str | None = None, name: str | None = None
                    ) -> dict[str, Any]:
        if self.db.get_formula(formula_id) is None:
            raise NotFoundError("formula", formula_id)
        validate_formula_spec(
            spec, known_ingredient_ids=self._draft_ingredient_ids())
        version = self.db.add_formula_version(formula_id, spec, note)
        if name:
            self.db.rename_formula(formula_id, name)
        return self.get_spec(formula_id, version)

    def list(self) -> list[dict[str, Any]]:
        return self.db.list_formulas()

    def get(self, formula_id: str) -> dict[str, Any]:
        row = self.db.get_formula(formula_id)
        if row is None:
            raise NotFoundError("formula", formula_id)
        row = dict(row)
        row["latest_version"] = self.db.latest_formula_version(formula_id)
        return row

    def get_spec(self, formula_id: str,
                 version: int | None = None) -> dict[str, Any]:
        row = self.db.get_formula_spec(formula_id, version)
        if row is None:
            raise NotFoundError("formula_version",
                                f"{formula_id}@{version}")
        return row

    def list_versions(self, formula_id: str) -> list[dict[str, Any]]:
        if self.db.get_formula(formula_id) is None:
            raise NotFoundError("formula", formula_id)
        return self.db.list_formula_versions(formula_id)
