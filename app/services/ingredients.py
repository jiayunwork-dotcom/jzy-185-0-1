"""Ingredient library service: draft editing + immutable versions."""
from __future__ import annotations

from typing import Any

from ..errors import NotFoundError, ValidationError
from ..storage import Database
from ..validation import validate_ingredient


class IngredientService:
    def __init__(self, db: Database):
        self.db = db

    def upsert(self, payload: dict[str, Any]) -> dict[str, Any]:
        validate_ingredient(payload)
        self.db.upsert_draft_ingredient(payload)
        row = self.db.get_draft_ingredient(payload["ingredient_id"])
        assert row is not None
        return row

    def bulk_upsert(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(items, list) or not items:
            raise ValidationError("ingredients must be a non-empty list",
                                  {"ingredients": "列表不能为空"})
        for item in items:
            validate_ingredient(item)
        for item in items:
            self.db.upsert_draft_ingredient(item)
        return self.list_draft()

    def delete_draft(self, ingredient_id: str) -> None:
        if not self.db.delete_draft_ingredient(ingredient_id):
            raise NotFoundError("ingredient", ingredient_id)

    def list_draft(self) -> list[dict[str, Any]]:
        return self.db.list_draft_ingredients()

    def get_draft(self, ingredient_id: str) -> dict[str, Any]:
        row = self.db.get_draft_ingredient(ingredient_id)
        if row is None:
            raise NotFoundError("ingredient", ingredient_id)
        return row

    def publish(self, note: str | None = None) -> dict[str, Any]:
        draft = self.db.list_draft_ingredients()
        if not draft:
            from ..errors import ValidationError
            raise ValidationError("cannot publish an empty ingredient library",
                                  {"ingredients": "草稿为空，无法发布版本"})
        version = self.db.publish_library(note)
        return {"version": version, "ingredient_count": len(draft),
                "ingredients": self.db.list_snapshot_ingredients(version)}

    def list_versions(self) -> list[dict[str, Any]]:
        return self.db.list_library_versions()

    def get_version(self, version: int | None = None) -> dict[str, Any]:
        if version is None:
            version = self.db.latest_library_version()
            if version is None:
                raise NotFoundError("library_version", "latest")
        if not self.db.library_version_exists(version):
            raise NotFoundError("library_version", version)
        return {"version": version,
                "ingredients": self.db.list_snapshot_ingredients(version)}

    def diff_versions(self, old: int, new: int) -> dict[str, Any]:
        for v in (old, new):
            if not self.db.library_version_exists(v):
                raise NotFoundError("library_version", v)
        return self.db.diff_library_versions(old, new)
