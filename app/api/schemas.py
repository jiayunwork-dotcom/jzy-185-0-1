"""Pydantic request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


# ---- ingredients ------------------------------------------------------------
class IngredientIn(BaseModel):
    ingredient_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    price: float
    dry_matter: float
    nutrients: dict[str, float] = Field(default_factory=dict)


class BulkIngredientsIn(BaseModel):
    ingredients: list[IngredientIn] = Field(min_length=1)


class PublishIn(BaseModel):
    note: str | None = None


# ---- formulas ---------------------------------------------------------------
class IngredientUse(BaseModel):
    ingredient_id: str
    min: float = 0.0
    max: float = 1.0


class NutrientBound(BaseModel):
    min: float | None = None
    max: float | None = None


class RatioConstraint(BaseModel):
    id: str
    numerator: str
    denominator: str
    min: float | None = None
    max: float | None = None


class FormulaSpecIn(BaseModel):
    ingredients: list[IngredientUse] = Field(min_length=1)
    nutrient_bounds: dict[str, NutrientBound] = Field(default_factory=dict)
    ratios: list[RatioConstraint] = Field(default_factory=list)


class FormulaCreateIn(BaseModel):
    name: str = Field(min_length=1)
    spec: FormulaSpecIn
    note: str | None = None


class FormulaUpdateIn(BaseModel):
    spec: FormulaSpecIn
    note: str | None = None
    name: str | None = None


# ---- optimisation -----------------------------------------------------------
class OptimizeIn(BaseModel):
    library_version: int | None = None
    formula_version: int | None = None
    force_cold_start: bool = False


class RepriceIn(BaseModel):
    library_version: int | None = None
    ingredient_ids: list[str] | None = None
    formula_ids: list[str] | None = None
