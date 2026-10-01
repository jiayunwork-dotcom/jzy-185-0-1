"""Input validation with field-path annotated errors."""
from __future__ import annotations

import math
from typing import Any

from .errors import ValidationError


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _finite(v: Any) -> bool:
    return _is_num(v) and math.isfinite(float(v))


def validate_ingredient(payload: dict[str, Any]) -> None:
    err = ValidationError("invalid ingredient")
    iid = payload.get("ingredient_id")
    if not isinstance(iid, str) or not iid.strip():
        err.add("ingredient_id", "原料 ID 不能为空")

    if not isinstance(payload.get("name"), str) or not payload["name"].strip():
        err.add("name", "名称不能为空")

    price = payload.get("price")
    if not _finite(price):
        err.add("price", "单价必须是有限数值")
    elif float(price) < 0:
        err.add("price", "单价不能为负")

    dm = payload.get("dry_matter")
    if dm is None:
        err.add("dry_matter", "干物质含量必填")
    elif not _finite(dm):
        err.add("dry_matter", "干物质必须是有限数值")
    elif not (0.0 <= float(dm) <= 1.0):
        err.add("dry_matter", "干物质应在 0 到 1 之间")

    nutrients = payload.get("nutrients", {})
    if not isinstance(nutrients, dict):
        err.add("nutrients", "营养含量必须是 code -> 数值 的映射")
    else:
        for code, val in nutrients.items():
            field = f"nutrients.{code}"
            if not isinstance(code, str) or not code:
                err.add("nutrients", "营养项目 code 必须是非空字符串")
            elif not _finite(val):
                err.add(field, "营养含量必须是有限数值")
            elif float(val) < 0:
                err.add(field, "营养含量不能为负")
    if err.fields:
        raise err


def validate_formula_spec(
    spec: dict[str, Any],
    *,
    known_ingredient_ids: set[str] | None = None,
    known_nutrient_codes: set[str] | None = None,
    snapshot_nutrients: dict[str, dict[str, float]] | None = None,
) -> None:
    """Validate a formula specification.

    Structural checks (bounds ordering, inclusion ranges, ratio shape) always
    run.  Reference checks against the ingredient library run when the
    corresponding known set is supplied: at create/update time only the
    ingredient IDs can be validated, while full nutrient/ratio validation
    happens against a concrete library snapshot.
    """
    err = ValidationError("invalid formula specification")

    ingredients = spec.get("ingredients")
    if not isinstance(ingredients, list) or not ingredients:
        err.add("ingredients", "配方至少包含一种原料")
        ingredients = []

    seen: set[str] = set()
    resolved: dict[str, dict[str, float]] = {}
    for i, item in enumerate(ingredients):
        if not isinstance(item, dict):
            err.add(f"ingredients[{i}]", "原料条目必须是对象")
            continue
        iid = item.get("ingredient_id")
        field = f"ingredients[{i}].ingredient_id"
        if not isinstance(iid, str) or not iid:
            err.add(field, "原料 ID 不能为空")
            continue
        if iid in seen:
            err.add(field, f"原料 {iid} 重复出现")
        seen.add(iid)
        if known_ingredient_ids is not None and iid not in known_ingredient_ids:
            err.add(field, f"引用了不存在的原料: {iid}")

        lo = item.get("min", 0.0)
        hi = item.get("max", 1.0)
        if not _finite(lo):
            err.add(f"ingredients[{i}].min", "最小比例必须是有限数值")
        elif not (0.0 <= float(lo) <= 1.0):
            err.add(f"ingredients[{i}].min", "添加比例必须在 0 到 1 之间")
        if not _finite(hi):
            err.add(f"ingredients[{i}].max", "最大比例必须是有限数值")
        elif not (0.0 <= float(hi) <= 1.0):
            err.add(f"ingredients[{i}].max", "添加比例必须在 0 到 1 之间")
        if _finite(lo) and _finite(hi) and float(lo) > float(hi):
            err.add(f"ingredients[{i}].max", "最大比例不能小于最小比例")
        if snapshot_nutrients is not None and iid in snapshot_nutrients:
            resolved[iid] = snapshot_nutrients[iid]

    nb = spec.get("nutrient_bounds", {})
    if not isinstance(nb, dict):
        err.add("nutrient_bounds", "营养约束必须是 code -> {min?,max?} 的映射")
        nb = {}
    for code, bounds in nb.items():
        field = f"nutrient_bounds.{code}"
        if not isinstance(bounds, dict):
            err.add(field, "营养约束必须是对象")
            continue
        if known_nutrient_codes is not None and code not in known_nutrient_codes:
            err.add(field, f"引用了不存在的营养项: {code}")
        lo = bounds.get("min")
        hi = bounds.get("max")
        if lo is not None:
            if not _finite(lo):
                err.add(f"{field}.min", "下限必须是有限数值")
            elif float(lo) < 0:
                err.add(f"{field}.min", "营养下限不能为负")
        if hi is not None:
            if not _finite(hi):
                err.add(f"{field}.max", "上限必须是有限数值")
            elif float(hi) < 0:
                err.add(f"{field}.max", "营养上限不能为负")
        if lo is not None and hi is not None and _finite(lo) and _finite(hi) \
                and float(lo) > float(hi):
            err.add(f"{field}.max", "上限不能低于下限")

    ratios = spec.get("ratios", [])
    if not isinstance(ratios, list):
        err.add("ratios", "比值约束必须是列表")
        ratios = []
    ratio_ids: set[str] = set()
    for i, r in enumerate(ratios):
        pre = f"ratios[{i}]"
        if not isinstance(r, dict):
            err.add(pre, "比值约束必须是对象")
            continue
        rid = r.get("id", f"#{i}")
        if rid in ratio_ids:
            err.add(f"{pre}.id", f"比值约束 id 重复: {rid}")
        ratio_ids.add(rid)
        for key in ("numerator", "denominator"):
            val = r.get(key)
            if not isinstance(val, str) or not val:
                err.add(f"{pre}.{key}", f"{key} 必须是非空营养 code")
            elif known_nutrient_codes is not None \
                    and val not in known_nutrient_codes:
                err.add(f"{pre}.{key}", f"引用了不存在的营养项: {val}")
        lo = r.get("min")
        hi = r.get("max")
        if lo is None and hi is None:
            err.add(pre, "比值约束至少要给 min 或 max 之一")
        if lo is not None and (not _finite(lo) or float(lo) < 0):
            err.add(f"{pre}.min", "比值下限必须是非负有限数值")
        if hi is not None and (not _finite(hi) or float(hi) < 0):
            err.add(f"{pre}.max", "比值上限必须是非负有限数值")
        if lo is not None and hi is not None and _finite(lo) and _finite(hi) \
                and float(lo) > float(hi):
            err.add(f"{pre}.max", "比值上限不能低于下限")

        # denominator-zero check against concrete snapshot
        if (snapshot_nutrients is not None and resolved
                and isinstance(r.get("denominator"), str)):
            den_code = r["denominator"]
            num_code = r.get("numerator")
            codes_ok = all(
                isinstance(c, str) and any(
                    abs(float(ing.get(c, 0.0) or 0.0)) > 0.0
                    for ing in resolved.values())
                for c in (den_code, num_code) if isinstance(c, str))
            if not codes_ok:
                err.add(
                    f"{pre}.denominator",
                    f"比值约束的分母营养 {den_code} 在所有允许原料中均为零")

    if err.fields:
        raise err
