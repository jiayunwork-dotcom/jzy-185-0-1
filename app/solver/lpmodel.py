"""Build the standard-form LP of a feed formulation problem.

Decision variables (one per raw material, 1 kg of finished feed total):

    x_i = kilograms of ingredient i        (min_i <= x_i <= max_i)

All original rows are normalised so that the *total mass* coefficient and the
right-hand side keep a consistent representation. Rows::

    mass equality:    sum_i x_i                       = 1
    nutrient lower:   sum_i n_{ik} x_i  >= lb_k        (may be omitted at 0)
    nutrient upper:   sum_i n_{ik} x_i  <= ub_k
    ratio (e.g. Ca/P):
        lb * sum_i d_{ik} x_i <= sum_i a_{ik} x_i <= ub * sum_i d x_i
    ingredient min:   x_i >= min_i   ->   -x_i + s = -min_i
    ingredient max:   x_i <= max_i   ->    x_i + s =  max_i

Bounds rows are emitted even at 0/1 because their slack carries a dual
variable used when presenting the result.

Each row records enough metadata to map canonical dual variables back to
shadow prices of the original constraint.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simplex import FEAS_TOL


@dataclass
class RowMeta:
    """Metadata for one canonical equality row."""

    kind: str                 # 'mass' | 'nutrient' | 'ratio_low' | ... | 'min' | 'max'
    ref: str | None           # nutrient code / ratio id / ingredient id
    sense: str                # original sense: '=' | '>=' | '<='
    flip: float               # sign sigma: canonical row = sigma*(lhs vs rhs)
    rhs_original: float       # right-hand side of the original constraint


@dataclass
class StandardModel:
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    rows: list[RowMeta]
    ingredient_cols: list[int]
    ingredient_ids: list[str]
    slack_cols: list[int]
    artificial_cols: list[int]

    def structure_key(self) -> tuple:
        """Signature of the exact LP (rows, columns AND right-hand sides).

        Used for warm-start compatibility: a basis is only reusable when the
        matrix columns and the RHS vector are the same. RHS changes
        (different prices do not affect it, but different inclusion limits or
        nutrient limits do) can make the old basis infeasible; including the
        RHS here is a conservative, always-safe guard.
        """
        return (
            tuple(self.ingredient_ids),
            tuple((r.kind, r.ref, r.sense) for r in self.rows),
            tuple(float(v) for v in self.b),
        )


def _with_flip(meta: RowMeta, flip: float) -> RowMeta:
    return RowMeta(
        kind=meta.kind,
        ref=meta.ref,
        sense=meta.sense,
        flip=flip,
        rhs_original=meta.rhs_original,
    )


def build_model(
    ingredient_ids: list[str],
    prices: list[float],
    nutrients: dict[str, list[float]],
    dry_matter: list[float],
    mins: list[float],
    maxs: list[float],
    nutrient_bounds: dict[str, tuple[float | None, float | None]],
    ratios: list[dict],
) -> StandardModel:
    """Construct the standard LP.

    ``nutrients`` maps nutrient code -> value per kg for each ingredient.
    ``ratios`` is a list of dicts with keys id, numerator, denominator,
    min (optional), max (optional).
    """
    n = len(ingredient_ids)
    c = np.array(prices, dtype=float)

    pending: list[tuple[list[tuple[int, float]], float, str, RowMeta]] = []

    # ---- mass equality ---------------------------------------------------
    pending.append((
        [(i, 1.0) for i in range(n)], 1.0, "=",
        RowMeta("mass", None, "=", 1.0, 1.0),
    ))

    # ---- nutrient rows ---------------------------------------------------
    for code, (lb, ub) in nutrient_bounds.items():
        vals = nutrients[code]
        if lb is not None and lb > 0:
            pending.append((
                [(i, float(vals[i])) for i in range(n)],
                float(lb), ">=",
                RowMeta("nutrient", code, ">=", 1.0, float(lb)),
            ))
        if ub is not None and np.isfinite(ub):
            pending.append((
                [(i, float(vals[i])) for i in range(n)],
                float(ub), "<=",
                RowMeta("nutrient", code, "<=", 1.0, float(ub)),
            ))

    # ---- ratio rows ------------------------------------------------------
    for r in ratios:
        num = nutrients[r["numerator"]]
        den = nutrients[r["denominator"]]
        rng = range(n)
        if r.get("min") is not None:
            lb = float(r["min"])
            # lb * sum(den*x) <= sum(num*x)  ->  sum((num - lb*den)x) >= 0
            pending.append((
                [(i, float(num[i] - lb * den[i])) for i in rng],
                0.0, ">=",
                RowMeta("ratio_low", r["id"], ">=", 1.0, 0.0),
            ))
        if r.get("max") is not None:
            ub = float(r["max"])
            pending.append((
                [(i, float(ub * den[i] - num[i])) for i in rng],
                0.0, ">=",
                RowMeta("ratio_high", r["id"], ">=", 1.0, 0.0),
            ))

    # ---- bound rows (explicit, so every bound has a slack/dual) ----------
    # ingredient lower: x_i - s = min_i  (i.e. x_i >= min_i, slack sign -1)
    # ingredient upper: x_i + s = max_i  (i.e. x_i <= max_i, slack sign +1)
    for i, ing_id in enumerate(ingredient_ids):
        lo = float(mins[i])
        hi = float(maxs[i])
        pending.append((
            [(i, 1.0)], lo, ">=",
            RowMeta("min", ing_id, ">=", 1.0, lo),
        ))
        pending.append((
            [(i, 1.0)], hi, "<=",
            RowMeta("max", ing_id, "<=", 1.0, hi),
        ))

    # Materialise columns: ingredient cols [0, n), then one slack per row,
    # then one artificial per row that needs one.
    slack_cols: list[int] = []
    artificial_cols: list[int] = []
    rows: list[RowMeta] = []
    rhs_list: list[float] = []

    # First decide which rows need an artificial column (a row only has a
    # trivially feasible slack basis when its slack sign is +1 and RHS>=0).
    need_art: list[bool] = []
    for coeffs, rhs, sense, meta in pending:
        if sense == "=":
            need = True
        elif sense == ">=":
            need = rhs >= -FEAS_TOL
        else:  # <=
            need = rhs < -FEAS_TOL
        need_art.append(need)

    n_real = n + len(pending)
    n_total = n_real + sum(need_art)
    A = np.zeros((len(pending), n_total))
    art_index = 0
    for ridx, ((coeffs, rhs, sense, meta), need) in enumerate(
        zip(pending, need_art)
    ):
        for col, val in coeffs:
            A[ridx, col] = val
        # slack column
        slack_col = n + ridx
        slack_cols.append(slack_col)
        if sense == "=":
            slack_sign = 0.0
        elif sense == ">=":
            slack_sign = -1.0
        else:
            slack_sign = +1.0
        A[ridx, slack_col] = slack_sign
        flip = 1.0
        if sense != "=" and rhs < -FEAS_TOL:
            A[ridx, :] = -A[ridx, :]
            rhs = -rhs
            flip = -1.0
        if need:
            art_col = n_real + art_index
            art_index += 1
            A[ridx, art_col] = 1.0
            artificial_cols.append(art_col)
        rhs_list.append(rhs)
        rows.append(_with_flip(meta, flip))

    c_full = np.zeros(n_total)
    c_full[:n] = c

    return StandardModel(
        A=A,
        b=np.array(rhs_list, dtype=float),
        c=c_full,
        rows=rows,
        ingredient_cols=list(range(n)),
        ingredient_ids=list(ingredient_ids),
        slack_cols=slack_cols,
        artificial_cols=artificial_cols,
    )
