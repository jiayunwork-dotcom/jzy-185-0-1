"""Two-phase revised simplex method implemented from scratch.

The only numerical dependency is NumPy (linear algebra for dense linear
systems). No LP/optimisation library is used.

The input is a standard-form linear program

    minimize  c @ x
    subject to A x = b
               x >= 0

where every equality row originates from a constraint expressed with a
non-negative slack variable::

    original  <=:   a x + s = b       (slack coefficient +1)
    original  >=:   a x - s = b       (slack coefficient -1)
    original  = :                       (no slack; needs an artificial col)

Rows whose right-hand side is negative also receive an artificial column
because the slack cannot serve as an initial basic variable there.

Degeneracy / cycling
--------------------
Pricing uses Bland's rule (smallest indexed entering column; smallest indexed
basic variable on ratio-test ties). Bland's rule is guaranteed to terminate
without cycling, which also makes every run bit-for-bit deterministic: the
same input always produces the same optimal basis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Numerical tolerances (input quantities are O(10) at most).
PRICING_TOL = 1e-9        # reduced cost considered negative / positive
RATIO_TOL = 1e-11         # direction coefficient considered positive
FEAS_TOL = 1e-8           # basic variable considered non-negative
PHASE1_TOL = 1e-8         # phase-I objective considered zero
MAX_ITER = 100_000


class SimplexError(RuntimeError):
    pass


class InfeasibleLP(SimplexError):
    """Raised when phase-I proves the standard LP has no feasible point."""


@dataclass
class SimplexOutcome:
    x: np.ndarray                 # full primal solution (all columns)
    basis: np.ndarray             # basic column index per row
    y: np.ndarray                 # dual variables on canonical rows
    reduced_costs: np.ndarray     # c - A^T y for every column
    iterations: int
    phase1_iterations: int
    dropped_rows: tuple[int, ...]


def _solve_once(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    initial_basis: np.ndarray,
    artificial_cols: set[int],
    candidate_cols: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, set[int]]:
    """Run primal revised simplex from a primal-feasible basis.

    ``candidate_cols`` limits which columns may enter (phase I: all columns;
    phase II: real + slack columns only, never artificials).

    Returns (x, basis, y, reduced_costs, iterations, zero_artificial_rows).
    The last value is non-empty only for phase I: rows that still carry an
    artificial basic variable at value zero (redundant rows to be removed or
    repaired before phase II).
    """
    m, n = A.shape
    basis = np.array(initial_basis, dtype=int)

    def basic_values(bas: np.ndarray) -> np.ndarray:
        B = A[:, bas]
        return np.linalg.solve(B, b)

    xb = basic_values(basis)
    iterations = 0
    zero_art_rows: set[int] = set()

    while True:
        if iterations > MAX_ITER:
            raise SimplexError("simplex iteration limit exceeded")
        iterations += 1

        B = A[:, basis]
        c_b = c[basis]
        # Dual variables: B^T y = c_b
        y = np.linalg.solve(B.T, c_b)
        rc = c - A.T @ y

        # Bland entering rule: smallest eligible column index.
        entering = -1
        for j in sorted(candidate_cols):
            if j in basis:
                continue
            if rc[j] < -PRICING_TOL:
                entering = j
                break
        if entering < 0:
            break  # optimal for the current objective

        d = np.linalg.solve(B, A[:, entering])

        # Ratio test, Bland tie-break on the *basic variable* column index.
        theta = np.inf
        leaving = -1
        tie_basic_col = np.inf
        for i in range(m):
            if d[i] > RATIO_TOL:
                ratio = xb[i] / d[i]
                if ratio < -FEAS_TOL:
                    raise SimplexError("basis lost primal feasibility")
                if ratio < 0:
                    ratio = 0.0
                if ratio < theta - FEAS_TOL or (
                    abs(ratio - theta) <= FEAS_TOL and basis[i] < tie_basic_col
                ):
                    theta = ratio
                    leaving = i
                    tie_basic_col = basis[i]

        if leaving < 0:
            # Defensive: a feed problem always contains a mass equality,
            # so a finite optimum must exist.
            raise SimplexError("linear program appears unbounded")

        xb = xb - theta * d
        xb[leaving] = theta
        # Remove tiny negative drift caused by round-off.
        if -FEAS_TOL < xb[leaving] < 0.0:
            xb[leaving] = 0.0
        basis[leaving] = entering

    # Recompute cleanly at the end.
    B = A[:, basis]
    y = np.linalg.solve(B.T, c[basis])
    xb = np.linalg.solve(B, b)
    xb = np.where(np.abs(xb) < FEAS_TOL, 0.0, xb)
    rc = c - A.T @ y

    x = np.zeros(n)
    x[basis] = xb

    for i, col in enumerate(basis):
        if col in artificial_cols and abs(xb[i]) <= PHASE1_TOL:
            zero_art_rows.add(i)

    return x, basis, y, rc, iterations, zero_art_rows


def solve_standard(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    initial_basis: np.ndarray | None,
    artificial_cols: list[int],
    dropable_rows: set[int] | None = None,
) -> SimplexOutcome:
    """Solve a standard-form LP with two-phase revised simplex.

    ``initial_basis`` (optional) is a warm-start basis; it is only accepted
    when it is primal feasible, otherwise a full phase-I cold start runs.

    ``dropable_rows`` names canonical rows that the caller already knows may
    be deleted if phase I finds them redundant (used internally for retries).
    """
    m, n = A.shape
    art_set = set(artificial_cols)
    dropped: list[int] = []
    phase1_total = 0

    # Repeatedly solve, dropping provably redundant rows when phase I ends on
    # a zero-valued artificial variable.
    guard = 0
    while True:
        guard += 1
        if guard > 2 * m + 4:
            raise SimplexError("simplex made no progress across restarts")
        rows_kept = np.array([i for i in range(m) if i not in dropped], dtype=int)
        Ar = A[rows_kept, :]
        br = b[rows_kept]
        mr = Ar.shape[0]

        def valid_basis(bas: np.ndarray) -> np.ndarray | None:
            """Return basic values if bas is a feasible nondegenerate basis."""
            if bas.shape != (mr,) or len(set(bas.tolist())) != mr:
                return None
            try:
                vals = np.linalg.solve(Ar[:, bas], br)
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(vals)) or np.any(vals < -FEAS_TOL):
                return None
            return np.where(np.abs(vals) < FEAS_TOL, 0.0, vals)

        # ---- choose a starting basis ------------------------------------
        basis: np.ndarray | None = None
        xb_start: np.ndarray | None = None
        if initial_basis is not None and not dropped:
            cand = np.asarray(initial_basis, dtype=int)
            vals = valid_basis(cand)
            if vals is not None:
                basis = cand.copy()
                xb_start = vals

        if basis is None:
            basis = np.empty(mr, dtype=int)
            for i in range(mr):
                # Prefer an artificial column where one exists for this row;
                # else a non-negative slack (unit column).
                art = [j for j in artificial_cols if Ar[i, j] == 1.0
                       and np.count_nonzero(Ar[:, j]) == 1]
                if art:
                    basis[i] = art[0]
                else:
                    unit = [j for j in range(n)
                            if Ar[i, j] != 0.0
                            and np.count_nonzero(Ar[:, j]) == 1
                            and Ar[i, j] > 0.0
                            and j not in art_set]
                    if not unit:
                        raise SimplexError(
                            f"row {i} has no usable initial basic column")
                    basis[i] = unit[0]

        # ---- phase I -----------------------------------------------------
        # A warm basis must only skip phase I if it is primal feasible AND
        # every basic variable is real: an artificial anywhere in the basis
        # (even value 0) requires phase I to certify feasibility / pivot it out.
        needs_phase1 = True
        if xb_start is not None and not (set(basis.tolist()) & art_set):
            needs_phase1 = False

        basis_for_phase2: np.ndarray | None = None
        if needs_phase1:
            c1 = np.zeros(n)
            c1[list(art_set)] = 1.0
            x1, b1, _, _, it1, zero_art_rows = _solve_once(
                Ar, br, c1, basis, art_set, set(range(n)))
            phase1_total += it1
            phase1_obj = float(x1[list(art_set)].sum()) if artificial_cols else 0.0
            if phase1_obj > PHASE1_TOL:
                raise InfeasibleLP(
                    f"phase-I objective {phase1_obj:.3e} proves infeasibility")

            if zero_art_rows:
                # Each zero-valued artificial marks a row redundant at this
                # degenerate point. We try to pivot every such artificial out
                # simultaneously; a candidate column for one row has to keep
                # the *whole* joint basis feasible (single-row feasible swaps
                # can fail jointly), so columns are chosen greedily with a
                # full-basis feasibility check after each selection. If the
                # joint swap cannot be completed, the rows that remain are
                # genuinely redundant and deleted; phase I then restarts cold
                # on the smaller system.
                zrows = sorted(zero_art_rows)
                b_work = b1.copy()

                def _feasible(bas: np.ndarray) -> bool:
                    if len(set(bas.tolist())) != mr:
                        return False
                    try:
                        vals = np.linalg.solve(Ar[:, bas], br)
                    except np.linalg.LinAlgError:
                        return False
                    return bool(np.all(np.isfinite(vals))
                                and np.all(vals >= -FEAS_TOL))

                stuck: list[int] = []
                for ri in zrows:
                    if int(b_work[ri]) not in art_set:
                        continue
                    others = set(b_work.tolist()) - {int(b_work[ri])}
                    chosen: int | None = None
                    for j in range(n):
                        if j in art_set or j in others:
                            continue
                        if abs(Ar[ri, j]) <= RATIO_TOL:
                            continue
                        trial = b_work.copy()
                        trial[ri] = j
                        if _feasible(trial):
                            chosen = j
                            break
                    if chosen is None:
                        stuck.append(ri)
                    else:
                        b_work[ri] = chosen

                if stuck:
                    global_rows = sorted({int(rows_kept[r]) for r in stuck})
                    if not all(dropable_rows is not None
                               and t in dropable_rows for t in global_rows):
                        raise SimplexError(
                            f"rows {global_rows} redundant but not dropable")
                    dropped.extend(global_rows)
                    initial_basis = None
                    continue

                # Joint feasible swap succeeded and the new basis contains no
                # artificial column: it directly certifies phase I, so skip
                # straight to phase II instead of rerunning phase I.
                if not (set(b_work.tolist()) & art_set):
                    basis_for_phase2 = b_work
                else:
                    basis = b_work
                    continue
            else:
                basis_for_phase2 = b1
        else:
            basis_for_phase2 = basis
        assert basis_for_phase2 is not None
        real_cols = {j for j in range(n) if j not in art_set}
        x, bfinal, y, rc, it2, _ = _solve_once(
            Ar, br, c, basis_for_phase2, art_set, real_cols)

        # Expand dual variables back to the original row ordering.
        y_full = np.zeros(m)
        y_full[rows_kept] = y

        return SimplexOutcome(
            x=x,
            basis=bfinal,
            y=y_full,
            reduced_costs=rc,
            iterations=phase1_total + it2,
            phase1_iterations=phase1_total,
            dropped_rows=tuple(dropped),
        )
