"""Unit tests for the infeasibility diagnostics layer itself."""
from __future__ import annotations

import numpy as np

from app.solver.diagnose import diagnose, structural_checks
from app.solver.simplex import solve_standard
from app.solver.lpmodel import build_model


def test_structural_nutrient_lower_conflict():
    conflicts = structural_checks(
        ingredient_ids=["a", "b"],
        nutrients={"CP": [10.0, 20.0]},
        mins=[0.0, 0.0], maxs=[1.0, 1.0],
        nutrient_bounds={"CP": (25.0, None)},
        ratios=[])
    assert conflicts and conflicts[0]["code"] == "nutrient_lower_unattainable"


def test_structural_box_sum_conflicts():
    c = structural_checks(
        ["a", "b"], {}, [0.0, 0.0], [0.4, 0.4], {}, [])
    assert {x["code"] for x in c} == {"max_inclusion_below_total"}

    c = structural_checks(
        ["a", "b"], {}, [0.7, 0.6], [1.0, 1.0], {}, [])
    assert {x["code"] for x in c} == {"min_inclusion_above_total"}


def test_deletion_filter_finds_minimal_set():
    # Individually fine, jointly infeasible:
    #   sum x = 1; nutrient N average >= 100 while every ingredient carries
    #   exactly N=0; and a mandatory inclusion. Constructed so the greedy
    #   structural check finds the nutrient one first.
    ids = ["a", "b"]
    nutrients = {"N": [0.0, 0.0]}
    conflicts = diagnose(
        ingredient_ids=ids, prices=[1.0, 1.0], nutrients=nutrients,
        mins=[0.5, 0.5], maxs=[1.0, 1.0],
        nutrient_bounds={"N": (100.0, None)}, ratios=[])
    assert conflicts
    # at least one explanation must mention the nutrient lower bound
    labels = {lab for c in conflicts for lab in c.get("constraints", [])}
    assert any("N>=" in lab for lab in labels)


def test_deletion_filter_on_bound_vs_mass_conflict():
    # mins consistent-ish individually, but force mins via deletion filter
    # when structural pre-checks are bypassed: directly call diagnose with
    # a box whose mins exceed 1 (structural catches it, labels both sides).
    conflicts = diagnose(
        ["a", "b"], [1.0, 1.0], {}, [0.6, 0.6], [1.0, 1.0], {}, [])
    assert any(c["code"] == "min_inclusion_above_total" for c in conflicts)


def test_feasible_problem_has_no_conflict():
    m = build_model(
        ["a", "b"], [1.0, 2.0], {"N": [5.0, 15.0]}, [0.9, 0.9],
        [0.0, 0.0], [1.0, 1.0], {"N": (10.0, None)}, [])
    out = solve_standard(m.A, m.b, m.c, None, m.artificial_cols,
                         dropable_rows=set(range(len(m.rows))))
    # cheapest feasible: 0.5/0.5 mix (N average exactly 10), cost 1.5
    assert abs(out.x[0] - 0.5) < 1e-9 and abs(out.x[1] - 0.5) < 1e-9
    assert abs(float(np.array([1.0, 2.0]) @ out.x[:2]) - 1.5) < 1e-9
    # and structural checks agree nothing is wrong
    assert structural_checks(
        ["a", "b"], {"N": [5.0, 15.0]}, [0.0, 0.0], [1.0, 1.0],
        {"N": (10.0, None)}, []) == []
