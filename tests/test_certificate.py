"""Optimality certificate: primal/dual/CS checks and duality gap."""
from __future__ import annotations

from .conftest import (CORN, FISH, LIMESTONE, SBM, create_formula,
                       publish_basic_library)


def rich_spec():
    return {
        "ingredients": [
            {"ingredient_id": "corn", "min": 0.0, "max": 0.75},
            {"ingredient_id": "sbm", "min": 0.05, "max": 0.50},
            {"ingredient_id": "fish", "min": 0.0, "max": 0.15},
            {"ingredient_id": "lime", "min": 0.0, "max": 0.05},
        ],
        "nutrient_bounds": {
            "CP": {"min": 20.0},
            "ME": {"min": 2.9},
            "Ca": {"min": 0.8, "max": 1.2},
            "P": {"min": 0.5},
        },
        "ratios": [
            {"id": "cap", "numerator": "Ca", "denominator": "P",
             "min": 1.2, "max": 2.0},
        ],
    }


def test_certificate_on_rich_formula(client):
    publish_basic_library(client, (CORN, SBM, FISH, LIMESTONE))
    fid, _ = create_formula(client, rich_spec())
    d = client.post(f"/formulas/{fid}/optimize", json={}).json()
    assert d["status"] == "ok"

    cert = d["certificate"]
    assert cert["primal_feasible"] is True
    assert cert["dual_feasible"] is True
    assert cert["complementary_slackness"] is True
    assert cert["max_primal_residual"] < 1e-9
    assert cert["min_reduced_cost"] > -1e-9
    assert cert["max_cs_violation"] < 1e-9
    assert cert["relative_gap"] < 1e-9
    assert abs(cert["primal_objective"] - cert["dual_objective"]) < 1e-12

    # all ingredients add to 1 kg and nutrient constraints are truly met
    qty = {x["ingredient_id"]: x["quantity"] for x in d["ingredients"]}
    assert abs(sum(qty.values()) - 1.0) < 1e-9
    cp = 8*qty["corn"] + 44*qty["sbm"] + 62*qty["fish"]
    assert cp >= 20.0 - 1e-8
    me = 3.35*qty["corn"] + 2.6*qty["sbm"] + 3.0*qty["fish"]
    assert me >= 2.9 - 1e-8
    ca = 0.02*qty["corn"] + 0.3*qty["sbm"] + 4.0*qty["fish"] \
        + 38.0*qty["lime"]
    p = 0.27*qty["corn"] + 0.65*qty["sbm"] + 2.8*qty["fish"]
    assert 0.8 - 1e-8 <= ca <= 1.2 + 1e-8
    assert p >= 0.5 - 1e-8
    assert 1.2 - 1e-8 <= ca / p <= 2.0 + 1e-8

    # reduced costs non-negative; used ingredients are basic with rc ~ 0
    for row in d["reduced_costs"]:
        assert row["reduced_cost"] >= -1e-9
    for row in d["ingredients"]:
        if row["quantity"] > 1e-8:
            assert row["reduced_cost"] < 1e-8


def test_complementary_slackness_columns(client):
    publish_basic_library(client, (CORN, SBM, FISH, LIMESTONE))
    fid, _ = create_formula(client, rich_spec())
    d = client.post(f"/formulas/{fid}/optimize", json={}).json()
    for col in d["certificate"]["columns"]:
        # every column: value * reduced_cost == 0
        assert abs(col["product"]) < 1e-9
