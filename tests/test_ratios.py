"""Ratio constraints (Ca/P window) work end to end."""
from __future__ import annotations

from .conftest import (CORN, FISH, LIMESTONE, SBM, create_formula,
                       publish_basic_library)


def test_calcium_phosphorus_ratio_enforced(client):
    publish_basic_library(client, (CORN, SBM, FISH, LIMESTONE))
    spec = {
        "ingredients": [
            {"ingredient_id": "corn", "min": 0.0, "max": 0.8},
            {"ingredient_id": "sbm", "min": 0.0, "max": 0.6},
            {"ingredient_id": "fish", "min": 0.0, "max": 0.2},
            {"ingredient_id": "lime", "min": 0.0, "max": 0.05},
        ],
        "nutrient_bounds": {"CP": {"min": 18.0}},
        "ratios": [
            {"id": "cap", "numerator": "Ca", "denominator": "P",
             "min": 1.2, "max": 2.0},
        ],
    }
    fid, _ = create_formula(client, spec)
    d = client.post(f"/formulas/{fid}/optimize", json={}).json()
    assert d["status"] == "ok"

    qty = {x["ingredient_id"]: x["quantity"] for x in d["ingredients"]}
    ca = 0.02*qty["corn"] + 0.30*qty["sbm"] + 4.0*qty["fish"] \
        + 38.0*qty["lime"]
    p = 0.27*qty["corn"] + 0.65*qty["sbm"] + 2.8*qty["fish"]
    ratio = ca / p
    assert 1.2 - 1e-8 <= ratio <= 2.0 + 1e-8

    ratio_rows = [c for c in d["constraints"]
                  if c["kind"] in ("ratio_low", "ratio_high")]
    assert len(ratio_rows) == 2
    # at least one side of the ratio window should bind in a minimum-Ca feed
    assert any(r["active"] for r in ratio_rows)
    # certificate stands
    cert = d["certificate"]
    assert cert["primal_feasible"] and cert["dual_feasible"]
    assert cert["complementary_slackness"]
    assert cert["relative_gap"] < 1e-9


def test_infeasible_ratio_conflict(client):
    # demand Ca/P >= 50 when ingredients cannot provide it (no limestone room)
    publish_basic_library(client, (CORN, SBM))
    spec = {
        "ingredients": [
            {"ingredient_id": "corn", "min": 0.0, "max": 1.0},
            {"ingredient_id": "sbm", "min": 0.0, "max": 1.0},
        ],
        "nutrient_bounds": {},
        "ratios": [
            {"id": "cap", "numerator": "Ca", "denominator": "P",
             "min": 50.0},
        ],
    }
    fid, _ = create_formula(client, spec)
    r = client.post(f"/formulas/{fid}/optimize", json={})
    assert r.status_code == 409
    codes = {c["code"] for c in r.json()["conflicts"]}
    assert "ratio_lower_infeasible" in codes
