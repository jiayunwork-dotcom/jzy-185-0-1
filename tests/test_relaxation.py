"""Relaxing an active constraint cannot increase optimal cost."""
from __future__ import annotations

from .conftest import CORN, SBM, create_formula, publish_basic_library


def _spec(cp_min):
    return {
        "ingredients": [
            {"ingredient_id": "corn", "min": 0.0, "max": 1.0},
            {"ingredient_id": "sbm", "min": 0.0, "max": 1.0},
        ],
        "nutrient_bounds": {"CP": {"min": cp_min}},
        "ratios": [],
    }


def test_relax_active_lower_bound(client):
    publish_basic_library(client, (CORN, SBM))
    fid, _ = create_formula(client, _spec(18.0))
    tight = client.post(f"/formulas/{fid}/optimize", json={}).json()
    assert tight["cost"] > 0.30  # forced to use some sbm

    # relax CP from 18 down to 8: corn alone reaches 8, CP constraint slack
    client.put(f"/formulas/{fid}", json={"spec": _spec(8.0)})
    relaxed = client.post(f"/formulas/{fid}/optimize", json={}).json()
    assert relaxed["cost"] <= tight["cost"] + 1e-12
    assert relaxed["cost"] < tight["cost"] - 1e-9

    qty = {x["ingredient_id"]: x["quantity"]
           for x in relaxed["ingredients"]}
    assert abs(qty["corn"] - 1.0) < 1e-9
    cp_row = next(c for c in relaxed["constraints"]
                  if c["kind"] == "nutrient" and c["ref"] == "CP")
    assert cp_row["active"] is False
    assert cp_row["shadow_price"] == 0.0


def test_tighten_active_bound_raises_cost(client):
    publish_basic_library(client, (CORN, SBM))
    fid, _ = create_formula(client, _spec(18.0))
    base = client.post(f"/formulas/{fid}/optimize", json={}).json()

    client.put(f"/formulas/{fid}", json={"spec": _spec(22.0)})
    tighter = client.post(f"/formulas/{fid}/optimize", json={}).json()
    assert tighter["cost"] >= base["cost"] - 1e-12
    assert tighter["cost"] > base["cost"] + 1e-9

    # cost increase matches shadow price * delta within first order
    cp = next(c for c in base["constraints"]
              if c["kind"] == "nutrient" and c["ref"] == "CP")
    predicted = cp["shadow_price"] * (22.0 - 18.0)
    assert abs((tighter["cost"] - base["cost"]) - predicted) < 1e-9
