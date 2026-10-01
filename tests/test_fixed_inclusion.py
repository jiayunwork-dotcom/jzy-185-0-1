"""Fixed inclusions (min == max) and heavy degeneracy must solve."""
from __future__ import annotations

from .conftest import CORN, SBM, create_formula


def _third_ingredient(client):
    client.put("/ingredients/wheat", json={
        "ingredient_id": "wheat", "name": "小麦", "price": 0.40,
        "dry_matter": 0.89,
        "nutrients": {"CP": 20.0, "ME": 3.1, "Ca": 0.05, "P": 0.32}})


def test_fixed_inclusion_formula(client):
    _third_ingredient(client)
    for ing in (CORN, SBM):
        client.put(f"/ingredients/{ing['ingredient_id']}", json=ing)
    client.post("/library/publish", json={})

    spec = {
        "ingredients": [
            {"ingredient_id": "corn", "min": 0.5, "max": 0.5},
            {"ingredient_id": "sbm", "min": 0.2, "max": 0.2},
            {"ingredient_id": "wheat", "min": 0.3, "max": 0.3},
        ],
        "nutrient_bounds": {"CP": {"min": 15.0}},
        "ratios": [],
    }
    fid, _ = create_formula(client, spec)
    d = client.post(f"/formulas/{fid}/optimize", json={}).json()
    assert d["status"] == "ok"
    qty = {x["ingredient_id"]: x["quantity"] for x in d["ingredients"]}
    assert qty == {"corn": 0.5, "sbm": 0.2, "wheat": 0.3}
    cert = d["certificate"]
    assert cert["primal_feasible"]
    assert cert["dual_feasible"]
    assert cert["complementary_slackness"]
    assert cert["relative_gap"] < 1e-9


def test_partial_fixed_inclusions(client):
    _third_ingredient(client)
    for ing in (CORN, SBM):
        client.put(f"/ingredients/{ing['ingredient_id']}", json=ing)
    client.post("/library/publish", json={})

    spec = {
        "ingredients": [
            {"ingredient_id": "corn", "min": 0.3, "max": 0.3},
            {"ingredient_id": "sbm", "min": 0.0, "max": 1.0},
            {"ingredient_id": "wheat", "min": 0.0, "max": 1.0},
        ],
        "nutrient_bounds": {"CP": {"min": 16.0}},
        "ratios": [],
    }
    fid, _ = create_formula(client, spec)
    d = client.post(f"/formulas/{fid}/optimize", json={}).json()
    assert d["status"] == "ok"
    qty = {x["ingredient_id"]: x["quantity"] for x in d["ingredients"]}
    assert abs(qty["corn"] - 0.3) < 1e-9
    assert abs(sum(qty.values()) - 1.0) < 1e-9
    cp = 8*qty["corn"] + 44*qty["sbm"] + 20*qty["wheat"]
    assert cp >= 16.0 - 1e-8
    assert d["certificate"]["relative_gap"] < 1e-9
