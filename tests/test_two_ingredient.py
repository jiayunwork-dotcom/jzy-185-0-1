"""Hand-computed two ingredient example: corn + soybean meal, CP >= 18%."""
from __future__ import annotations

from .conftest import CORN, SBM, create_formula, publish_basic_library


def spec(cp_min=18.0):
    return {
        "ingredients": [
            {"ingredient_id": "corn", "min": 0.0, "max": 1.0},
            {"ingredient_id": "sbm", "min": 0.0, "max": 1.0},
        ],
        "nutrient_bounds": {"CP": {"min": cp_min}},
        "ratios": [],
    }


def test_hand_computed_case(client):
    publish_basic_library(client, (CORN, SBM))
    fid, _ = create_formula(client, spec())

    r = client.post(f"/formulas/{fid}/optimize", json={})
    assert r.status_code == 200, r.text
    d = r.json()

    qty = {x["ingredient_id"]: x["quantity"] for x in d["ingredients"]}
    # sbm = (18-8)/(44-8) = 0.27777...
    assert abs(qty["sbm"] - 10.0 / 36.0) < 1e-9
    assert abs(qty["corn"] - 26.0 / 36.0) < 1e-9
    # cost = .3*26/36 + .5*10/36 = 12.8/36
    assert abs(d["cost"] - 12.8 / 36.0) < 1e-9
    assert abs(qty["corn"] + qty["sbm"] - 1.0) < 1e-10


def test_shadow_price_of_cp_bound(client):
    publish_basic_library(client, (CORN, SBM))
    fid, _ = create_formula(client, spec())
    d = client.post(f"/formulas/{fid}/optimize", json={}).json()

    cp = next(c for c in d["constraints"]
              if c["kind"] == "nutrient" and c["ref"] == "CP")
    assert cp["active"] is True
    # analytical shadow price: (0.5-0.3)/(44-8) = 0.005555... per %-kg
    assert abs(cp["shadow_price"] - 0.2 / 36.0) < 1e-9

    # mass equality is also active and has a positive shadow price
    mass = next(c for c in d["constraints"] if c["kind"] == "mass")
    assert mass["active"] is True
    assert mass["shadow_price"] > 0


def test_repeated_runs_are_identical(client):
    publish_basic_library(client, (CORN, SBM))
    fid, _ = create_formula(client, spec())
    a = client.post(f"/formulas/{fid}/optimize", json={}).json()
    b = client.post(f"/formulas/{fid}/optimize", json={}).json()
    assert a["cost"] == b["cost"]
    assert [x["quantity"] for x in a["ingredients"]] == \
           [x["quantity"] for x in b["ingredients"]]
