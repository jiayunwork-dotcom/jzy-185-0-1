"""Price sensitivity: composition stays inside the interval and changes once
the price leaves it."""
from __future__ import annotations

from .conftest import CORN, SBM, create_formula


def _spec():
    return {
        "ingredients": [
            {"ingredient_id": "corn", "min": 0.0, "max": 1.0},
            {"ingredient_id": "sbm", "min": 0.0, "max": 1.0},
        ],
        "nutrient_bounds": {"CP": {"min": 18.0}},
        "ratios": [],
    }


def _reopt(client, fid, lv):
    return client.post(
        f"/formulas/{fid}/optimize",
        json={"library_version": lv}).json()


def _set_prices_and_publish(client, corn_p, sbm_p):
    for ing, p in ((CORN, corn_p), (SBM, sbm_p)):
        y = dict(ing)
        y["price"] = p
        r = client.put(f"/ingredients/{ing['ingredient_id']}", json=y)
        assert r.status_code == 200
    return client.post("/library/publish", json={}).json()["version"]


def test_composition_stable_inside_range(client):
    _set_prices_and_publish(client, 0.30, 0.50)
    fid, _ = create_formula(client, _spec())
    base = _reopt(client, fid, 1)
    ranges = {s["ingredient_id"]: s for s in base["sensitivity"]}

    # corn current 0.30, upper 0.50 -> 0.45 is inside
    lv = _set_prices_and_publish(client, 0.45, 0.50)
    d = _reopt(client, fid, lv)
    q0 = {x["ingredient_id"]: x["quantity"] for x in base["ingredients"]}
    q1 = {x["ingredient_id"]: x["quantity"] for x in d["ingredients"]}
    assert abs(q1["corn"] - q0["corn"]) < 1e-9
    assert abs(q1["sbm"] - q0["sbm"]) < 1e-9
    assert ranges["corn"]["price_high"] == 0.5
    assert ranges["sbm"]["price_low"] == 0.3


def test_composition_changes_outside_range(client):
    _set_prices_and_publish(client, 0.30, 0.50)
    fid, _ = create_formula(client, _spec())
    base = _reopt(client, fid, 1)
    q0 = {x["ingredient_id"]: x["quantity"] for x in base["ingredients"]}

    # corn above 0.5 -> switch fully to sbm (still satisfies CP>=18)
    lv = _set_prices_and_publish(client, 0.55, 0.50)
    d = _reopt(client, fid, lv)
    q1 = {x["ingredient_id"]: x["quantity"] for x in d["ingredients"]}
    assert abs(q1["corn"] - q0["corn"]) > 1e-6
    assert q1["sbm"] > 0.999999

    # sbm below 0.3 (cheaper than corn) -> use all sbm too
    lv = _set_prices_and_publish(client, 0.30, 0.25)
    d = _reopt(client, fid, lv)
    q2 = {x["ingredient_id"]: x["quantity"] for x in d["ingredients"]}
    assert abs(q2["sbm"] - 1.0) < 1e-9
