"""Scaling invariance: multiplying every price by k>0 leaves quantities
unchanged while cost and shadow prices scale by exactly k."""
from __future__ import annotations

from .conftest import CORN, FISH, LIMESTONE, SBM, create_formula


def _spec():
    return {
        "ingredients": [
            {"ingredient_id": "corn", "min": 0.0, "max": 0.8},
            {"ingredient_id": "sbm", "min": 0.0, "max": 0.6},
            {"ingredient_id": "fish", "min": 0.0, "max": 0.2},
            {"ingredient_id": "lime", "min": 0.0, "max": 0.05},
        ],
        "nutrient_bounds": {"CP": {"min": 21.0},
                            "Ca": {"min": 0.7, "max": 1.3}},
        "ratios": [],
    }


def _optimize(client, fid, lv):
    return client.post(f"/formulas/{fid}/optimize",
                       json={"library_version": lv}).json()


def test_positive_price_scaling(client):
    scale = 3.7
    ing = [dict(CORN), dict(SBM), dict(FISH), dict(LIMESTONE)]
    for x in ing:
        client.put(f"/ingredients/{x['ingredient_id']}", json=x)
    v1 = client.post("/library/publish", json={"note": "v1"}).json()["version"]

    fid, _ = create_formula(client, _spec())
    d1 = _optimize(client, fid, v1)
    q1 = {x["ingredient_id"]: x["quantity"] for x in d1["ingredients"]}

    # republish with every price multiplied by the same positive factor
    for x in ing:
        y = dict(x)
        y["price"] = x["price"] * scale
        client.put(f"/ingredients/{y['ingredient_id']}", json=y)
    v2 = client.post("/library/publish", json={"note": "v2"}).json()["version"]

    # solve against the same formula spec using the new library version
    d2 = _optimize(client, fid, v2)
    q2 = {x["ingredient_id"]: x["quantity"] for x in d2["ingredients"]}

    for iid in q1:
        assert abs(q1[iid] - q2[iid]) < 1e-9, iid
    assert abs(d2["cost"] - d1["cost"] * scale) < 1e-9 * scale

    s1 = {c["index"]: c["shadow_price"] for c in d1["constraints"]}
    for c2 in d2["constraints"]:
        assert abs(c2["shadow_price"] - s1[c2["index"]] * scale) < 1e-8 * scale

    # reduced costs scale too
    rc1 = {r["ingredient_id"]: r["reduced_cost"]
           for r in d1["reduced_costs"]}
    for r in d2["reduced_costs"]:
        assert abs(r["reduced_cost"] - rc1[r["ingredient_id"]] * scale) \
            < 1e-8 * scale
