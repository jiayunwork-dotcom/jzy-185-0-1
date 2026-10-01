"""Field-name annotated validation errors."""
from __future__ import annotations

from .conftest import CORN, SBM, create_formula, publish_basic_library


def _put(client, iid, **overrides):
    body = dict(CORN)
    body["ingredient_id"] = iid
    body.update(overrides)
    return client.put(f"/ingredients/{iid}", json=body)


def test_ingredient_field_errors(client):
    r = _put(client, "x", price=-1.0)
    assert r.status_code == 422
    assert "price" in r.json()["fields"]

    # non-finite values must arrive as numbers: send raw JSON
    r = client.put(
        "/ingredients/x",
        content=b'{"ingredient_id":"x","name":"x","price":NaN,'
                b'"dry_matter":0.9,"nutrients":{}}',
        headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert "price" in r.json()["fields"]

    r = client.put(
        "/ingredients/x",
        content=b'{"ingredient_id":"x","name":"x","price":1e999,'
                b'"dry_matter":0.9,"nutrients":{}}',
        headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert "price" in r.json()["fields"]

    r = _put(client, "x", nutrients={"CP": -3.0})
    assert r.status_code == 422
    assert "nutrients.CP" in r.json()["fields"]

    r = client.put(
        "/ingredients/x",
        content=b'{"ingredient_id":"x","name":"x","price":1.0,'
                b'"dry_matter":0.9,"nutrients":{"CP":"lots"}}',
        headers={"content-type": "application/json"})
    assert r.status_code == 422
    body = r.json()
    assert "fields" in body  # our envelope or pydantic's detail list
    assert "nutrients.CP" in body["fields"]

    r = _put(client, "x", name="")
    assert r.status_code == 422
    assert "name" in r.json()["fields"]

    r = _put(client, "x", dry_matter=1.5)
    assert r.status_code == 422
    assert "dry_matter" in r.json()["fields"]


def test_formula_bound_field_errors(client):
    publish_basic_library(client, (CORN, SBM))

    def post(spec):
        return client.post("/formulas", json={"name": "f", "spec": spec})

    base = [
        {"ingredient_id": "corn", "min": 0.0, "max": 1.0},
        {"ingredient_id": "sbm", "min": 0.0, "max": 1.0},
    ]

    r = post({"ingredients": [{**base[0], "min": 1.5}, base[1]],
              "nutrient_bounds": {}, "ratios": []})
    assert r.status_code == 422
    assert "ingredients[0].min" in r.json()["fields"]

    r = post({"ingredients": [{**base[0], "min": 0.8, "max": 0.2}, base[1]],
              "nutrient_bounds": {}, "ratios": []})
    assert r.status_code == 422
    assert "ingredients[0].max" in r.json()["fields"]

    r = post({"ingredients": [{"ingredient_id": "ghost",
                               "min": 0.0, "max": 1.0}],
              "nutrient_bounds": {}, "ratios": []})
    assert r.status_code == 422
    assert "ingredients[0].ingredient_id" in r.json()["fields"]

    r = post({"ingredients": base,
              "nutrient_bounds": {"CP": {"min": 30.0, "max": 10.0}},
              "ratios": []})
    assert r.status_code == 422
    assert "nutrient_bounds.CP.max" in r.json()["fields"]

    r = post({"ingredients": base, "nutrient_bounds": {},
              "ratios": [{"id": "r1", "numerator": "Ca",
                          "denominator": "P", "min": 2.0, "max": 1.0}]})
    assert r.status_code == 422
    assert "ratios[0].max" in r.json()["fields"]


def test_unknown_nutrient_rejected_at_optimize(client):
    publish_basic_library(client, (CORN, SBM))
    fid, _ = create_formula(client, {
        "ingredients": [
            {"ingredient_id": "corn", "min": 0.0, "max": 1.0},
            {"ingredient_id": "sbm", "min": 0.0, "max": 1.0},
        ],
        "nutrient_bounds": {"GHOST": {"min": 1.0}},
        "ratios": []})
    r = client.post(f"/formulas/{fid}/optimize", json={})
    assert r.status_code == 422
    assert "nutrient_bounds.GHOST" in r.json()["fields"]


def test_ratio_denominator_all_zero_rejected(client):
    # ingredients that both have zero Ca
    c = dict(CORN); c["nutrients"] = dict(CORN["nutrients"])
    c["nutrients"]["Ca"] = 0.0
    s = dict(SBM); s["nutrients"] = dict(SBM["nutrients"])
    s["nutrients"]["Ca"] = 0.0
    publish_basic_library(client, (c, s))
    fid, _ = create_formula(client, {
        "ingredients": [
            {"ingredient_id": "corn", "min": 0.0, "max": 1.0},
            {"ingredient_id": "sbm", "min": 0.0, "max": 1.0},
        ],
        "nutrient_bounds": {},
        "ratios": [{"id": "cap", "numerator": "P", "denominator": "Ca",
                    "min": 1.2, "max": 2.0}],
    })
    r = client.post(f"/formulas/{fid}/optimize", json={})
    assert r.status_code == 422
    fields = r.json()["fields"]
    assert any(k.startswith("ratios[0].denominator") for k in fields)
