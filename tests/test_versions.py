"""Versioning behaviour: immutable snapshots, diffs, result bindings."""
from __future__ import annotations

from .conftest import CORN, SBM, create_formula, publish_basic_library


def test_library_versions_are_immitable_and_diffable(client):
    publish_basic_library(client, (CORN, SBM), note="w1")
    # update corn price + nutrient in draft, republish
    y = dict(CORN)
    y["price"] = 0.35
    y["nutrients"] = dict(CORN["nutrients"])
    y["nutrients"]["CP"] = 8.5
    client.put("/ingredients/corn", json=y)
    client.post("/library/publish", json={"note": "w2"})

    v1 = client.get("/library/versions/1").json()
    v2 = client.get("/library/versions/2").json()
    corn1 = next(i for i in v1["ingredients"] if i["ingredient_id"] == "corn")
    corn2 = next(i for i in v2["ingredients"] if i["ingredient_id"] == "corn")
    assert corn1["price"] == 0.30 and corn2["price"] == 0.35
    assert corn1["nutrients"]["CP"] == 8.0 and corn2["nutrients"]["CP"] == 8.5

    diff = client.get("/library/versions/1/diff/2").json()
    assert [c["ingredient_id"] for c in diff["changed"]] == ["corn"]
    assert diff["changed"][0]["old_price"] == 0.30
    assert diff["changed"][0]["new_price"] == 0.35
    assert diff["added"] == [] and diff["removed"] == []


def test_optimization_results_bind_concrete_versions(client):
    publish_basic_library(client, (CORN, SBM))
    spec = {
        "ingredients": [
            {"ingredient_id": "corn", "min": 0.0, "max": 1.0},
            {"ingredient_id": "sbm", "min": 0.0, "max": 1.0},
        ],
        "nutrient_bounds": {"CP": {"min": 18.0}},
        "ratios": [],
    }
    fid, fv1 = create_formula(client, spec)
    d1 = client.post(f"/formulas/{fid}/optimize", json={}).json()
    assert d1["library_version"] == 1
    assert d1["formula_version"] == fv1 == 1

    # new formula version (looser CP) -> result binds fv2 with lv1
    spec2 = {**spec, "nutrient_bounds": {"CP": {"min": 16.0}}}
    r = client.put(f"/formulas/{fid}", json={"spec": spec2})
    assert r.status_code == 201
    fv2 = r.json()["version"]
    d2 = client.post(f"/formulas/{fid}/optimize", json={}).json()
    assert d2["formula_version"] == fv2 == 2
    assert d2["library_version"] == 1
    assert d2["cost"] < d1["cost"]

    # can still optimise the OLD formula version against the SAME library
    old = client.post(
        f"/formulas/{fid}/optimize",
        json={"formula_version": 1, "library_version": 1}).json()
    assert abs(old["cost"] - d1["cost"]) < 1e-12

    # history keeps both, newest first, each with distinct binding
    hist = client.get(f"/formulas/{fid}/results").json()["results"]
    assert {row["formula_version"] for row in hist} == {1, 2}
    assert all(row["library_version"] == 1 for row in hist)


def test_unknown_version_404(client):
    publish_basic_library(client, (CORN, SBM))
    spec = {
        "ingredients": [
            {"ingredient_id": "corn", "min": 0.0, "max": 1.0},
            {"ingredient_id": "sbm", "min": 0.0, "max": 1.0},
        ],
        "nutrient_bounds": {"CP": {"min": 18.0}},
        "ratios": [],
    }
    fid, _ = create_formula(client, spec)
    r = client.post(f"/formulas/{fid}/optimize",
                    json={"library_version": 99})
    assert r.status_code == 404
    assert r.json()["resource"] == "library_version"
