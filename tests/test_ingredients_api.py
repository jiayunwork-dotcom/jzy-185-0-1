"""Bulk draft editing and extensible nutrient catalogue."""
from __future__ import annotations

from .conftest import CORN, SBM, create_formula


def test_bulk_upsert_and_draft_crud(client):
    r = client.post("/ingredients/bulk", json={"ingredients": [CORN, SBM]})
    assert r.status_code == 200
    assert len(r.json()["ingredients"]) == 2

    r = client.get("/ingredients")
    assert {i["ingredient_id"] for i in r.json()["ingredients"]} == {
        "corn", "sbm"}

    r = client.delete("/ingredients/sbm")
    assert r.status_code == 204
    r = client.get("/ingredients")
    assert {i["ingredient_id"] for i in r.json()["ingredients"]} == {"corn"}
    assert client.get("/ingredients/sbm").status_code == 404


def test_new_nutrient_code_is_supported(client):
    # CP/ME known initially; publish a library introducing a new code "NSP"
    corn = dict(CORN)
    corn["nutrients"] = dict(CORN["nutrients"], NSP=9.0)
    sbm = dict(SBM)
    sbm["nutrients"] = dict(SBM["nutrients"], NSP=18.0)
    client.post("/ingredients/bulk", json={"ingredients": [corn, sbm]})
    v = client.post("/library/publish", json={}).json()["version"]

    spec = {
        "ingredients": [
            {"ingredient_id": "corn", "min": 0.0, "max": 1.0},
            {"ingredient_id": "sbm", "min": 0.0, "max": 1.0},
        ],
        "nutrient_bounds": {"CP": {"min": 18.0}, "NSP": {"min": 12.0}},
        "ratios": [],
    }
    fid, _ = create_formula(client, spec)
    d = client.post(
        f"/formulas/{fid}/optimize",
        json={"library_version": v}).json()
    assert d["status"] == "ok"
    qty = {x["ingredient_id"]: x["quantity"] for x in d["ingredients"]}
    nsp = 9.0 * qty["corn"] + 18.0 * qty["sbm"]
    assert nsp >= 12.0 - 1e-8
