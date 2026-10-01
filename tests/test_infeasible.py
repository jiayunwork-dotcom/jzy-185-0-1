"""Infeasible formulas must return a concrete conflicting constraint set."""
from __future__ import annotations

from .conftest import CORN, SBM, create_formula, publish_basic_library


def test_nutrient_lower_above_max_achievable(client):
    publish_basic_library(client, (CORN, SBM))
    spec = {
        "ingredients": [
            {"ingredient_id": "corn", "min": 0.0, "max": 1.0},
            {"ingredient_id": "sbm", "min": 0.0, "max": 1.0},
        ],
        "nutrient_bounds": {"CP": {"min": 60.0}},
        "ratios": [],
    }
    fid, _ = create_formula(client, spec)
    r = client.post(f"/formulas/{fid}/optimize", json={})
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "infeasible"
    codes = {c["code"] for c in body["conflicts"]}
    assert "nutrient_lower_unattainable" in codes
    c = next(c for c in body["conflicts"]
             if c["code"] == "nutrient_lower_unattainable")
    assert c["required"] == 60.0
    assert c["attainable_max"] == 44.0
    # witness names the highest contributor
    assert any(w["ingredient_id"] == "sbm" for w in c["witness"])
    assert c["constraints"] == ["CP>=60"]


def test_max_inclusion_sum_below_one(client):
    publish_basic_library(client, (CORN, SBM))
    spec = {
        "ingredients": [
            {"ingredient_id": "corn", "min": 0.0, "max": 0.4},
            {"ingredient_id": "sbm", "min": 0.0, "max": 0.3},
        ],
        "nutrient_bounds": {},
        "ratios": [],
    }
    fid, _ = create_formula(client, spec)
    r = client.post(f"/formulas/{fid}/optimize", json={})
    assert r.status_code == 409
    codes = {c["code"] for c in r.json()["conflicts"]}
    assert "max_inclusion_below_total" in codes


def test_min_inclusion_sum_above_one(client):
    publish_basic_library(client, (CORN, SBM))
    spec = {
        "ingredients": [
            {"ingredient_id": "corn", "min": 0.7, "max": 1.0},
            {"ingredient_id": "sbm", "min": 0.6, "max": 1.0},
        ],
        "nutrient_bounds": {},
        "ratios": [],
    }
    fid, _ = create_formula(client, spec)
    r = client.post(f"/formulas/{fid}/optimize", json={})
    assert r.status_code == 409
    codes = {c["code"] for c in r.json()["conflicts"]}
    assert "min_inclusion_above_total" in codes


def test_infeasible_result_persisted_in_history(client):
    publish_basic_library(client, (CORN, SBM))
    spec = {
        "ingredients": [
            {"ingredient_id": "corn", "min": 0.0, "max": 1.0},
            {"ingredient_id": "sbm", "min": 0.0, "max": 1.0},
        ],
        "nutrient_bounds": {"CP": {"min": 60.0}},
        "ratios": [],
    }
    fid, _ = create_formula(client, spec)
    r = client.post(f"/formulas/{fid}/optimize", json={})
    assert r.status_code == 409
    hist = client.get(f"/formulas/{fid}/results").json()["results"]
    assert hist[0]["status"] == "infeasible"
    assert hist[0]["conflicts"]
