"""Warm start vs cold start must yield identical optimal cost."""
from __future__ import annotations

from .conftest import (CORN, FISH, LIMESTONE, SBM, create_formula)


def _publish(client, prices):
    for ing, price in zip((CORN, SBM, FISH, LIMESTONE), prices):
        y = dict(ing)
        y["price"] = price
        r = client.put(f"/ingredients/{ing['ingredient_id']}", json=y)
        assert r.status_code == 200
    return client.post("/library/publish", json={}).json()["version"]


def _spec():
    return {
        "ingredients": [
            {"ingredient_id": "corn", "min": 0.0, "max": 0.7},
            {"ingredient_id": "sbm", "min": 0.05, "max": 0.5},
            {"ingredient_id": "fish", "min": 0.0, "max": 0.15},
            {"ingredient_id": "lime", "min": 0.0, "max": 0.05},
        ],
        "nutrient_bounds": {"CP": {"min": 20.0}, "ME": {"min": 2.9},
                            "Ca": {"min": 0.8, "max": 1.2}},
        "ratios": [{"id": "cap", "numerator": "Ca", "denominator": "P",
                    "min": 1.2, "max": 2.0}],
    }


def test_warm_and_cold_match_across_price_changes(client):
    base_prices = [0.30, 0.50, 0.90, 0.20]
    prev_lv = _publish(client, base_prices)
    fid, _ = create_formula(client, _spec())
    first = client.post(
        f"/formulas/{fid}/optimize",
        json={"library_version": prev_lv}).json()
    assert first["status"] == "ok"

    perturbations = [
        (0.31, 0.52, 0.88, 0.21),   # small changes: warm should be accepted
        (0.30, 0.49, 0.95, 0.19),
        (0.35, 0.45, 0.80, 0.25),   # bigger but same basis region maybe
        (0.28, 0.55, 1.20, 0.18),
    ]

    for prices in perturbations:
        lv = _publish(client, prices)
        warm = client.post(
            f"/formulas/{fid}/optimize",
            json={"library_version": lv}).json()
        cold = client.post(
            f"/formulas/{fid}/optimize",
            json={"library_version": lv, "force_cold_start": True}).json()
        assert warm["status"] == cold["status"] == "ok"
        # relative difference within 1e-9
        denom = max(abs(cold["cost"]), 1e-12)
        assert abs(warm["cost"] - cold["cost"]) / denom < 1e-9, prices
        for a, b in zip(warm["ingredients"], cold["ingredients"]):
            assert abs(a["quantity"] - b["quantity"]) < 1e-9
        # certificates agree too
        assert abs(warm["certificate"]["relative_gap"]) < 1e-9
        assert abs(cold["certificate"]["relative_gap"]) < 1e-9
        prev_lv = lv


def test_warm_start_skips_phase_one_when_basis_still_feasible(client):
    lv = _publish(client, [0.30, 0.50, 0.90, 0.20])
    fid, _ = create_formula(client, _spec())
    first = client.post(
        f"/formulas/{fid}/optimize",
        json={"library_version": lv}).json()
    assert first["phase1_iterations"] >= 0

    lv2 = _publish(client, [0.305, 0.495, 0.91, 0.198])
    second = client.post(
        f"/formulas/{fid}/optimize",
        json={"library_version": lv2}).json()
    # tiny price moves: previous basis primal feasible -> phase I skipped
    assert second["warm_start_accepted"] is True
    assert second["phase1_iterations"] == 0

    cold = client.post(
        f"/formulas/{fid}/optimize",
        json={"library_version": lv2, "force_cold_start": True}).json()
    assert cold["warm_start_accepted"] is False
    assert abs(second["cost"] - cold["cost"]) < 1e-10


def test_warm_basis_invalidated_by_constraint_change_falls_back(client):
    lv = _publish(client, [0.30, 0.50, 0.90, 0.20])
    fid, _ = create_formula(client, _spec())
    client.post(f"/formulas/{fid}/optimize",
                json={"library_version": lv}).json()

    # tighten CP to 26: old basis becomes infeasible, must run phase I again
    spec = _spec()
    spec["nutrient_bounds"]["CP"] = {"min": 26.0}
    client.put(f"/formulas/{fid}", json={"spec": spec})
    r = client.post(f"/formulas/{fid}/optimize",
                    json={"library_version": lv}).json()
    assert r["status"] == "ok"
    # warm basis structure key includes RHS, so it was not accepted here
    assert r["warm_start_accepted"] is False
    cold = client.post(f"/formulas/{fid}/optimize",
                       json={"library_version": lv,
                             "force_cold_start": True}).json()
    assert abs(r["cost"] - cold["cost"]) < 1e-10
