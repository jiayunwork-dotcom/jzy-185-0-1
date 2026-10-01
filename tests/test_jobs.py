"""Batch re-optimisation jobs: scope, progress, version locking,
cancellation atomicity, and concurrent no-clobber behaviour."""
from __future__ import annotations

import pytest

from .conftest import (CORN, SBM, create_formula, publish_basic_library,
                       wait_for_job)


def _spec(cp=18.0):
    return {
        "ingredients": [
            {"ingredient_id": "corn", "min": 0.0, "max": 1.0},
            {"ingredient_id": "sbm", "min": 0.0, "max": 1.0},
        ],
        "nutrient_bounds": {"CP": {"min": cp}},
        "ratios": [],
    }


def _republish(client, corn_p=None, sbm_p=None):
    if corn_p is not None:
        y = dict(CORN); y["price"] = corn_p
        client.put("/ingredients/corn", json=y)
    if sbm_p is not None:
        y = dict(SBM); y["price"] = sbm_p
        client.put("/ingredients/sbm", json=y)
    return client.post("/library/publish", json={}).json()["version"]


def test_batch_reprice_covers_affected_formulas(client):
    publish_basic_library(client, (CORN, SBM))
    fids = []
    for _ in range(3):
        fid, _ = create_formula(client, _spec())
        client.post(f"/formulas/{fid}/optimize", json={})
        fids.append(fid)

    v2 = _republish(client, corn_p=0.33)
    r = client.post("/jobs/reprice", json={"library_version": v2})
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["target_count"] == 3
    assert job["library_version"] == v2

    done = wait_for_job(client, job["job_id"])
    assert done["status"] == "completed"
    assert done["succeeded"] == 3
    assert done["progress"] == {"done": 3, "total": 3}
    assert {i["formula_id"] for i in done["items"]} == set(fids)
    # results exist and reference the job-locked library version
    for fid in fids:
        row = client.get(f"/formulas/{fid}/results").json()["results"][0]
        assert row["library_version"] == v2
        assert row["job_id"] == job["job_id"]
        assert row["status"] == "ok"


def test_version_lock_during_running_job(client):
    publish_basic_library(client, (CORN, SBM))
    fid, _ = create_formula(client, _spec())
    client.post(f"/formulas/{fid}/optimize", json={})

    v2 = _republish(client, corn_p=0.34)
    # slow down the scheduler so we can publish v3 while the job runs
    client.app.state.feed.scheduler.item_delay = 0.15
    job_id = client.post(
        "/jobs/reprice", json={"library_version": v2}).json()["job_id"]

    import time
    time.sleep(0.05)
    _republish(client, corn_p=0.40)
    done = wait_for_job(client, job_id)
    assert done["status"] == "completed"
    assert done["library_version"] == v2

    row = client.get(f"/formulas/{fid}/results").json()["results"][0]
    assert row["library_version"] == v2  # never mixed with v3


def test_cancel_leaves_no_partial_results(client):
    publish_basic_library(client, (CORN, SBM))
    fids = []
    for _ in range(6):
        fid, _ = create_formula(client, _spec())
        fids.append(fid)
    v2 = _republish(client, corn_p=0.32)

    client.app.state.feed.scheduler.item_delay = 0.1
    job_id = client.post(
        "/jobs/reprice", json={"library_version": v2}).json()["job_id"]
    import time
    time.sleep(0.22)  # ~2 items done
    r = client.post(f"/jobs/{job_id}/cancel")
    assert r.json()["cancelled"] is True
    done = wait_for_job(client, job_id)
    assert done["status"] == "cancelled"

    # none of the formulas keeps an optimisation row from this job
    for fid in fids:
        rows = client.get(f"/formulas/{fid}/results").json()["results"]
        assert all(row["job_id"] != job_id for row in rows)
    assert all(i["status"] == "cancelled" for i in done["items"])


def test_concurrent_jobs_same_formula_do_not_clobber(client):
    publish_basic_library(client, (CORN, SBM))
    fid, _ = create_formula(client, _spec())

    v2 = _republish(client, corn_p=0.32)
    client.app.state.feed.scheduler.item_delay = 0.05
    j1 = client.post("/jobs/reprice",
                     json={"library_version": v2}).json()["job_id"]
    # second manual optimisation of the same formula while job runs
    manual = client.post(f"/formulas/{fid}/optimize",
                         json={"library_version": v2})
    assert manual.status_code == 200
    wait_for_job(client, j1)

    rows = client.get(f"/formulas/{fid}/results").json()["results"]
    job_rows = [r for r in rows if r["job_id"] == j1]
    manual_rows = [r for r in rows if r["trigger"] == "manual"]
    assert len(job_rows) == 1
    assert len(manual_rows) >= 1
    # both rows are complete ok results
    assert all(r["status"] == "ok" for r in job_rows + manual_rows)
    # distinct ids -> no overwrite
    ids = {r["optimization_id"] for r in rows}
    assert len(ids) == len(rows)


def test_compare_two_results(client):
    publish_basic_library(client, (CORN, SBM))
    fid, _ = create_formula(client, _spec())
    r1 = client.post(f"/formulas/{fid}/optimize", json={}).json()

    _republish(client, corn_p=0.55)
    r2 = client.post(f"/formulas/{fid}/optimize", json={}).json()
    assert r2["cost"] > r1["cost"]

    cmp_ = client.get(
        f"/formulas/{fid}/compare/{r1['optimization_id']}/"
        f"{r2['optimization_id']}").json()
    assert cmp_["cost_delta"] == pytest.approx(r2["cost"] - r1["cost"], abs=1e-12)
    delta = {d["ingredient_id"]: d["delta"] for d in cmp_["ingredient_deltas"]}
    assert delta["corn"] != 0
    # versions are attached to each side
    assert cmp_["a"]["library_version"] != cmp_["b"]["library_version"]


def test_job_progress_and_unknown_job_404(client):
    publish_basic_library(client, (CORN, SBM))
    fid, _ = create_formula(client, _spec())
    v2 = _republish(client, sbm_p=0.55)
    job_id = client.post(
        "/jobs/reprice", json={"library_version": v2}).json()["job_id"]
    done = wait_for_job(client, job_id)
    assert done["progress"]["done"] == done["progress"]["total"] == 1
    assert client.get("/jobs/does-not-exist").status_code == 404
