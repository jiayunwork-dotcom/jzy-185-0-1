"""Shared pytest fixtures."""
from __future__ import annotations

import tempfile

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app


@pytest.fixture()
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture()
def client(tmp_dir):
    app = create_app(tmp_dir)
    with TestClient(app) as c:
        yield c
    app.state.feed.scheduler.shutdown(wait=True)


# ---- data factories ---------------------------------------------------------

CORN = {"ingredient_id": "corn", "name": "玉米", "price": 0.30,
        "dry_matter": 0.88,
        "nutrients": {"CP": 8.0, "ME": 3.35, "Ca": 0.02, "P": 0.27,
                      "Lys": 0.25, "Met": 0.18}}
SBM = {"ingredient_id": "sbm", "name": "豆粕", "price": 0.50,
       "dry_matter": 0.89,
       "nutrients": {"CP": 44.0, "ME": 2.60, "Ca": 0.30, "P": 0.65,
                     "Lys": 2.70, "Met": 0.60}}
FISH = {"ingredient_id": "fish", "name": "鱼粉", "price": 0.90,
        "dry_matter": 0.92,
        "nutrients": {"CP": 62.0, "ME": 3.0, "Ca": 4.0, "P": 2.8,
                      "Lys": 4.8, "Met": 1.7}}
LIMESTONE = {"ingredient_id": "lime", "name": "石粉", "price": 0.20,
             "dry_matter": 0.99,
             "nutrients": {"CP": 0.0, "ME": 0.0, "Ca": 38.0, "P": 0.0,
                           "Lys": 0.0, "Met": 0.0}}


def publish_basic_library(client, ingredients=(CORN, SBM), note="v1"):
    for ing in ingredients:
        r = client.put(f"/ingredients/{ing['ingredient_id']}", json=ing)
        assert r.status_code == 200, r.text
    r = client.post("/library/publish", json={"note": note})
    assert r.status_code == 200, r.text
    return r.json()["version"]


def create_formula(client, spec, name="f1"):
    r = client.post("/formulas", json={"name": name, "spec": spec})
    assert r.status_code == 201, r.text
    return r.json()["formula_id"], r.json()["version"]


def wait_for_job(client, job_id, timeout=30.0):
    import time
    deadline = time.time() + timeout
    while True:
        r = client.get(f"/jobs/{job_id}")
        assert r.status_code == 200
        st = r.json()["status"]
        if st in ("completed", "cancelled", "failed"):
            return r.json()
        if time.time() > deadline:
            raise AssertionError(f"job {job_id} stuck in {st}")
        time.sleep(0.02)
