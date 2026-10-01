"""FastAPI application factory and routes (HTTP only)."""
from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..errors import (FeedOptError, InfeasibleError, NotFoundError,
                      ValidationError)
from ..services.formulas import FormulaService
from ..services.ingredients import IngredientService
from ..services.reprice import RepriceService
from ..services.runner import OptimizationRunner
from ..services.scheduler import JobScheduler
from ..storage import Database
from . import schemas


class AppState:
    def __init__(self, data_dir: str):
        os.makedirs(data_dir, exist_ok=True)
        self.db = Database(os.path.join(data_dir, "feedopt.db"))
        self.ingredients = IngredientService(self.db)
        self.formulas = FormulaService(self.db)
        self.runner = OptimizationRunner(self.db)
        self.scheduler = JobScheduler(
            self.db, lambda: OptimizationRunner(self.db),
            max_workers=int(os.environ.get("FEEDOPT_JOB_WORKERS", "2")))
        self.reprice = RepriceService(self.db, self.scheduler)


def create_app(data_dir: str | None = None) -> FastAPI:
    data_dir = data_dir or os.environ.get(
        "FEEDOPT_DATA_DIR", os.path.join(os.getcwd(), "data"))
    app = FastAPI(title="Feed Formula Optimisation Service", version="1.0.0")
    app.state.feed = AppState(data_dir)

    feed = app.state.feed

    # -- error handlers -----------------------------------------------------
    @app.exception_handler(ValidationError)
    async def _validation(request, exc: ValidationError):
        return JSONResponse(status_code=422, content={
            "error": "validation_error",
            "message": exc.message,
            "fields": exc.fields,
        })

    @app.exception_handler(NotFoundError)
    async def _not_found(request, exc: NotFoundError):
        return JSONResponse(status_code=404, content={
            "error": "not_found",
            "resource": exc.resource,
            "identifier": exc.identifier,
        })

    @app.exception_handler(InfeasibleError)
    async def _infeasible(request, exc: InfeasibleError):
        return JSONResponse(status_code=409, content={
            "error": "infeasible",
            "message": "配方无可行解",
            "conflicts": exc.conflicts,
        })

    @app.exception_handler(FeedOptError)
    async def _generic(request, exc: FeedOptError):
        return JSONResponse(status_code=400, content={
            "error": type(exc).__name__, "message": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def _request_validation(request, exc: RequestValidationError):
        # Map pydantic's loc tuples to dotted field paths so every 422 from
        # the API carries the same {fields: {field: reason}} envelope.
        fields: dict[str, str] = {}
        for err in exc.errors():
            loc = [p for p in err.get("loc", ()) if p != "body"]
            path = ""
            for p in loc:
                if isinstance(p, int):
                    path = f"{path}[{p}]"
                else:
                    path = f"{path}.{p}" if path else str(p)
            fields.setdefault(path or "body",
                              f"{err.get('type')}: {err.get('msg')}")
        return JSONResponse(status_code=422, content={
            "error": "validation_error",
            "message": "request validation failed",
            "fields": fields,
        })

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # -- ingredients ---------------------------------------------------------
    @app.get("/ingredients")
    async def list_draft():
        return {"ingredients": feed.ingredients.list_draft()}

    @app.put("/ingredients/{ingredient_id}")
    async def upsert_ingredient(ingredient_id: str,
                                body: schemas.IngredientIn):
        data = body.model_dump()
        data["ingredient_id"] = ingredient_id
        return feed.ingredients.upsert(data)

    @app.post("/ingredients/bulk")
    async def bulk_ingredients(body: schemas.BulkIngredientsIn):
        return {"ingredients": feed.ingredients.bulk_upsert(
            [d for d in body.model_dump()["ingredients"]])}

    @app.get("/ingredients/{ingredient_id}")
    async def get_ingredient(ingredient_id: str):
        return feed.ingredients.get_draft(ingredient_id)

    @app.delete("/ingredients/{ingredient_id}", status_code=204)
    async def delete_ingredient(ingredient_id: str):
        feed.ingredients.delete_draft(ingredient_id)
        return JSONResponse(status_code=204, content=None)

    @app.post("/library/publish")
    async def publish_library(body: schemas.PublishIn):
        return feed.ingredients.publish(body.note)

    @app.get("/library/versions")
    async def list_versions():
        return {"versions": feed.ingredients.list_versions()}

    @app.get("/library/versions/{version}")
    async def get_version(version: int):
        return feed.ingredients.get_version(version)

    @app.get("/library/versions/{old}/diff/{new}")
    async def diff_versions(old: int, new: int):
        return feed.ingredients.diff_versions(old, new)

    # -- formulas ------------------------------------------------------------
    @app.post("/formulas", status_code=201)
    async def create_formula(body: schemas.FormulaCreateIn):
        spec = _spec_to_dict(body.spec)
        return feed.formulas.create(body.name, spec, body.note)

    @app.get("/formulas")
    async def list_formulas():
        return {"formulas": feed.formulas.list()}

    @app.get("/formulas/{formula_id}")
    async def get_formula(formula_id: str):
        return feed.formulas.get(formula_id)

    @app.put("/formulas/{formula_id}", status_code=201)
    async def update_formula(formula_id: str, body: schemas.FormulaUpdateIn):
        return feed.formulas.update_spec(
            formula_id, _spec_to_dict(body.spec), body.note, body.name)

    @app.get("/formulas/{formula_id}/versions")
    async def formula_versions(formula_id: str):
        return {"versions": feed.formulas.list_versions(formula_id)}

    @app.get("/formulas/{formula_id}/versions/{version}")
    async def formula_spec(formula_id: str, version: int):
        return feed.formulas.get_spec(formula_id, version)

    # -- optimisation --------------------------------------------------------
    @app.post("/formulas/{formula_id}/optimize")
    async def optimize(formula_id: str, body: schemas.OptimizeIn):
        return feed.runner.run_optimization(
            formula_id=formula_id,
            formula_version=body.formula_version,
            library_version=body.library_version,
            trigger="manual",
            force_cold_start=body.force_cold_start)

    @app.get("/formulas/{formula_id}/results")
    async def results(formula_id: str, limit: int = Query(100, ge=1, le=500)):
        return {"results": feed.runner.history(formula_id, limit)}

    @app.get("/optimizations/{optimization_id}")
    async def get_optimization(optimization_id: str):
        return feed.runner.get(optimization_id)

    @app.get("/formulas/{formula_id}/compare/{a}/{b}")
    async def compare(formula_id: str, a: str, b: str):
        return feed.runner.compare(formula_id, a, b)

    # -- jobs ----------------------------------------------------------------
    @app.post("/jobs/reprice", status_code=202)
    async def submit_reprice(body: schemas.RepriceIn):
        out = feed.reprice.submit_reprice(
            library_version=body.library_version,
            ingredient_ids=body.ingredient_ids,
            formula_ids=body.formula_ids)
        return out

    @app.get("/jobs/{job_id}")
    async def job_status(job_id: str):
        return feed.scheduler.get_status(job_id)

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        ok = feed.scheduler.cancel(job_id)
        return {"cancelled": ok}

    return app


def _spec_to_dict(spec: schemas.FormulaSpecIn) -> dict[str, Any]:
    return spec.model_dump()


app = create_app()
