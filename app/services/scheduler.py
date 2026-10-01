"""Background batch re-optimisation scheduler.

Design choices (see docs/DESIGN.md for the full rationale):

* Jobs run on an in-process :class:`~concurrent.futures.ThreadPoolExecutor`.
  SQLite + the global write lock make concurrent workers safe; workers can be
  capped to 1 for deterministic tests.
* **Version locking.** Both the ingredient-library version and every formula's
  specification version are resolved *when the job is submitted* and stored on
  the job/items. A library published while the job is running has no effect on
  it — every solve reads the same snapshot rows.
* **Cancellation is cooperative and checked between items.** On cancel,
  optimisation rows already produced by the job are deleted in one transaction,
  so a cancelled job never leaves a partial batch behind.
* **No result clobbering.** Results are append-only rows keyed by their own
  optimisation id; two jobs running the same formula concurrently each insert
  their own row.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from ..errors import NotFoundError
from ..storage import Database


class JobCancelled(Exception):
    pass


class JobScheduler:
    def __init__(self, db: Database, runner_factory: Callable[[], Any],
                 max_workers: int = 2):
        self.db = db
        # Factory so each thread builds its own runner (they share the db).
        self._runner_factory = runner_factory
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="feedopt-job")
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        # Test hooks: event waited on inside a job, and a barrier fired
        # before the first item is solved.
        self.item_delay: float = 0.0

    # -- submission -----------------------------------------------------------
    def submit(self, *, library_version: int, scope: dict[str, Any],
               formula_targets: list[tuple[str, int]],
               trigger: str = "reprice") -> str:
        if not formula_targets:
            from ..errors import ValidationError
            raise ValidationError("no formulas to optimise",
                                  {"formula_ids": "受影响配方为空"})
        jid = self.db.create_job(
            library_version=library_version, scope=scope,
            items=formula_targets)
        future = self._executor.submit(self._run, jid)
        with self._lock:
            self._futures[jid] = future
        return jid

    # -- status ---------------------------------------------------------------
    def get_status(self, job_id: str) -> dict[str, Any]:
        job = self.db.get_job(job_id)
        if job is None:
            raise NotFoundError("job", job_id)
        items = self.db.list_job_items(job_id)
        done = sum(1 for i in items if i["status"] in
                   ("ok", "failed", "skipped", "cancelled"))
        return {
            **{k: v for k, v in job.items()},
            "progress": {"done": done, "total": len(items)},
            "items": [
                {"seq": i["seq"], "formula_id": i["formula_id"],
                 "formula_version": i["formula_version"],
                 "status": i["status"],
                 "optimization_id": i["optimization_id"],
                 "error": i["error"]}
                for i in items
            ],
        }

    def cancel(self, job_id: str) -> bool:
        job = self.db.get_job(job_id)
        if job is None:
            raise NotFoundError("job", job_id)
        ok = self.db.request_cancel(job_id)
        return bool(ok)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

    # -- worker ---------------------------------------------------------------
    def _run(self, job_id: str) -> None:
        runner = self._runner_factory()
        self.db.mark_job_started(job_id)
        error: str | None = None
        final_status = "completed"
        try:
            items = self.db.list_job_items(job_id)
            for item in items:
                if item["status"] != "pending":
                    continue
                if self.db.is_cancel_requested(job_id):
                    raise JobCancelled()
                if self.item_delay:
                    import time
                    time.sleep(self.item_delay)
                try:
                    out = runner.run_optimization(
                        formula_id=item["formula_id"],
                        formula_version=item["formula_version"],
                        library_version=self.db.get_job(job_id)["library_version"],
                        trigger="job",
                        job_id=job_id,
                    )
                    self.db.mark_item(
                        item["item_id"], "ok",
                        optimization_id=out.get("optimization_id"))
                    self.db.bump_job_counts(job_id, ok=True)
                except JobCancelled:
                    raise
                except Exception as e:
                    # Item-level isolation. Infeasible runs persist an
                    # "infeasible" optimisation row (kept as history); other
                    # errors record only their message.
                    from ..errors import InfeasibleError
                    opt_id = getattr(e, "optimization_id", None)
                    if isinstance(e, InfeasibleError):
                        self.db.mark_item(
                            item["item_id"], "infeasible",
                            optimization_id=opt_id,
                            error=f"{len(e.conflicts)} conflicting constraints")
                    else:
                        self.db.mark_item(item["item_id"], "failed", error=str(e))
                    self.db.bump_job_counts(job_id, ok=False)

            if self.db.is_cancel_requested(job_id):
                raise JobCancelled()
        except JobCancelled:
            # Roll back every result this job produced: no half batch.
            self.db.delete_job_optimizations(job_id)
            for item in self.db.list_job_items(job_id):
                # clear the dangling optimization reference as the rows are gone
                self.db.mark_item(item["item_id"], "cancelled",
                                  optimization_id=None)
            final_status = "cancelled"
        except Exception as e:  # pragma: no cover - defensive
            error = str(e)
            final_status = "failed"
        self.db.finish_job(job_id, final_status, error=error)
