"""SQLite persistence.

Everything lives in one SQLite database file inside the mounted data
directory.  Writes are serialised by a process-wide lock and SQLite runs in
WAL journal mode so background job threads never block HTTP reads.

All historical data is append-only: ingredient library versions, formula
specification versions and optimisation results are never overwritten, which
also guarantees that two concurrent jobs touching the same formula produce two
distinct result rows instead of clobbering each other.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

_lock = threading.RLock()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def new_id() -> str:
    return uuid.uuid4().hex


SCHEMA = """
CREATE TABLE IF NOT EXISTS ingredient_draft (
    ingredient_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    price REAL NOT NULL,
    dry_matter REAL NOT NULL,
    nutrients_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS library_versions (
    version INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    note TEXT
);

CREATE TABLE IF NOT EXISTS ingredient_snapshots (
    version INTEGER NOT NULL,
    ingredient_id TEXT NOT NULL,
    name TEXT NOT NULL,
    price REAL NOT NULL,
    dry_matter REAL NOT NULL,
    nutrients_json TEXT NOT NULL,
    PRIMARY KEY (version, ingredient_id)
);

CREATE TABLE IF NOT EXISTS nutrient_codes (
    code TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS formulas (
    formula_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS formula_versions (
    formula_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    spec_json TEXT NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (formula_id, version)
);

CREATE TABLE IF NOT EXISTS optimizations (
    optimization_id TEXT PRIMARY KEY,
    formula_id TEXT NOT NULL,
    formula_version INTEGER NOT NULL,
    library_version INTEGER NOT NULL,
    trigger TEXT NOT NULL,
    job_id TEXT,
    status TEXT NOT NULL,
    result_json TEXT,
    conflicts_json TEXT,
    warm_from_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_opt_formula ON optimizations(formula_id, created_at);
CREATE INDEX IF NOT EXISTS idx_opt_job ON optimizations(job_id);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    library_version INTEGER NOT NULL,
    scope_json TEXT NOT NULL,
    total INTEGER NOT NULL,
    succeeded INTEGER NOT NULL,
    failed INTEGER NOT NULL,
    cancel_requested INTEGER NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS job_items (
    item_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    formula_id TEXT NOT NULL,
    formula_version INTEGER NOT NULL,
    status TEXT NOT NULL,
    optimization_id TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_items_job ON job_items(job_id, seq);
"""


class Database:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._tls = threading.local()
        with self.write() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(SCHEMA)

    # -- connection management --------------------------------------------
    def _new_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._tls, "conn", None)
        if c is None:
            c = self._new_conn()
            self._tls.conn = c
        return c

    class _Ctx:
        def __init__(self, db: "Database", write: bool):
            self.db = db
            self.write = write

        def __enter__(self) -> sqlite3.Connection:
            if self.write:
                _lock.acquire()
            return self.db.conn

        def __exit__(self, exc_type, exc, tb) -> None:
            c = self.db.conn
            if exc_type is None:
                c.commit()
            else:
                c.rollback()
            if self.write:
                _lock.release()

    def read(self) -> "Database._Ctx":
        return self._Ctx(self, False)

    def write(self) -> "Database._Ctx":
        return self._Ctx(self, True)

    # -- ingredient draft ---------------------------------------------------
    def upsert_draft_ingredient(self, payload: dict[str, Any]) -> None:
        with self.write() as c:
            c.execute(
                """INSERT INTO ingredient_draft(ingredient_id, name, price,
                       dry_matter, nutrients_json, updated_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(ingredient_id) DO UPDATE SET
                       name=excluded.name, price=excluded.price,
                       dry_matter=excluded.dry_matter,
                       nutrients_json=excluded.nutrients_json,
                       updated_at=excluded.updated_at""",
                (payload["ingredient_id"], payload["name"],
                 float(payload["price"]), float(payload["dry_matter"]),
                 json.dumps(payload["nutrients"], sort_keys=True), utcnow()),
            )

    def delete_draft_ingredient(self, ingredient_id: str) -> bool:
        with self.write() as c:
            cur = c.execute(
                "DELETE FROM ingredient_draft WHERE ingredient_id=?",
                (ingredient_id,))
            return cur.rowcount > 0

    def list_draft_ingredients(self) -> list[dict[str, Any]]:
        with self.read() as c:
            rows = c.execute(
                "SELECT * FROM ingredient_draft ORDER BY ingredient_id"
            ).fetchall()
        return [self._ingredient_row(r) for r in rows]

    def get_draft_ingredient(self, ingredient_id: str) -> dict[str, Any] | None:
        with self.read() as c:
            r = c.execute(
                "SELECT * FROM ingredient_draft WHERE ingredient_id=?",
                (ingredient_id,)).fetchone()
        return self._ingredient_row(r) if r else None

    @staticmethod
    def _ingredient_row(r: sqlite3.Row) -> dict[str, Any]:
        return {
            "ingredient_id": r["ingredient_id"],
            "name": r["name"],
            "price": r["price"],
            "dry_matter": r["dry_matter"],
            "nutrients": json.loads(r["nutrients_json"]),
            "updated_at": r["updated_at"],
        }

    # -- nutrient codes ------------------------------------------------------
    def ensure_nutrient_codes(self, codes: Iterable[str]) -> None:
        with self.write() as c:
            for code in codes:
                c.execute(
                    "INSERT OR IGNORE INTO nutrient_codes(code, created_at) "
                    "VALUES (?,?)", (code, utcnow()))

    def list_nutrient_codes(self) -> list[str]:
        with self.read() as c:
            return [r["code"] for r in c.execute(
                "SELECT code FROM nutrient_codes ORDER BY code")]

    # -- library versions ----------------------------------------------------
    def publish_library(self, note: str | None) -> int:
        with self.write() as c:
            cur = c.execute(
                "INSERT INTO library_versions(created_at, note) VALUES (?,?)",
                (utcnow(), note))
            version = int(cur.lastrowid)
            c.execute(
                """INSERT INTO ingredient_snapshots(version, ingredient_id,
                       name, price, dry_matter, nutrients_json)
                   SELECT ?, ingredient_id, name, price, dry_matter,
                          nutrients_json FROM ingredient_draft""",
                (version,))
            for r in c.execute(
                    "SELECT nutrients_json FROM ingredient_draft"):
                for code in json.loads(r["nutrients_json"]):
                    c.execute(
                        "INSERT OR IGNORE INTO nutrient_codes(code, created_at)"
                        " VALUES (?,?)", (code, utcnow()))
        return version

    def latest_library_version(self) -> int | None:
        with self.read() as c:
            r = c.execute(
                "SELECT MAX(version) AS v FROM library_versions").fetchone()
        return None if r is None or r["v"] is None else int(r["v"])

    def list_library_versions(self) -> list[dict[str, Any]]:
        with self.read() as c:
            rows = c.execute(
                "SELECT * FROM library_versions ORDER BY version"
            ).fetchall()
        return [{"version": r["version"], "created_at": r["created_at"],
                 "note": r["note"]} for r in rows]

    def list_snapshot_ingredients(self, version: int) -> list[dict[str, Any]]:
        with self.read() as c:
            rows = c.execute(
                """SELECT * FROM ingredient_snapshots WHERE version=?
                   ORDER BY ingredient_id""", (version,)).fetchall()
        return [{
            "ingredient_id": r["ingredient_id"],
            "name": r["name"],
            "price": r["price"],
            "dry_matter": r["dry_matter"],
            "nutrients": json.loads(r["nutrients_json"]),
        } for r in rows]

    def library_version_exists(self, version: int) -> bool:
        with self.read() as c:
            r = c.execute(
                "SELECT 1 FROM library_versions WHERE version=?",
                (version,)).fetchone()
        return r is not None

    def diff_library_versions(self, old: int, new: int) -> dict[str, Any]:
        a = {r["ingredient_id"]: r for r in self.list_snapshot_ingredients(old)}
        b = {r["ingredient_id"]: r for r in self.list_snapshot_ingredients(new)}
        added = sorted(set(b) - set(a))
        removed = sorted(set(a) - set(b))
        changed = []
        for iid in sorted(set(a) & set(b)):
            if (a[iid]["price"] != b[iid]["price"]
                    or a[iid]["dry_matter"] != b[iid]["dry_matter"]
                    or a[iid]["nutrients"] != b[iid]["nutrients"]):
                changed.append({
                    "ingredient_id": iid,
                    "name": b[iid]["name"],
                    "old_price": a[iid]["price"],
                    "new_price": b[iid]["price"],
                    "old": a[iid],
                    "new": b[iid],
                })
        return {"old_version": old, "new_version": new,
                "added": added, "removed": removed, "changed": changed}

    # -- formulas ------------------------------------------------------------
    def create_formula(self, name: str, spec: dict[str, Any],
                       note: str | None = None) -> tuple[str, int]:
        fid = new_id()
        with self.write() as c:
            c.execute(
                "INSERT INTO formulas(formula_id, name, created_at) "
                "VALUES (?,?,?)", (fid, name, utcnow()))
            c.execute(
                """INSERT INTO formula_versions(formula_id, version, spec_json,
                       note, created_at)
                   VALUES (?,1,?,?,?)""",
                (fid, json.dumps(spec, sort_keys=True), note, utcnow()))
        return fid, 1

    def add_formula_version(self, formula_id: str, spec: dict[str, Any],
                            note: str | None = None) -> int:
        with self.write() as c:
            r = c.execute(
                "SELECT MAX(version) AS v FROM formula_versions "
                "WHERE formula_id=?", (formula_id,)).fetchone()
            version = int(r["v"]) + 1
            c.execute(
                """INSERT INTO formula_versions(formula_id, version, spec_json,
                       note, created_at) VALUES (?,?,?,?,?)""",
                (formula_id, version,
                 json.dumps(spec, sort_keys=True), note, utcnow()))
        return version

    def rename_formula(self, formula_id: str, name: str) -> None:
        with self.write() as c:
            c.execute("UPDATE formulas SET name=? WHERE formula_id=?",
                      (name, formula_id))

    def list_formulas(self) -> list[dict[str, Any]]:
        with self.read() as c:
            rows = c.execute("SELECT * FROM formulas ORDER BY name").fetchall()
        return [{"formula_id": r["formula_id"], "name": r["name"],
                 "created_at": r["created_at"]} for r in rows]

    def get_formula(self, formula_id: str) -> dict[str, Any] | None:
        with self.read() as c:
            r = c.execute("SELECT * FROM formulas WHERE formula_id=?",
                          (formula_id,)).fetchone()
        if not r:
            return None
        return {"formula_id": r["formula_id"], "name": r["name"],
                "created_at": r["created_at"]}

    def latest_formula_version(self, formula_id: str) -> int | None:
        with self.read() as c:
            r = c.execute(
                "SELECT MAX(version) AS v FROM formula_versions "
                "WHERE formula_id=?", (formula_id,)).fetchone()
        return None if r is None or r["v"] is None else int(r["v"])

    def get_formula_spec(self, formula_id: str,
                         version: int | None = None) -> dict[str, Any] | None:
        with self.read() as c:
            if version is None:
                r = c.execute(
                    """SELECT * FROM formula_versions WHERE formula_id=?
                       ORDER BY version DESC LIMIT 1""",
                    (formula_id,)).fetchone()
            else:
                r = c.execute(
                    "SELECT * FROM formula_versions WHERE formula_id=? "
                    "AND version=?", (formula_id, version)).fetchone()
        if not r:
            return None
        return {"formula_id": formula_id, "version": r["version"],
                "note": r["note"], "created_at": r["created_at"],
                "spec": json.loads(r["spec_json"])}

    def list_formula_versions(self, formula_id: str) -> list[dict[str, Any]]:
        with self.read() as c:
            rows = c.execute(
                """SELECT version, note, created_at FROM formula_versions
                   WHERE formula_id=? ORDER BY version""",
                (formula_id,)).fetchall()
        return [{"version": r["version"], "note": r["note"],
                 "created_at": r["created_at"]} for r in rows]

    # -- optimizations -------------------------------------------------------
    def insert_optimization(self, *, formula_id: str, formula_version: int,
                            library_version: int, trigger: str,
                            job_id: str | None, status: str,
                            result: dict[str, Any] | None = None,
                            conflicts: list | None = None,
                            warm_from_id: str | None = None) -> str:
        oid = new_id()
        with self.write() as c:
            c.execute(
                """INSERT INTO optimizations(optimization_id, formula_id,
                       formula_version, library_version, trigger, job_id,
                       status, result_json, conflicts_json, warm_from_id,
                       created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (oid, formula_id, formula_version, library_version, trigger,
                 job_id, status,
                 json.dumps(result, sort_keys=True) if result else None,
                 json.dumps(conflicts, sort_keys=True) if conflicts else None,
                 warm_from_id, utcnow()))
        return oid

    def delete_job_optimizations(self, job_id: str) -> None:
        """Used on cancellation: a cancelled job leaves no partial results."""
        with self.write() as c:
            c.execute("DELETE FROM optimizations WHERE job_id=?", (job_id,))

    def get_optimization(self, optimization_id: str) -> dict[str, Any] | None:
        with self.read() as c:
            r = c.execute(
                "SELECT * FROM optimizations WHERE optimization_id=?",
                (optimization_id,)).fetchone()
        return self._opt_row(r) if r else None

    def list_optimizations(self, formula_id: str,
                           limit: int = 100) -> list[dict[str, Any]]:
        with self.read() as c:
            rows = c.execute(
                """SELECT * FROM optimizations WHERE formula_id=?
                   ORDER BY rowid DESC LIMIT ?""",
                (formula_id, limit)).fetchall()
        return [self._opt_row(r) for r in rows]

    @staticmethod
    def _opt_row(r: sqlite3.Row) -> dict[str, Any]:
        return {
            "optimization_id": r["optimization_id"],
            "formula_id": r["formula_id"],
            "formula_version": r["formula_version"],
            "library_version": r["library_version"],
            "trigger": r["trigger"],
            "job_id": r["job_id"],
            "status": r["status"],
            "result": json.loads(r["result_json"]) if r["result_json"] else None,
            "conflicts": (json.loads(r["conflicts_json"])
                          if r["conflicts_json"] else None),
            "warm_from_id": r["warm_from_id"],
            "created_at": r["created_at"],
        }

    def latest_successful_optimization(self, formula_id: str,
                                       ) -> dict[str, Any] | None:
        with self.read() as c:
            r = c.execute(
                """SELECT * FROM optimizations
                   WHERE formula_id=? AND status='ok'
                   ORDER BY rowid DESC LIMIT 1""",
                (formula_id,)).fetchone()
        return self._opt_row(r) if r else None

    # -- jobs ----------------------------------------------------------------
    def create_job(self, *, library_version: int, scope: dict[str, Any],
                   items: list[tuple[str, int]]) -> str:
        jid = new_id()
        with self.write() as c:
            c.execute(
                """INSERT INTO jobs(job_id, status, library_version, scope_json,
                       total, succeeded, failed, cancel_requested, created_at)
                   VALUES (?,?,?,?,?,0,0,0,?)""",
                (jid, "queued", library_version,
                 json.dumps(scope, sort_keys=True), len(items), utcnow()))
            for seq, (fid, fv) in enumerate(items):
                c.execute(
                    """INSERT INTO job_items(item_id, job_id, seq, formula_id,
                           formula_version, status)
                       VALUES (?,?,?,?,?,'pending')""",
                    (new_id(), jid, seq, fid, fv))
        return jid

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.read() as c:
            r = c.execute("SELECT * FROM jobs WHERE job_id=?",
                          (job_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["scope"] = json.loads(d.pop("scope_json"))
        return d

    def list_job_items(self, job_id: str) -> list[dict[str, Any]]:
        with self.read() as c:
            rows = c.execute(
                "SELECT * FROM job_items WHERE job_id=? ORDER BY seq",
                (job_id,)).fetchall()
        return [dict(r) for r in rows]

    def request_cancel(self, job_id: str) -> bool:
        with self.write() as c:
            cur = c.execute(
                "UPDATE jobs SET cancel_requested=1 WHERE job_id=? "
                "AND status IN ('queued','running')", (job_id,))
            return cur.rowcount > 0

    def is_cancel_requested(self, job_id: str) -> bool:
        with self.read() as c:
            r = c.execute(
                "SELECT cancel_requested FROM jobs WHERE job_id=?",
                (job_id,)).fetchone()
        return bool(r and r["cancel_requested"])

    def mark_job_started(self, job_id: str) -> None:
        with self.write() as c:
            c.execute(
                "UPDATE jobs SET status='running', started_at=? "
                "WHERE job_id=? AND status='queued'", (utcnow(), job_id))

    def mark_item(self, item_id: str, status: str,
                  optimization_id: str | None = None,
                  error: str | None = None) -> None:
        with self.write() as c:
            c.execute(
                """UPDATE job_items SET status=?, optimization_id=?, error=?
                   WHERE item_id=?""",
                (status, optimization_id, error, item_id))

    def bump_job_counts(self, job_id: str, ok: bool) -> None:
        with self.write() as c:
            col = "succeeded" if ok else "failed"
            c.execute(
                f"UPDATE jobs SET {col}={col}+1 WHERE job_id=?", (job_id,))

    def finish_job(self, job_id: str, status: str,
                   error: str | None = None) -> None:
        with self.write() as c:
            c.execute(
                "UPDATE jobs SET status=?, finished_at=?, error=? "
                "WHERE job_id=?", (status, utcnow(), error, job_id))

    def queued_items(self, job_id: str) -> list[sqlite3.Row]:
        with self.read() as c:
            return c.execute(
                "SELECT * FROM job_items WHERE job_id=? AND status='pending' "
                "ORDER BY seq", (job_id,)).fetchall()
