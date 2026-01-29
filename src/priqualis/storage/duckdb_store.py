"""DuckDB storage for Priqualis."""

import logging
from datetime import datetime
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)


class DuckDBStore:
    def __init__(self, db_path: Path | str = "./data/priqualis.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS validation_runs (
                run_id VARCHAR PRIMARY KEY, run_at TIMESTAMP,
                total_records INTEGER, violations INTEGER, warnings INTEGER,
                pass_rate DOUBLE, duration_ms DOUBLE
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY, run_id VARCHAR, case_id VARCHAR,
                rule_id VARCHAR, severity VARCHAR, message VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                audit_id VARCHAR PRIMARY KEY, case_id VARCHAR, rule_id VARCHAR,
                patch_applied BOOLEAN, user_id VARCHAR, applied_at TIMESTAMP, changes_json VARCHAR
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fpa_history (
                id INTEGER PRIMARY KEY, recorded_at TIMESTAMP,
                period_start DATE, period_end DATE,
                total_submitted INTEGER, total_accepted INTEGER, fpa_rate DOUBLE, source VARCHAR
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS rejections (
                id INTEGER PRIMARY KEY, case_id VARCHAR, rejection_date DATE,
                error_code VARCHAR, error_message VARCHAR, batch_id VARCHAR,
                imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def record_validation_run(self, run_id: str, total: int, violations: int, warnings: int, pass_rate: float, duration_ms: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO validation_runs VALUES (?, ?, ?, ?, ?, ?, ?)",
            [run_id, datetime.now(), total, violations, warnings, pass_rate, duration_ms]
        )

    def record_violations(self, run_id: str, violations: list[dict]) -> int:
        for v in violations:
            self.conn.execute(
                "INSERT INTO violations (run_id, case_id, rule_id, severity, message) VALUES (?, ?, ?, ?, ?)",
                [run_id, v.get("case_id"), v.get("rule_id"), v.get("severity", "error"), v.get("message")]
            )
        return len(violations)

    def record_audit(self, audit_id: str, case_id: str, rule_id: str, user: str, changes: str, applied: bool = True) -> None:
        self.conn.execute(
            "INSERT INTO audit_log VALUES (?, ?, ?, ?, ?, ?, ?)",
            [audit_id, case_id, rule_id, applied, user, datetime.now(), changes]
        )

    def record_fpa(self, submitted: int, accepted: int, source: str = "validation") -> None:
        rate = accepted / submitted if submitted > 0 else 0
        self.conn.execute(
            "INSERT INTO fpa_history (recorded_at, period_start, period_end, total_submitted, total_accepted, fpa_rate, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [datetime.now(), datetime.now().date(), datetime.now().date(), submitted, accepted, rate, source]
        )

    def import_rejections(self, rejections: list[dict]) -> int:
        for r in rejections:
            self.conn.execute(
                "INSERT INTO rejections (case_id, rejection_date, error_code, error_message, batch_id) VALUES (?, ?, ?, ?, ?)",
                [r.get("case_id"), r.get("rejection_date"), r.get("error_code"), r.get("error_message"), r.get("batch_id")]
            )
        logger.info("Imported %d rejections", len(rejections))
        return len(rejections)

    def get_fpa_trend(self, days: int = 30) -> list[dict]:
        rows = self.conn.execute(
            "SELECT recorded_at::DATE as day, AVG(fpa_rate) as avg_fpa FROM fpa_history WHERE recorded_at >= CURRENT_DATE - INTERVAL ? DAY GROUP BY day ORDER BY day",
            [days]
        ).fetchall()
        return [{"day": str(r[0]), "fpa": r[1]} for r in rows]

    def get_violation_stats(self) -> dict:
        rows = self.conn.execute("SELECT rule_id, COUNT(*) as cnt, severity FROM violations GROUP BY rule_id, severity ORDER BY cnt DESC").fetchall()
        return {"by_rule": [{"rule_id": r[0], "count": r[1], "severity": r[2]} for r in rows]}

    def get_recent_runs(self, limit: int = 10) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM validation_runs ORDER BY run_at DESC LIMIT ?", [limit]).fetchall()
        return [{"run_id": r[0], "run_at": str(r[1]), "total": r[2], "violations": r[3], "warnings": r[4], "pass_rate": r[5]} for r in rows]

    def query(self, sql: str) -> list[tuple]:
        return self.conn.execute(sql).fetchall()

    def close(self) -> None:
        self.conn.close()


_store: DuckDBStore | None = None

def get_duckdb_store(db_path: str | None = None) -> DuckDBStore:
    global _store
    if _store is None:
        _store = DuckDBStore(db_path or "./data/priqualis.duckdb")
    return _store
