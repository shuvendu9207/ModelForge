"""
Experiment Tracker
Stores experiment details in SQLite for comparison and reproducibility.
"""

import sqlite3
import json
import time
import os
from datetime import datetime


DB_PATH = "experiments/experiments.db"


class ExperimentTracker:

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT,
                    dataset       TEXT,
                    model         TEXT,
                    params        TEXT,
                    metrics       TEXT,
                    training_time REAL
                )
            """)
            conn.commit()

    def log(self, dataset: str, model_type: str, params: dict,
            metrics: dict, training_time: float = 0.0) -> int:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO experiments
                   (timestamp, dataset, model, params, metrics, training_time)
                   VALUES (?,?,?,?,?,?)""",
                (ts, dataset, model_type,
                 json.dumps(params), json.dumps(metrics), training_time)
            )
            conn.commit()
            exp_id = cur.lastrowid
        print(f"[Tracker] Experiment #{exp_id} logged at {ts}")
        return exp_id

    def load_all(self) -> list:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, timestamp, dataset, model, params, metrics, training_time "
                "FROM experiments ORDER BY id DESC"
            ).fetchall()
        results = []
        for row in rows:
            results.append({
                "id":            row[0],
                "timestamp":     row[1],
                "dataset":       row[2],
                "model":         row[3],
                "params":        json.loads(row[4]),
                "metrics":       json.loads(row[5]),
                "training_time": row[6],
            })
        return results

    def load(self, exp_id: int) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id=?", (exp_id,)
            ).fetchone()
        if not row:
            raise ValueError(f"Experiment #{exp_id} not found.")
        return {
            "id":            row[0],
            "timestamp":     row[1],
            "dataset":       row[2],
            "model":         row[3],
            "params":        json.loads(row[4]),
            "metrics":       json.loads(row[5]),
            "training_time": row[6],
        }
