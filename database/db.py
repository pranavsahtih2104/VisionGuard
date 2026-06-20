"""
database/db.py
SQLite-backed storage for:
  - Registered persons + their 128-D embeddings
  - Recognition logs
  - Unknown face alerts
"""

import sqlite3
import json
import os
from datetime import datetime


DB_PATH = "database/visionguard.db"


class Database:
    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS persons (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT    NOT NULL,
                embedding  TEXT    NOT NULL,
                created_at TEXT    DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS recognition_logs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id  INTEGER REFERENCES persons(id),
                distance   REAL,
                timestamp  TEXT    DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path  TEXT,
                embedding   TEXT,
                resolved    INTEGER DEFAULT 0,
                timestamp   TEXT    DEFAULT (datetime('now'))
            );
        """)
        self.conn.commit()

    # ── Persons ───────────────────────────────────────────────────────────────

    def register_person(self, name: str, embedding: list) -> int:
        cur = self.conn.execute(
            "INSERT INTO persons (name, embedding) VALUES (?, ?)",
            (name, json.dumps(embedding))
        )
        self.conn.commit()
        return cur.lastrowid

    def get_person_name(self, person_id: int) -> str:
        row = self.conn.execute(
            "SELECT name FROM persons WHERE id=?", (person_id,)
        ).fetchone()
        return row['name'] if row else "Unknown"

    def get_all_embeddings(self) -> list:
        rows = self.conn.execute("SELECT id, embedding FROM persons").fetchall()
        return [{'id': r['id'], 'embedding': json.loads(r['embedding'])} for r in rows]

    # ── Logs ──────────────────────────────────────────────────────────────────

    def log_recognition(self, person_id: int, distance: float):
        self.conn.execute(
            "INSERT INTO recognition_logs (person_id, distance) VALUES (?, ?)",
            (person_id, distance)
        )
        self.conn.commit()

    def log_alert(self, image_path: str, embedding: list):
        self.conn.execute(
            "INSERT INTO alerts (image_path, embedding) VALUES (?, ?)",
            (image_path, json.dumps(embedding))
        )
        self.conn.commit()

    def get_recent_logs(self, n=20) -> list:
        rows = self.conn.execute("""
            SELECT rl.timestamp, p.name, rl.distance
            FROM recognition_logs rl
            JOIN persons p ON p.id = rl.person_id
            ORDER BY rl.id DESC LIMIT ?
        """, (n,)).fetchall()
        return [dict(r) for r in rows]

    def get_alerts(self, n=20) -> list:
        rows = self.conn.execute(
            "SELECT id, image_path, timestamp, resolved FROM alerts ORDER BY id DESC LIMIT ?",
            (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        total_persons = self.conn.execute("SELECT COUNT(*) as c FROM persons").fetchone()['c']
        total_recog   = self.conn.execute("SELECT COUNT(*) as c FROM recognition_logs").fetchone()['c']
        total_alerts  = self.conn.execute("SELECT COUNT(*) as c FROM alerts").fetchone()['c']
        today_recog   = self.conn.execute(
            "SELECT COUNT(*) as c FROM recognition_logs WHERE date(timestamp)=date('now')"
        ).fetchone()['c']
        return {
            'total_persons':    total_persons,
            'total_recognitions': total_recog,
            'total_alerts':     total_alerts,
            'today_recognitions': today_recog,
        }
