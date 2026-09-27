"""Memoria persistente: hechos sobre ti que la IA recuerda entre sesiones (SQLite + FTS5)."""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

_WORD_RE = re.compile(r"\w{3,}", re.UNICODE)


@dataclass
class Memory:
    id: int
    content: str
    created: float


class MemoryStore:
    def __init__(self, path: str | Path):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL UNIQUE,
                created REAL NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(content, content='memories', content_rowid='id');
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content)
                    VALUES ('delete', old.id, old.content);
            END;
            """
        )
        self.db.commit()

    def add(self, content: str) -> bool:
        """Guarda un recuerdo. Devuelve False si estaba vacío o ya existía."""
        content = " ".join(content.split())
        if not content:
            return False
        cur = self.db.execute(
            "INSERT OR IGNORE INTO memories (content, created) VALUES (?, ?)",
            (content, time.time()),
        )
        self.db.commit()
        return cur.rowcount > 0

    def search(self, query: str, k: int = 8) -> list[Memory]:
        """Recuerdos más relevantes para `query` (ranking BM25)."""
        words = {w.lower() for w in _WORD_RE.findall(query)}
        if not words:
            return []
        fts_query = " OR ".join(f'"{w}"' for w in sorted(words))
        rows = self.db.execute(
            """
            SELECT m.id, m.content, m.created FROM memories_fts
            JOIN memories m ON m.id = memories_fts.rowid
            WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?
            """,
            (fts_query, k),
        ).fetchall()
        return [Memory(*r) for r in rows]

    def recent(self, k: int = 5) -> list[Memory]:
        rows = self.db.execute(
            "SELECT id, content, created FROM memories ORDER BY created DESC, id DESC LIMIT ?",
            (k,),
        ).fetchall()
        return [Memory(*r) for r in rows]

    def all(self) -> list[Memory]:
        rows = self.db.execute("SELECT id, content, created FROM memories ORDER BY id").fetchall()
        return [Memory(*r) for r in rows]

    def delete(self, memory_id: int) -> bool:
        cur = self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.db.commit()
        return cur.rowcount > 0

    def clear(self) -> None:
        self.db.execute("DELETE FROM memories")
        self.db.commit()
