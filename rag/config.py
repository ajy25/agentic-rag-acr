"""Configuration helpers for the agentic RAG system."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "lancedb"
SQLITE_DB_PATH = DATA_DIR / "agent_history.db"
KNOWLEDGE_DB_PATH = DATA_DIR / "knowledge.db"
MANIFEST_PATH = DATA_DIR / "ingest_manifest.json"
SOURCES_PATH = BASE_DIR / "sources_pdf"

for path in (DATA_DIR, VECTOR_DB_DIR, SOURCES_PATH):
    path.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env", override=False)


@dataclass
class Settings:
    openai_api_key: str
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    vector_table_name: str = "acr_sources"
    max_search_results: int = 5
    history_user_id: str = "default-user"
    history_session_id: str = "default-session"

    @classmethod
    def load(cls) -> "Settings":
        api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Please add it to your .env file."
            )
        return cls(
            openai_api_key=api_key,
            chat_model=os.getenv("OPENAI_MODEL", cls.chat_model),
            embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", cls.embedding_model
            ),
            vector_table_name=os.getenv("VECTOR_TABLE", cls.vector_table_name),
            max_search_results=int(os.getenv("MAX_SEARCH_RESULTS", 5)),
            history_user_id=os.getenv("AGENT_HISTORY_USER", cls.history_user_id),
            history_session_id=os.getenv(
                "AGENT_HISTORY_SESSION", cls.history_session_id
            ),
        )


settings = Settings.load()

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "VECTOR_DB_DIR",
    "SQLITE_DB_PATH",
    "KNOWLEDGE_DB_PATH",
    "MANIFEST_PATH",
    "SOURCES_PATH",
    "settings",
]
