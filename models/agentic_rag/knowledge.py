from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict

from agno.db.sqlite import SqliteDb
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.vectordb.lancedb import LanceDb, SearchType

from .config import BASE_DIR, KNOWLEDGE_DB_PATH, VECTOR_DB_DIR, settings


def _ensure_vector_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_knowledge() -> Knowledge:
    _ensure_vector_dir(VECTOR_DB_DIR)
    vector_db = LanceDb(
        uri=str(VECTOR_DB_DIR),
        table_name=settings.vector_table_name,
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(id=settings.embedding_model),
    )
    knowledge = Knowledge(
        name="acr-source-pdfs",
        description="Knowledge base sourced from PDFs in sources_pdf/",
        vector_db=vector_db,
        contents_db=SqliteDb(db_file=str(KNOWLEDGE_DB_PATH)),
        max_results=settings.max_search_results,
    )
    return knowledge


def chunk_metadata(file_path: Path, chunk_id: int, page_range: str) -> Dict[str, str]:
    try:
        relative = file_path.relative_to(BASE_DIR)
    except ValueError:
        relative = file_path
    return {
        "source": "pdf",
        "file_name": file_path.name,
        "relative_path": str(relative),
        "chunk": str(chunk_id),
        "page_range": page_range,
    }
