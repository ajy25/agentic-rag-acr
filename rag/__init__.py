"""Utilities for the agentic RAG system."""
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)

__all__ = ["BASE_DIR"]
