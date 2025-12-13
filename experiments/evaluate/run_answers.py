from __future__ import annotations

import csv
import json
import sys
import logging
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.progress import track


BASE_DIR = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = BASE_DIR / "experiments" / "benchmark_generation" / "questions.jsonl"
OUTPUT_CSV = BASE_DIR / "experiments" / "evaluate" / "model_answers.csv"
LIMIT = None


sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "models"))

from agentic_rag.config import settings
from agentic_rag.agent import RagResponse, build_agent
from models.base import llm as base_llm
from models.rag.pipeline import run_rag


console = Console()

logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("lancedb").setLevel(logging.WARNING)
logging.getLogger("agno").setLevel(logging.WARNING)
logging.getLogger("agno.knowledge").setLevel(logging.WARNING)


def _load_questions(path: Path, limit: int | None) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with path.open() as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def answer_base(question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a concise, helpful assistant. Answer directly.",
        },
        {"role": "user", "content": question},
    ]
    return base_llm._run_completion(messages)  # type: ignore[attr-defined]


def answer_rag(question: str) -> str:
    response, _ = run_rag(question)
    return response


def answer_agentic(agent, question: str) -> str:
    result = agent.run(question)
    content = getattr(result, "content", result)
    if isinstance(content, RagResponse):
        return content.answer
    return str(content)


def run() -> None:
    records = _load_questions(QUESTIONS_PATH, LIMIT)
    if not records:
        console.print("[red]No questions loaded; nothing to do.")
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question",
        "reference_answer",
        "base_answer",
        "rag_answer",
        "agentic_rag_answer",
    ]

    agent = build_agent()

    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in track(records, description="Running models"):
            question = rec.get("question", "").strip()
            reference = rec.get("answer", "").strip()

            base_answer = answer_base(question)
            rag_answer = answer_rag(question)
            agent_answer = answer_agentic(agent, question)

            console.print("\n[cyan]Question:[/cyan]", question)
            console.print("[magenta]Reference:[/magenta]", reference)
            console.print("[blue]Base:[/blue]", base_answer)
            console.print("[green]RAG:[/green]", rag_answer)
            console.print("[yellow]Agentic RAG:[/yellow]", agent_answer)

            writer.writerow(
                {
                    "question": question,
                    "reference_answer": reference,
                    "base_answer": base_answer,
                    "rag_answer": rag_answer,
                    "agentic_rag_answer": agent_answer,
                }
            )

    console.print(f"[green]Wrote answers for {len(records)} questions to {OUTPUT_CSV}")


if __name__ == "__main__":
    run()
