from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Tuple

import typer
from openai import OpenAI
from rich.console import Console
from rich.table import Table

from agentic_rag.config import settings
from agentic_rag.knowledge import get_knowledge

console = Console()
cli = typer.Typer(
    add_completion=False, help="Vanilla RAG over the shared knowledge base"
)


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def _format_context(blocks: Iterable[Tuple[int, str, str]]) -> str:
    lines: List[str] = []
    for idx, header, text in blocks:
        lines.append(f"[{idx}] {header}\n{text}")
    return "\n\n".join(lines)


def _prepare_context(question: str, k: int):
    knowledge = get_knowledge()
    docs = knowledge.search(question, max_results=k)
    context_blocks = []
    for idx, doc in enumerate(docs, start=1):
        meta = getattr(doc, "metadata", None) or getattr(doc, "meta_data", {}) or {}
        header = f"{meta.get('file_name', 'unknown')} ({meta.get('page_range', '?')})"
        context_blocks.append((idx, header, (doc.content or "").strip()))
    return context_blocks


def _preview_table(blocks: Iterable[Tuple[int, str, str]]) -> None:
    table = Table(title="Retrieved context", show_lines=True)
    table.add_column("#", justify="right")
    table.add_column("Source")
    table.add_column("Snippet", overflow="fold")
    for idx, header, text in blocks:
        table.add_row(str(idx), header, text[:300] + ("…" if len(text) > 300 else ""))
    console.print(table)


def run_rag(
    question: str,
    *,
    k: int = 4,
    temperature: float = 0.2,
    max_completion_tokens: int = 400,
) -> Tuple[str, List[Tuple[int, str, str]]]:
    blocks = _prepare_context(question, k)
    context = _format_context(blocks)
    client = get_client()
    system_prompt = (
        "You answer user questions using only the provided context blocks."
        " Do not include citations or source references in your reply."
        " If the answer is unknown from the context, say you do not know."
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Context:\n{context if context else 'No context available.'}\n\n"
        "Answer succinctly without citing sources."
    )
    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )
    content = response.choices[0].message.content or ""
    return content, blocks


@cli.command()
def ask(
    question: str = typer.Argument(..., help="Your question for the model"),
    k: int = typer.Option(4, "--k", help="How many chunks to retrieve"),
    temperature: float = typer.Option(0.2, help="Sampling temperature"),
    max_tokens: int = typer.Option(400, help="Max tokens for the answer"),
    show_context: bool = typer.Option(
        False, "--show-context", "-c", help="Print retrieved context"
    ),
) -> None:
    answer, blocks = run_rag(
        question,
        k=k,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if show_context:
        _preview_table(blocks)
    console.print(answer)


@cli.command()
def preview(
    question: str, k: int = typer.Option(4, "--k", help="How many chunks to retrieve")
) -> None:
    blocks = _prepare_context(question, k)
    if not blocks:
        console.print("[yellow]No matching context found.")
        raise typer.Exit(code=0)
    _preview_table(blocks)


if __name__ == "__main__":
    cli()
