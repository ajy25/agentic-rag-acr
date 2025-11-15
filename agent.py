"""CLI entrypoint for the Agno-powered agentic RAG system."""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import typer
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from rag.config import SQLITE_DB_PATH, settings
from rag.knowledge import get_knowledge

console = Console()
cli = typer.Typer(add_completion=False, help="Ask questions over the ingested PDFs")


class RagResponse(BaseModel):
    """Structured answer used for deterministic outputs."""

    answer: str = Field(..., description="Final response grounded in the knowledge base")
    citations: List[str] = Field(
        default_factory=list,
        description="List of sources cited in the answer",
    )


def _citation_block(citations: Iterable[str]) -> str:
    items = list(citations)
    if not items:
        return ""
    bullet_list = "\n".join(f"- {item}" for item in items)
    return f"\n[bold]Citations[/bold]:\n{bullet_list}"


@lru_cache(maxsize=1)
def build_agent() -> Agent:
    knowledge = get_knowledge()
    instructions = (
        "You are a precise research assistant. Use the provided knowledge base"
        " to answer questions. Include short citations referencing the file name"
        " and page range from the metadata of each chunk you reference."
    )
    return Agent(
        name="acr-agent",
        instructions=instructions,
        model=OpenAIChat(id=settings.chat_model, api_key=settings.openai_api_key),
        knowledge=knowledge,
        search_knowledge=True,
        markdown=True,
        output_schema=RagResponse,
        db=SqliteDb(db_file=str(SQLITE_DB_PATH)),
        user_id=settings.history_user_id,
        add_history_to_context=True,
        num_history_runs=3,
    )


def preview_context(question: str, limit: int) -> None:
    knowledge = get_knowledge()
    documents = knowledge.search(question, max_results=limit)
    if not documents:
        console.print("[yellow]No matching chunks found in the knowledge base.")
        return
    table = Table(title="Top retrieved chunks", show_lines=True)
    table.add_column("#", justify="right")
    table.add_column("File")
    table.add_column("Pages")
    table.add_column("Snippet", overflow="fold")
    for idx, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        snippet = (doc.content or "").strip().replace("\n", " ")
        table.add_row(
            str(idx),
            metadata.get("file_name", "unknown"),
            metadata.get("page_range", "?"),
            snippet[:400] + ("…" if len(snippet) > 400 else ""),
        )
    console.print(table)


def render_response(result) -> None:
    content = getattr(result, "content", result)
    if isinstance(content, RagResponse):
        console.print(content.answer)
        citations_block = _citation_block(content.citations)
        if citations_block:
            console.print(citations_block)
    else:
        console.print(content)


@cli.command()
def ask(
    question: str = typer.Argument(..., help="Your question for the agent"),
    show_context: bool = typer.Option(
        False,
        "--show-context",
        "-c",
        help="Preview the top retrieved chunks before answering",
    ),
    context_k: int = typer.Option(3, "--context-k", help="How many chunks to preview"),
) -> None:
    """Ask a single question and print the model's response."""
    agent = build_agent()
    if show_context:
        preview_context(question, context_k)
    result = agent.run(question)
    render_response(result)


@cli.command()
def chat(context_k: int = typer.Option(0, help="Preview this many chunks each turn")) -> None:
    """Open an interactive REPL with conversation memory."""
    agent = build_agent()
    console.print("[bold]Interactive chat.[/bold] Type /exit to quit.")
    while True:
        try:
            user_input = typer.prompt("You")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[red]Exiting chat.")
            break
        if not user_input.strip():
            continue
        if user_input.strip().lower() in {"/exit", "exit", "quit", ":q"}:
            console.print("[cyan]Bye!")
            break
        if context_k > 0:
            preview_context(user_input, context_k)
        result = agent.run(user_input)
        render_response(result)


if __name__ == "__main__":
    cli()
