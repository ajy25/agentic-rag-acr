from __future__ import annotations

from functools import lru_cache
from typing import List, MutableSequence

import typer
from openai import OpenAI
from rich.console import Console

from agentic_rag.config import settings

console = Console()
cli = typer.Typer(
    add_completion=False, help="Direct access to the chat model without RAG"
)


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def _run_completion(
    messages: MutableSequence[dict],
    *,
    temperature: float = 0.3,
    max_completion_tokens: int = 400,
) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=settings.chat_model,
        messages=list(messages),
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )
    return resp.choices[0].message.content or ""


@cli.command()
def ask(
    prompt: str = typer.Argument(..., help="User prompt"),
    system: str = typer.Option(
        "You are a concise, helpful assistant.",
        "--system",
        help="Optional system prompt",
    ),
    temperature: float = typer.Option(0.3, help="Sampling temperature"),
    max_tokens: int = typer.Option(400, help="Max tokens in the response"),
) -> None:
    messages: List[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    answer = _run_completion(
        messages, temperature=temperature, max_completion_tokens=max_tokens
    )
    console.print(answer)


@cli.command()
def chat(
    system: str = typer.Option(
        "You are a concise, helpful assistant.",
        "--system",
        help="Optional system prompt",
    ),
    temperature: float = typer.Option(0.3, help="Sampling temperature"),
    max_tokens: int = typer.Option(400, help="Max tokens per turn"),
) -> None:
    console.print("[bold]Chat session. Type /exit to leave.[/bold]")
    messages: List[dict] = [{"role": "system", "content": system}]
    while True:
        try:
            user_input = typer.prompt("You")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[cyan]Bye!")
            break
        if user_input.strip().lower() in {"/exit", "exit", "quit", ":q"}:
            console.print("[cyan]Bye!")
            break
        if not user_input.strip():
            continue
        messages.append({"role": "user", "content": user_input})
        reply = _run_completion(
            messages, temperature=temperature, max_completion_tokens=max_tokens
        )
        messages.append({"role": "assistant", "content": reply})
        console.print(reply)


if __name__ == "__main__":
    cli()
