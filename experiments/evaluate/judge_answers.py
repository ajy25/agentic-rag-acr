from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Tuple

from openai import OpenAI
from rich.console import Console
from rich.progress import track


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_CSV = BASE_DIR / "experiments" / "evaluate" / "model_answers.csv"
OUTPUT_CSV = BASE_DIR / "experiments" / "evaluate" / "model_judgments.csv"
JUDGE_MODEL = None
LIMIT = None
INCLUDE_QUESTION = False


sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "models"))

from agentic_rag.config import settings


console = Console()


def _load_rows(path: Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            rows.append(row)
    return rows


def _parse_score(text: str) -> float:
    try:
        obj = json.loads(text)
        score = obj.get("score")
        return float(score)
    except Exception:
        pass

    match = re.search(r"\b(0\.5|0|1)\b", text)
    if match:
        return float(match.group(1))
    return 0.0


def _parse_reasoning(text: str) -> str:
    try:
        obj = json.loads(text)
        reason = obj.get("reasoning")
        if isinstance(reason, str):
            return reason
    except Exception:
        pass
    return text.strip()


def _score_with_llm(
    client: OpenAI,
    judge_model: str,
    question: str,
    reference: str,
    candidate: str,
    temperature: float = 0.0,
) -> Tuple[float, str]:
    system = (
        "You are a strict medical QA judge. Given a question, reference answer,"
        " and a candidate answer, return a JSON object with `score` in {0, 0.5, 1}"
        " and a short `reasoning`."
        " 1 = fully correct, 0.5 = partially correct or incomplete, 0 = incorrect."
        " Be concise and avoid extra commentary."
    )
    user = (
        f"Question: {question}\n"
        f"Reference answer: {reference}\n"
        f"Candidate answer: {candidate}\n"
        'Respond as JSON: {"score": number, "reasoning": string}'
    )
    resp = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_completion_tokens=200,
    )
    content = resp.choices[0].message.content or ""
    return _parse_score(content), _parse_reasoning(content)


def run() -> None:
    rows = _load_rows(INPUT_CSV, LIMIT)
    if not rows:
        console.print("[red]No rows loaded; nothing to score.")
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    judge_model = JUDGE_MODEL or settings.chat_model
    client = OpenAI(api_key=settings.openai_api_key)
    models = ["base", "rag", "agentic_rag"]
    fieldnames = []
    if INCLUDE_QUESTION:
        fieldnames.append("question")
    for name in models:
        fieldnames.extend([f"{name}_score", f"{name}_reasoning"])

    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in track(rows, description="Scoring answers"):
            question = row.get("question", "")
            reference = row.get("reference_answer", "")
            candidates: Dict[str, str] = {
                "base": row.get("base_answer", ""),
                "rag": row.get("rag_answer", ""),
                "agentic_rag": row.get("agentic_rag_answer", ""),
            }

            out_row: Dict[str, str | float] = {}
            if INCLUDE_QUESTION:
                out_row["question"] = question

            for name, answer in candidates.items():
                score, reasoning = _score_with_llm(
                    client, judge_model, question, reference, answer
                )
                out_row[f"{name}_score"] = score
                out_row[f"{name}_reasoning"] = reasoning

            writer.writerow(out_row)

    console.print(f"[green]Wrote judgments for {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    run()
