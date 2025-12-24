from __future__ import annotations
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal
import pandas as pd
import typer
from dotenv import load_dotenv
from rich.console import Console
from openai import OpenAI



BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env", override=False)

DEFAULT_CSV = (
    BASE_DIR / "sources" / "sources_table_format" / "acr-ir-guidelines-table.csv"
)
DEFAULT_OUTPUT = BASE_DIR / "experiments" / "benchmark_generation" / "questions.jsonl"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")

console = Console()
app = typer.Typer(
    add_completion=False,
    help="Generate benchmark questions from the ACR IR guideline table",
)

QuestionType = Literal[
    "single_appropriateness",
    "list_appropriate",
    "list_inappropriate",
    "compare_procedures",
    "radiation_aware",
    "clinical_scenario",
]

DIFFICULTY_MAP: Dict[str, Literal["easy", "medium", "hard"]] = {
    "single_appropriateness": "easy",
    "list_appropriate": "medium",
    "list_inappropriate": "medium",
    "compare_procedures": "medium",
    "radiation_aware": "hard",
    "clinical_scenario": "hard",
}

CATEGORY_MAP: Dict[str, str] = {
    "single_appropriateness": "basic_retrieval",
    "list_appropriate": "aggregation",
    "list_inappropriate": "negative_retrieval",
    "compare_procedures": "comparison",
    "radiation_aware": "multi_factor",
    "clinical_scenario": "clinical_reasoning",
}

ALLOWED_CATEGORIES: List[str] = [
    "basic_retrieval",
    "aggregation",
    "negative_retrieval",
    "comparison",
    "multi_factor",
    "clinical_reasoning",
]

CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "basic_retrieval": "Single-procedure appropriateness lookup",
    "aggregation": "List all appropriate procedures for a scenario",
    "negative_retrieval": "Identify not appropriate/contraindicated procedures",
    "comparison": "Compare appropriateness of two procedures",
    "multi_factor": "Consider multiple factors such as radiation or patient context",
    "clinical_reasoning": "Infer the clinical scenario from provided procedures",
}


@dataclass
class QARecord:
    question: str
    answer: str
    difficulty: Literal["easy", "medium", "hard"]
    category: str
    question_type: str
    source_variants: List[str] = field(default_factory=list)
    source_procedures: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def read_rows(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Variant", "Procedure", "Appropriateness Category"])
    return df


def group_by_variant(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {variant: group for variant, group in df.groupby("Variant")}


def procedures_by_appropriateness(group: pd.DataFrame) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for _, row in group.iterrows():
        appr = str(row["Appropriateness Category"]).strip()
        proc = str(row["Procedure"]).strip()
        result.setdefault(appr, []).append(proc)
    return result


def is_appropriate(appr: str) -> bool:
    lowered = appr.lower()
    return "usually appropriate" in lowered or "may be appropriate" in lowered


def is_inappropriate(appr: str) -> bool:
    return "not appropriate" in appr.lower()


def gen_single_appropriateness(
    df: pd.DataFrame, n: int, rng: random.Random
) -> List[QARecord]:
    records = []
    sampled = df.sample(n=n, replace=True, random_state=rng.randint(0, 100000)).to_dict(
        "records"
    )
    for row in sampled:
        variant = str(row["Variant"]).strip()
        procedure = str(row["Procedure"]).strip()
        appr = str(row["Appropriateness Category"]).strip()
        rrl = row.get("Relative Radiation Level", "")
        rrl_str = (
            "N/A"
            if pd.isna(rrl) or rrl == ""
            else str(int(rrl) if isinstance(rrl, float) else rrl)
        )

        question = f"According to ACR guidelines, how appropriate is '{procedure}' for the clinical scenario: '{variant}'?"
        answer_parts = [f"Appropriateness: {appr}"]
        if rrl_str != "N/A":
            answer_parts.append(f"Relative radiation level: {rrl_str}")
        answer = ". ".join(answer_parts) + "."

        records.append(
            QARecord(
                question=question,
                answer=answer,
                difficulty="easy",
                category="basic_retrieval",
                question_type="single_appropriateness",
                source_variants=[variant],
                source_procedures=[procedure],
            )
        )
    return records


def gen_list_appropriate(
    groups: Dict[str, pd.DataFrame], n: int, rng: random.Random
) -> List[QARecord]:
    records = []
    variants = list(groups.keys())
    for _ in range(n):
        variant = rng.choice(variants)
        group = groups[variant]
        by_appr = procedures_by_appropriateness(group)

        usually_appropriate = by_appr.get("Usually Appropriate", [])
        may_be_appropriate = [
            p
            for k, v in by_appr.items()
            for p in v
            if "may be appropriate" in k.lower()
        ]

        if not usually_appropriate and not may_be_appropriate:
            continue

        question = f"What are the appropriate imaging or procedural options for '{variant}'? List all procedures rated 'Usually Appropriate' and 'May Be Appropriate'."

        answer_lines = []
        if usually_appropriate:
            answer_lines.append(
                f"Usually Appropriate: {', '.join(usually_appropriate)}"
            )
        if may_be_appropriate:
            answer_lines.append(f"May Be Appropriate: {', '.join(may_be_appropriate)}")
        answer = "; ".join(answer_lines) + "."

        records.append(
            QARecord(
                question=question,
                answer=answer,
                difficulty="medium",
                category="aggregation",
                question_type="list_appropriate",
                source_variants=[variant],
                source_procedures=usually_appropriate + may_be_appropriate,
            )
        )
    return records


def gen_list_inappropriate(
    groups: Dict[str, pd.DataFrame], n: int, rng: random.Random
) -> List[QARecord]:
    records = []
    variants = list(groups.keys())
    for _ in range(n):
        variant = rng.choice(variants)
        group = groups[variant]
        by_appr = procedures_by_appropriateness(group)

        not_appropriate = [
            p for k, v in by_appr.items() for p in v if is_inappropriate(k)
        ]
        if not not_appropriate:
            continue

        question = (
            f"Which procedures are rated 'Usually Not Appropriate' for '{variant}'?"
        )
        answer = f"Usually Not Appropriate: {', '.join(not_appropriate)}."

        records.append(
            QARecord(
                question=question,
                answer=answer,
                difficulty="medium",
                category="negative_retrieval",
                question_type="list_inappropriate",
                source_variants=[variant],
                source_procedures=not_appropriate,
            )
        )
    return records


def gen_compare_procedures(
    df: pd.DataFrame, n: int, rng: random.Random
) -> List[QARecord]:
    records = []
    groups = group_by_variant(df)
    variants = [v for v, g in groups.items() if len(g) >= 2]
    if not variants:
        return records

    for _ in range(n):
        variant = rng.choice(variants)
        group = groups[variant]
        sampled = group.sample(n=2, random_state=rng.randint(0, 100000))
        rows = sampled.to_dict("records")
        proc1, appr1 = rows[0]["Procedure"], rows[0]["Appropriateness Category"]
        proc2, appr2 = rows[1]["Procedure"], rows[1]["Appropriateness Category"]

        question = f"For the scenario '{variant}', compare the appropriateness of '{proc1}' versus '{proc2}'."
        answer = f"'{proc1}' is rated '{appr1}', while '{proc2}' is rated '{appr2}'."

        records.append(
            QARecord(
                question=question,
                answer=answer,
                difficulty="medium",
                category="comparison",
                question_type="compare_procedures",
                source_variants=[variant],
                source_procedures=[str(proc1), str(proc2)],
            )
        )
    return records


def gen_radiation_aware(
    df: pd.DataFrame, n: int, rng: random.Random
) -> List[QARecord]:
    records = []
    df_with_rrl = df[
        df["Relative Radiation Level"].notna() & (df["Relative Radiation Level"] != "")
    ]
    if df_with_rrl.empty:
        return records

    groups = group_by_variant(df_with_rrl)
    variants = list(groups.keys())

    for _ in range(n):
        variant = rng.choice(variants)
        group = groups[variant]

        low_rad_appropriate = []
        for _, row in group.iterrows():
            appr = str(row["Appropriateness Category"]).strip()
            rrl = row["Relative Radiation Level"]
            proc = str(row["Procedure"]).strip()
            try:
                rrl_val = int(float(rrl))
            except (ValueError, TypeError):
                continue
            if is_appropriate(appr) and rrl_val <= 2:
                low_rad_appropriate.append((proc, appr, rrl_val))

        if not low_rad_appropriate:
            for _, row in group.iterrows():
                appr = str(row["Appropriateness Category"]).strip()
                if is_appropriate(appr):
                    proc = str(row["Procedure"]).strip()
                    rrl = row["Relative Radiation Level"]
                    try:
                        rrl_val = int(float(rrl))
                    except (ValueError, TypeError):
                        rrl_val = 0
                    low_rad_appropriate.append((proc, appr, rrl_val))
            if not low_rad_appropriate:
                continue

        question = f"For a patient requiring evaluation for '{variant}' who needs to minimize radiation exposure, what are the appropriate low-radiation imaging options?"

        answer_items = [
            f"{p} (RRL: {r}, {a})"
            for p, a, r in sorted(low_rad_appropriate, key=lambda x: x[2])
        ]
        answer = (
            "Recommended options (sorted by radiation level): "
            + "; ".join(answer_items)
            + "."
        )

        records.append(
            QARecord(
                question=question,
                answer=answer,
                difficulty="hard",
                category="multi_factor",
                question_type="radiation_aware",
                source_variants=[variant],
                source_procedures=[p for p, _, _ in low_rad_appropriate],
            )
        )
    return records


def gen_clinical_scenario(
    groups: Dict[str, pd.DataFrame], n: int, rng: random.Random
) -> List[QARecord]:
    records = []
    variants = list(groups.keys())

    for _ in range(n):
        variant = rng.choice(variants)
        group = groups[variant]
        by_appr = procedures_by_appropriateness(group)
        usually_appropriate = by_appr.get("Usually Appropriate", [])
        if len(usually_appropriate) < 2:
            continue

        procs_sample = rng.sample(usually_appropriate, min(3, len(usually_appropriate)))
        question = f"A radiologist recommends the following as 'Usually Appropriate': {', '.join(procs_sample)}. What clinical scenario are they likely evaluating?"
        answer = f"The clinical scenario is: '{variant}'."

        records.append(
            QARecord(
                question=question,
                answer=answer,
                difficulty="hard",
                category="clinical_reasoning",
                question_type="clinical_scenario",
                source_variants=[variant],
                source_procedures=procs_sample,
            )
        )
    return records


def offline_generate(df: pd.DataFrame, count: int, seed: int) -> List[QARecord]:
    rng = random.Random(seed)
    groups = group_by_variant(df)

    distribution = {
        "single_appropriateness": int(count * 0.20),
        "list_appropriate": int(count * 0.25),
        "list_inappropriate": int(count * 0.15),
        "compare_procedures": int(count * 0.15),
        "radiation_aware": int(count * 0.10),
        "clinical_scenario": int(count * 0.15),
    }
    total_planned = sum(distribution.values())
    distribution["single_appropriateness"] += count - total_planned

    records: List[QARecord] = []
    records.extend(
        gen_single_appropriateness(df, distribution["single_appropriateness"], rng)
    )
    records.extend(gen_list_appropriate(groups, distribution["list_appropriate"], rng))
    records.extend(
        gen_list_inappropriate(groups, distribution["list_inappropriate"], rng)
    )
    records.extend(gen_compare_procedures(df, distribution["compare_procedures"], rng))
    records.extend(gen_radiation_aware(df, distribution["radiation_aware"], rng))
    records.extend(
        gen_clinical_scenario(groups, distribution["clinical_scenario"], rng)
    )

    rng.shuffle(records)
    return records[:count]


SYSTEM_PROMPT = """\
You are an expert medical educator creating benchmark questions for evaluating RAG systems on ACR Appropriateness Criteria for Interventional Radiology.

Your task is to generate diverse, clinically relevant questions that test different capabilities:
1. Basic retrieval: single procedure appropriateness lookup
2. Aggregation: listing all appropriate/inappropriate procedures for a scenario
3. Comparison: comparing two procedures for the same indication
4. Multi-factor reasoning: considering radiation exposure, patient factors
5. Clinical reasoning: identifying scenarios from procedure recommendations

Guidelines:
- Questions should sound natural, as if asked by a medical student, resident, or clinician
- Vary question phrasing (don't always start with "What" or "According to")
- Include realistic clinical context when appropriate
- Answers must be grounded entirely in the provided data
- For list questions, include ALL relevant procedures from the data
- category must be one of: basic_retrieval, aggregation, negative_retrieval, comparison, multi_factor, clinical_reasoning
- Use these category definitions: basic_retrieval=Single-procedure appropriateness lookup; aggregation=List all appropriate procedures for a scenario; negative_retrieval=Identify not appropriate/contraindicated procedures; comparison=Compare appropriateness of two procedures; multi_factor=Consider multiple factors such as radiation or patient context; clinical_reasoning=Infer the clinical scenario from provided procedures
"""


def build_variant_context(variant: str, group: pd.DataFrame) -> str:
    lines = [f"Scenario: {variant}"]
    for _, row in group.iterrows():
        proc = row["Procedure"]
        appr = row["Appropriateness Category"]
        rrl = row.get("Relative Radiation Level", "")
        rrl_str = f", RRL={int(float(rrl))}" if pd.notna(rrl) and rrl != "" else ""
        lines.append(f"  - {proc}: {appr}{rrl_str}")
    return "\n".join(lines)


def online_generate_batch(
    client: "OpenAI",
    model: str,
    variant_contexts: List[str],
    question_types: List[QuestionType],
    target_count: int,
) -> List[QARecord]:
    context_block = "\n\n".join(variant_contexts)
    type_descriptions = "\n".join(
        [
            "- single_appropriateness: Ask about one specific procedure's rating",
            "- list_appropriate: Ask for all appropriate procedures for a scenario",
            "- list_inappropriate: Ask for procedures that are NOT appropriate",
            "- compare_procedures: Compare two procedures for the same scenario",
            "- radiation_aware: Ask about low-radiation options or radiation considerations",
            "- clinical_scenario: Given procedures, ask what scenario they address",
        ]
    )

    category_descriptions = ", ".join(
        f"{name} = {desc}" for name, desc in CATEGORY_DESCRIPTIONS.items()
    )

    user_prompt = f"""\
Using ONLY the data below, generate {target_count} diverse benchmark questions.

Question types to include (distribute evenly):
{type_descriptions}

Allowed categories (use exactly one): {', '.join(ALLOWED_CATEGORIES)}
Category definitions: {category_descriptions}
For each question_type, prefer the matching category: {CATEGORY_MAP}

Data:
{context_block}

Output as JSON Lines (one JSON object per line, no markdown):
{{"question": "...", "answer": "...", "difficulty": "easy|medium|hard", "category": "...", "question_type": "...", "source_variants": ["..."], "source_procedures": ["..."]}}

Requirements:
- Answers must be complete and accurate based on the data
- For list questions, include ALL matching procedures
- Vary question phrasing naturally
- difficulty: easy for single lookups, medium for lists/comparisons, hard for multi-factor/reasoning
"""

    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        max_completion_tokens=3000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content or ""
    records = []
    for line in content.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            records.append(
                QARecord(
                    question=obj.get("question", "").strip(),
                    answer=obj.get("answer", "").strip(),
                    difficulty=obj.get("difficulty", "medium"),
                    category=obj.get("category", ""),
                    question_type=obj.get("question_type", ""),
                    source_variants=obj.get("source_variants", []),
                    source_procedures=obj.get("source_procedures", []),
                )
            )
        except json.JSONDecodeError:
            continue
    if not records:
        console.log("[yellow]No records parsed from model output; raw content:")
        console.log(content)
    return records


def online_generate(
    df: pd.DataFrame, count: int, batch_size: int, api_key: str, model: str
) -> List[QARecord]:
    if OpenAI is None:
        raise RuntimeError("openai package not available")

    client = OpenAI(api_key=api_key)
    groups = group_by_variant(df)
    variants = list(groups.keys())
    question_types: List[QuestionType] = [
        "single_appropriateness",
        "list_appropriate",
        "list_inappropriate",
        "compare_procedures",
        "radiation_aware",
        "clinical_scenario",
    ]

    records: List[QARecord] = []
    rng = random.Random(42)
    empty_rounds = 0

    with console.status("[bold green]Generating questions via OpenAI...") as status:
        while len(records) < count:
            remaining = count - len(records)
            sample_variants = rng.sample(variants, min(batch_size // 2, len(variants)))
            variant_contexts = [
                build_variant_context(v, groups[v]) for v in sample_variants
            ]

            batch = online_generate_batch(
                client,
                model,
                variant_contexts,
                question_types,
                target_count=min(batch_size, remaining),
            )
            records.extend(batch)
            console.log(f"Generated {len(records)}/{count} questions")

            if not batch:
                empty_rounds += 1
                if empty_rounds >= 3:
                    raise RuntimeError(
                        "Model returned no parsable questions after 3 attempts. "
                        "Check OPENAI_API_KEY/OPENAI_MODEL and consider running in offline mode."
                    )
            else:
                empty_rounds = 0

    return records[:count]


def save(records: Iterable[QARecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.to_json() + "\n")


@app.command()
def generate(
    csv_path: Path = typer.Option(DEFAULT_CSV, help="Path to the guideline CSV"),
    output_path: Path = typer.Option(DEFAULT_OUTPUT, help="Where to write the JSONL"),
    count: int = typer.Option(100, help="How many questions to generate"),
    mode: Literal["offline", "online"] = typer.Option(
        "offline",
        "--mode",
        case_sensitive=False,
        help="offline = deterministic templates; online = OpenAI generation",
    ),
    batch_size: int = typer.Option(20, help="Batch size for online mode"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    df = read_rows(csv_path)
    console.print(f"[bold]Loaded {len(df)} rows from {csv_path.name}")

    if mode == "offline":
        records = offline_generate(df, count, seed)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise typer.BadParameter("OPENAI_API_KEY is required for online mode")
        model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        records = online_generate(df, count, batch_size, api_key, model)

    save(records, output_path)

    by_type = {}
    by_difficulty = {}
    for r in records:
        by_type[r.question_type] = by_type.get(r.question_type, 0) + 1
        by_difficulty[r.difficulty] = by_difficulty.get(r.difficulty, 0) + 1

    console.print(f"\n[bold green]Wrote {len(records)} questions to {output_path}")
    console.print("[bold]Distribution by question type:")
    for qt, c in sorted(by_type.items()):
        console.print(f"  {qt}: {c}")
    console.print("[bold]Distribution by difficulty:")
    for d, c in sorted(by_difficulty.items()):
        console.print(f"  {d}: {c}")


def main() -> None:
    mode = "online" if os.getenv("OPENAI_API_KEY") else "offline"
    console.print(f"[bold]Running with mode={mode}")
    generate(
        csv_path=DEFAULT_CSV,
        output_path=DEFAULT_OUTPUT,
        count=100,
        mode=mode,
        batch_size=5,
        seed=42,
    )


if __name__ == "__main__":
    main()
