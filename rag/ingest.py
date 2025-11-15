"""Utility to ingest PDFs in ``sources_pdf/`` into the LanceDB vector store."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import typer
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from rich.console import Console
from rich.progress import track

from .config import MANIFEST_PATH, SOURCES_PATH
from .knowledge import chunk_metadata, get_knowledge

console = Console()
app = typer.Typer(add_completion=False, help="Manage the knowledge base contents")


def load_manifest() -> Dict[str, Dict[str, str]]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: Dict[str, Dict[str, str]]) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def list_pdfs() -> List[Path]:
    if not SOURCES_PATH.exists():
        return []
    return sorted(SOURCES_PATH.glob("**/*.pdf"))


def extract_pages(pdf_path: Path) -> Iterator[Tuple[int, str]]:
    try:
        reader = PdfReader(str(pdf_path))
    except PdfReadError as exc:
        raise RuntimeError(f"Failed to read {pdf_path}: {exc}") from exc
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        yield idx, " ".join(text.split())


def chunk_pages(
    page_iter: Iterable[Tuple[int, str]],
    *,
    max_chars: int = 1800,
) -> Iterator[Tuple[int, int, str]]:
    buffer: List[str] = []
    start_page: int | None = None
    last_page: int | None = None
    current_len = 0

    def combined_len(parts: List[str]) -> int:
        if not parts:
            return 0
        return sum(len(part) for part in parts) + (len(parts) - 1)

    for page_num, page_text in page_iter:
        if not page_text:
            continue
        if start_page is None:
            start_page = page_num
        buffer.append(page_text)
        last_page = page_num
        current_len = combined_len(buffer)
        if current_len >= max_chars:
            assert start_page is not None
            assert last_page is not None
            yield start_page, last_page, " ".join(buffer).strip()
            buffer = []
            current_len = 0
            start_page = None
            last_page = None

    if buffer and start_page is not None and last_page is not None:
        yield start_page, last_page, " ".join(buffer).strip()


def ingest_pdf(pdf_path: Path, *, skip_existing: bool) -> Dict[str, str]:
    knowledge = get_knowledge()
    chunk_count = 0
    for chunk_id, (start_page, end_page, content) in enumerate(
        chunk_pages(extract_pages(pdf_path)), start=1
    ):
        metadata = chunk_metadata(
            pdf_path,
            chunk_id=chunk_id,
            page_range=f"p{start_page}-p{end_page}",
        )
        knowledge.add_content(
            name=f"{pdf_path.stem}-chunk-{chunk_id}",
            text_content=content,
            metadata=metadata,
            skip_if_exists=skip_existing,
        )
        chunk_count += 1
    return {
        "chunks": chunk_count,
        "last_ingested_at": datetime.now(timezone.utc).isoformat(),
        "mtime": str(int(pdf_path.stat().st_mtime)),
    }


@app.command()
def run(
    force: bool = typer.Option(
        False, "--force", help="Reprocess files even if unchanged"
    ),
    limit: int = typer.Option(
        0, "--limit", min=0, help="Ingest only the first N PDFs (0 = all)"
    ),
    skip_existing: bool = typer.Option(
        True, "--skip-existing", help="Skip re-uploading identical chunks"
    ),
) -> None:
    """Parse each PDF and push chunks into LanceDB."""
    manifest = load_manifest()
    pdfs = list_pdfs()
    if not pdfs:
        console.print("[yellow]No PDFs found in sources_pdf/. Add files first.")
        raise typer.Exit(code=0)

    targets = pdfs if limit in (0, None) else pdfs[:limit]
    console.print(f"[bold]Processing {len(targets)} PDF(s) from {SOURCES_PATH}")

    processed = 0
    for pdf_path in track(targets, description="Indexing"):
        key = str(pdf_path.relative_to(SOURCES_PATH))
        file_mtime = str(int(pdf_path.stat().st_mtime))
        already_ingested = manifest.get(key)
        if already_ingested and already_ingested.get("mtime") == file_mtime and not force:
            console.print(f"[cyan]Skipping {key}, no changes detected")
            continue

        stats = ingest_pdf(pdf_path, skip_existing=skip_existing)
        manifest[key] = stats
        manifest[key]["mtime"] = file_mtime
        console.print(
            f"[green]Ingested {key}: {stats['chunks']} chunk(s) @ {stats['last_ingested_at']}"
        )
        processed += 1

    save_manifest(manifest)
    console.print(
        f"[bold green]Finished ingesting {processed} file(s). Manifest saved to {MANIFEST_PATH}" 
    )


@app.command()
def clear(confirm: bool = typer.Option(False, "--yes", help="Skip confirmation")) -> None:
    """Remove all vectors from the knowledge base."""
    if not confirm:
        typer.confirm("This will delete all ingested content. Continue?", abort=True)
    knowledge = get_knowledge()
    knowledge.remove_all_content()
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()
    console.print("[red]Knowledge base cleared and manifest removed.")


if __name__ == "__main__":
    app()
