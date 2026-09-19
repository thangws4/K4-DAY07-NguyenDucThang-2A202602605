from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Protocol

from .models import Document

FRONT_MATTER_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", re.MULTILINE)
# Only treat "#" as a comment when whitespace precedes it, so a URL fragment survives.
INLINE_COMMENT = re.compile(r"\s+#")


class Chunker(Protocol):
    """Anything with .chunk(text) -> list[str], e.g. the chunkers in src/chunking.py."""

    def chunk(self, text: str) -> list[str]: ...


def _parse_value(raw: str) -> str:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1]
    return INLINE_COMMENT.split(raw, maxsplit=1)[0].strip()


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """
    Split a document into (metadata, body).

    Handles both styles found in this lab: values quoted by
    scripts/fetch_public_pages.py, and hand-written values with inline comments.
    Text without front matter comes back as ({}, text).
    """
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    metadata = {
        key: _parse_value(value)
        for key, value in FRONT_MATTER_LINE.findall(parts[1])
    }
    return metadata, parts[2].strip()


def load_document(path: Path | str, chunker: Chunker | None = None) -> list[Document]:
    """
    Read one corpus file into Documents.

    Without a chunker the file becomes a single Document. With one, it becomes
    several Documents that share the same id, so EmbeddingStore.delete_document()
    still removes the whole source file in one call.
    """
    path = Path(path)
    metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))

    doc_id = metadata.get("doc_id") or path.stem
    metadata.setdefault("doc_id", doc_id)
    metadata["source_path"] = str(path)

    if chunker is None:
        return [Document(id=doc_id, content=body, metadata=dict(metadata))]

    chunks = chunker.chunk(body)
    # Document.id identifies the chunk; metadata["doc_id"] keeps pointing at the
    # source file, which is what delete_document() and metadata filters use.
    return [
        Document(
            id=f"{doc_id}#{index}",
            content=chunk,
            metadata={**metadata, "chunk_index": index, "chunk_count": len(chunks)},
        )
        for index, chunk in enumerate(chunks)
    ]


def load_corpus(
    directory: Path | str,
    chunker: Chunker | None = None,
    patterns: Iterable[str] = ("*.md",),
) -> list[Document]:
    """
    Load every corpus file in a directory.

    Defaults to *.md because docs/DATA_COLLECTION.md stores one document per .md
    file; that also skips helper files such as Links.txt and sources.csv. Pass
    patterns=("*.md", "*.txt") to include the plain-text samples in data/.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Khong tim thay thu muc corpus: {directory}")

    paths = sorted({p for pattern in patterns for p in directory.glob(pattern)})
    documents: list[Document] = []
    for path in paths:
        documents.extend(load_document(path, chunker=chunker))
    return documents
