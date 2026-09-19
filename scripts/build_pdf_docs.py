#!/usr/bin/env python3
"""Turn the two AEP program regulations (PDF) into corpus documents.

fetch_public_pages.py only accepts HTML/text, so these two are handled here:
download -> pymupdf4llm -> repair the broken Vietnamese codepoints -> cut the
cover page, the trailing table of contents and the administrative chapter ->
write .md with front matter. Finally sources.csv is rebuilt from whatever .md
files are in the corpus directory, so it always matches one-to-one.

    pip install pymupdf4llm
    python scripts/build_pdf_docs.py
"""

from __future__ import annotations

import csv
import re
import sys
import time
import unicodedata
from pathlib import Path
from urllib.request import Request, urlopen

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CORPUS_DIR = Path("data/quy-che-dao-tao-neu")
CACHE_DIR = Path("build/pdf-cache")
USER_AGENT = "Day7DataFoundationsCourse/1.0 (+educational-lab)"

# The PDFs encode "ư"/"Ư" as U+01A3/U+01A2 (LATIN LETTER OI). Left alone, a query
# typed "chương trình chất lượng cao" cannot match "chƣơng trình chất lƣợng cao".
BROKEN_CODEPOINTS = {"\u01a3": "\u01b0", "\u01a2": "\u01af"}

SOURCES = [
    {
        "doc_id": "dao-tao-tien-tien",
        "title": "Quy định đào tạo đại học chính quy theo chương trình Tiên tiến",
        "url": "https://aep.neu.edu.vn/wp-content/uploads/2022/07/Quy-dinh-ve-dao-tao-theo-chuong-trinh-tien-tien.pdf",
        "document_version": "not-stated",  # cover page reads "HA NOI THANG ... NAM 2012"
        "program": "tien-tien",
        "stop_at": None,
        # Only the articles that actually differ from the mainstream regulation.
        # The rest runs parallel to it article by article, so keeping it adds
        # 70k characters of near-duplicate text that crowds out every other file.
        "keep_articles": [2, 7, 8, 10],
    },
    {
        "doc_id": "dao-tao-chat-luong-cao",
        "title": "Quy định đào tạo đại học chính quy chương trình Chất lượng cao",
        "url": "https://aep.neu.edu.vn/wp-content/uploads/2022/07/Quy-dinh-dao-tao-Chat-luong-cao.pdf",
        "document_version": "1299/QD-DHKTQD",
        "program": "chat-luong-cao",
        # Dieu 36+ are responsibilities of offices and lecturers: a different audience,
        # so they are left out rather than mixed into an audience=student file.
        "stop_at": r"\*\*Điều\s+36\.",
        "keep_articles": [2, 7, 8, 10, 11],
    },
]

ARTICLE_HEADING = re.compile(r"\*\*(Điều\s+\d+\s*[.:][^*\n]{3,80})\*\*")
# Same marker, but capturing the number so single articles can be pulled out.
ARTICLE_MARKER = re.compile(r"\*\*\s*(Điều\s+(\d+)\s*[.:][^*\n]{0,90}?)\s*\*\*")


def extract_articles(text: str, numbers: list[int]) -> str:
    """Keep only the listed articles, as markdown headings.

    Bold "**Điều 7. ...**" becomes "## Điều 7. ...", which makes these files
    structurally identical to the HTML-derived ones so a heading-based chunker
    treats the whole corpus the same way.
    """
    markers = list(ARTICLE_MARKER.finditer(text))
    wanted = set(numbers)

    sections = []
    for position, match in enumerate(markers):
        if int(match.group(2)) not in wanted:
            continue
        stop = markers[position + 1].start() if position + 1 < len(markers) else len(text)
        body = text[match.end():stop].strip()
        # Drop a chapter heading that belongs to the next article, not this one.
        body = re.sub(r"\n#{1,6}[^\n]*$", "", body).strip()
        sections.append(f"## {match.group(1).strip()}\n\n{body}")

    return "\n\n".join(sections)


def yaml_scalar(value: str) -> str:
    """Emit a plain YAML scalar, quoting only when the value would need it.

    The CHECKPOINT 2 script compares front-matter values literally, so a quoted
    doc_id never equals the file stem. Plain scalars keep both checkers happy.
    """
    text = str(value)
    if not text or text != text.strip() or text[0] in "\"'{}[]&*!|>%@`#-?:," or ": " in text or " #" in text:
        return '"' + text.replace('"', '\\"') + '"'
    return text


def download(url: str, destination: Path) -> Path:
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=120).read())
    time.sleep(1.0)
    return destination


def repair_text(text: str) -> str:
    for broken, correct in BROKEN_CODEPOINTS.items():
        text = text.replace(broken, correct)
    return unicodedata.normalize("NFC", text)


def strip_trailing_index(text: str) -> str:
    """Drop the table of contents that both PDFs carry after the body."""
    marker = re.search(r"\*\*?MỤC LỤC", text)
    if marker:
        return text[: marker.start()]

    # No explicit marker: the index shows up as many headings packed together.
    positions = [m.start() for m in ARTICLE_HEADING.finditer(text)]
    run_start = None
    for earlier, later in zip(positions, positions[1:]):
        if later - earlier < 400:
            run_start = run_start if run_start is not None else earlier
        else:
            run_start = None
    return text[:run_start] if run_start else text


def clean(raw_markdown: str, stop_at: str | None) -> str:
    text = repair_text(raw_markdown)

    first_article = re.search(r"\*\*Điều\s+1\s*[.:]", text)
    if first_article:
        text = text[first_article.start():]

    text = strip_trailing_index(text)

    if stop_at:
        stop = re.search(stop_at, text)
        if stop:
            text = text[: stop.start()]

    # Page numbers survive extraction as lines holding nothing but digits.
    text = re.sub(r"^[ 	]*\d{1,3}[ 	]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def rebuild_manifest() -> int:
    """Regenerate sources.csv from the front matter of every .md in the corpus."""
    fields = ["doc_id", "file_path", "title", "source_url", "retrieved_at", "document_version", "license_or_permission"]
    rows = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        front = path.read_text(encoding="utf-8").split("---")[1]
        meta = {k: v.strip().strip('"') for k, v in re.findall(r"^(\w+):\s*(.*)$", front, re.M)}
        rows.append({
            "doc_id": meta.get("doc_id", path.stem), "file_path": str(path),
            "title": meta.get("title", ""), "source_url": meta.get("source_url", ""),
            "retrieved_at": meta.get("retrieved_at", ""),
            "document_version": meta.get("document_version", ""),
            "license_or_permission": "public-source",
        })
    with (CORPUS_DIR / "sources.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    try:
        import pymupdf4llm
    except ImportError:
        print("Can cai truoc: pip install pymupdf4llm")
        return 2

    retrieved_at = time.strftime("%Y-%m-%d")
    for source in SOURCES:
        pdf_path = download(source["url"], CACHE_DIR / f"{source['doc_id']}.pdf")
        body = clean(pymupdf4llm.to_markdown(str(pdf_path), show_progress=False), source["stop_at"])
        body = extract_articles(body, source["keep_articles"])

        front_matter = {
            "doc_id": source["doc_id"], "title": source["title"], "source_url": source["url"],
            "retrieved_at": retrieved_at, "document_version": source["document_version"],
            "audience": "student", "department": "quan-ly-dao-tao",
            "program": source["program"], "category": "regulation", "language": "vi",
        }
        yaml = "\n".join(f"{k}: {yaml_scalar(v)}" for k, v in front_matter.items())
        output = CORPUS_DIR / f"{source['doc_id']}.md"
        output.write_text(f"---\n{yaml}\n---\n\n# {source['title']}\n\n{body}\n", encoding="utf-8")

        articles = len(re.findall(r"^## Điều", body, re.MULTILINE))
        print(f"[OK] {source['doc_id']:24} {len(body):>7,} ky tu, {articles:>2} Dieu -> {output}")

    print(f"\nsources.csv dung lai tu {rebuild_manifest()} tai lieu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
