#!/usr/bin/env python3
"""Verify a collected corpus against the CHECKPOINT 2 checklist.

Implements section 6 of docs/DATA_COLLECTION.md. Unlike an inline one-liner this
handles the quoted YAML values that fetch_public_pages.py writes, and prints
UTF-8 on a Windows console.

    python scripts/check_corpus.py data/<ten-chu-de>
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REQUIRED_FIELDS = ["doc_id", "title", "source_url", "retrieved_at", "document_version", "audience"]
# audience alone is not enough: K4_VARIANT.md asks for at least one more filter field.
EXTRA_FILTER_FIELDS = ["department", "category", "language", "difficulty"]
VALID_AUDIENCE = {"student", "faculty", "staff", "all"}
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FRONT_MATTER_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$", re.M)
# Leftovers that mean the page was never cleaned by hand.
NOISE_MARKERS = ["Chuyển đến nội dung", "Skip to content", "Chuyển đến thanh công cụ", "Đăng nhập", "Cookie"]


def parse_front_matter(path: Path) -> tuple[dict[str, str], str]:
    """Return (metadata, body). Values keep no surrounding quotes."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    metadata = {}
    for key, value in FRONT_MATTER_LINE.findall(parts[1]):
        value = value.split("#")[0].strip() if not value.startswith(('"', "'")) else value
        # fetch_public_pages.py quotes every value; hand-written files usually do not.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        metadata[key] = value.strip()
    return metadata, parts[2]


def check_file(path: Path) -> tuple[list[str], dict[str, str]]:
    problems: list[str] = []
    metadata, body = parse_front_matter(path)

    if not metadata:
        return ["khong doc duoc YAML front matter"], {}

    missing = [f for f in REQUIRED_FIELDS if not metadata.get(f)]
    if missing:
        problems.append(f"thieu metadata: {', '.join(missing)}")

    if metadata.get("doc_id") and metadata["doc_id"] != path.stem:
        problems.append(f"doc_id ({metadata['doc_id']}) khac ten file ({path.stem})")

    audience = metadata.get("audience")
    if audience and audience not in VALID_AUDIENCE:
        problems.append(f"audience khong hop le: {audience} (can: {'/'.join(sorted(VALID_AUDIENCE))})")

    retrieved_at = metadata.get("retrieved_at")
    if retrieved_at and not DATE_PATTERN.match(retrieved_at):
        problems.append(f"retrieved_at sai dinh dang YYYY-MM-DD: {retrieved_at}")

    source_url = metadata.get("source_url", "")
    if source_url and not source_url.startswith(("http://", "https://")):
        problems.append(f"source_url khong phai URL: {source_url}")

    if not any(metadata.get(f) for f in EXTRA_FILTER_FIELDS):
        problems.append(f"can it nhat 1 truong loc ngoai audience ({'/'.join(EXTRA_FILTER_FIELDS)})")

    found_noise = [m for m in NOISE_MARKERS if m in body]
    if found_noise:
        problems.append(f"con rac chua lam sach: {', '.join(found_noise)}")

    if len(body.strip()) < 200:
        problems.append(f"noi dung qua ngan ({len(body.strip())} ky tu) — co du de tra loi query khong?")

    return problems, metadata


def main() -> int:
    if len(sys.argv) < 2:
        print("Dung: python scripts/check_corpus.py data/<ten-chu-de>")
        return 2

    directory = Path(sys.argv[1])
    if not directory.is_dir():
        print(f"Khong tim thay thu muc: {directory}")
        return 2

    md_files = sorted(directory.glob("*.md"))
    if not md_files:
        print(f"Khong co file .md nao trong {directory}")
        return 1

    print(f"=== KIEM TRA CORPUS: {directory} ===\n")

    failed = 0
    doc_ids: list[str] = []
    audiences: dict[str, int] = {}
    versions: list[str] = []

    for path in md_files:
        problems, metadata = check_file(path)
        if metadata.get("doc_id"):
            doc_ids.append(metadata["doc_id"])
        if metadata.get("audience"):
            audiences[metadata["audience"]] = audiences.get(metadata["audience"], 0) + 1
        if metadata.get("document_version"):
            versions.append(metadata["document_version"])

        size = len(path.read_text(encoding="utf-8"))
        if problems:
            failed += 1
            print(f"[FAIL] {path.name}  ({size:,} ky tu)")
            for problem in problems:
                print(f"        - {problem}")
        else:
            print(f"[ OK ] {path.name}  ({size:,} ky tu)")

    print("\n=== TONG HOP CORPUS ===")

    corpus_ok = True

    count_ok = 5 <= len(md_files) <= 10
    corpus_ok &= count_ok
    print(f"[{'OK  ' if count_ok else 'FAIL'}] So tai lieu: {len(md_files)} (can 5-10)")

    duplicates = {d for d in doc_ids if doc_ids.count(d) > 1}
    corpus_ok &= not duplicates
    print(f"[{'OK  ' if not duplicates else 'FAIL'}] doc_id duy nhat" + (f" — trung: {duplicates}" if duplicates else ""))

    audience_ok = len(audiences) >= 2
    corpus_ok &= audience_ok
    print(f"[{'OK  ' if audience_ok else 'FAIL'}] audience co >=2 gia tri: {audiences or '(khong co)'}")
    if not audience_ok:
        print("        -> metadata_filter se khong loc duoc gi. Tach tai lieu gop nhieu doi tuong ra file rieng.")

    if "student" not in audiences:
        corpus_ok = False
        print("[FAIL] khong co tai lieu nao audience=student — benchmark bat buoc can filter nay")

    manifest_path = directory / "sources.csv"
    if not manifest_path.exists():
        corpus_ok = False
        print(f"[FAIL] thieu {manifest_path}")
    else:
        with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
            manifest_ids = sorted(row["doc_id"] for row in csv.DictReader(manifest_file) if row.get("doc_id"))
        matches = manifest_ids == sorted(doc_ids)
        corpus_ok &= matches
        print(f"[{'OK  ' if matches else 'FAIL'}] sources.csv khop 1-1 voi file .md")
        if not matches:
            only_csv = set(manifest_ids) - set(doc_ids)
            only_md = set(doc_ids) - set(manifest_ids)
            if only_csv:
                print(f"        - chi co trong csv (xoa dong nay): {only_csv}")
            if only_md:
                print(f"        - chi co trong .md (thieu dong csv): {only_md}")

    strays = [p.name for p in directory.iterdir() if p.suffix.lower() in {".pdf", ".html", ".htm", ".docx"}]
    if strays:
        print(f"[WARN] file tho chua chuyen doi trong data/: {', '.join(strays)}")

    if versions and all(v == "not-stated" for v in versions):
        print("[WARN] moi document_version deu 'not-stated' — kiem tra lai xem nguon co ghi ngay hieu luc khong")

    print()
    print("=== CHECKPOINT 2: " + ("DAT" if corpus_ok and not failed else "CHUA DAT") + " ===")
    return 0 if corpus_ok and not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
