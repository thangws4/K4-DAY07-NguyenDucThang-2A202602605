#!/usr/bin/env python3
"""Split the raw NEU credit-system regulation into per-topic corpus documents.

The source page (mfe.neu.edu.vn) carries the whole regulation - 41 Dieu across
7 Chuong - as one HTML page. docs/DATA_COLLECTION.md section 4 asks for one
audience per file, so Chuong VI (responsibilities of lecturers and faculty
offices) is separated from the student-facing articles. That split is what
gives search_with_filter() something real to filter.

    python scripts/split_quy_dinh_neu.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CORPUS_DIR = Path("data/quy-che-dao-tao-neu")
RAW_FILE = CORPUS_DIR / "quy-dinh-dao-tao-tin-chi-raw.md"

SOURCE_URL = "https://mfe.neu.edu.vn/bo-quy-dinh-dao-tao-tin-chi-dhktqd-2/"
# Stated in the preamble: "Ban hanh kem theo Quyet dinh so 1212/QD-DHKTQD, ngay 12 thang 12 nam 2012".
DOCUMENT_VERSION = "1212/QD-DHKTQD (2012-12-12)"
DEPARTMENT = "quan-ly-dao-tao"
LANGUAGE = "vi"

ARTICLE_RE = re.compile(r"^Điều\s*(\d+)\s*\.?\s*(.*)$")
CHUONG_RE = re.compile(r"^Chương\s+[IVXLC]+\s*$")
END_MARKER = "Bài viết liên quan"

# doc_id -> (title, [Dieu numbers], audience, category)
DOCUMENTS = [
    ("nhiem-vu-quyen-sinh-vien", "Nhiệm vụ, quyền và tiêu chí đánh giá sinh viên", [4, 5], "student", "student-rights"),
    ("dang-ky-hoc-phan", "Đăng ký khối lượng học tập, rút học phần và học lại", [10, 11], "student", "registration"),
    ("xep-hang-canh-bao-thoi-hoc", "Xếp hạng học lực, cảnh báo kết quả và buộc thôi học", [14, 15], "student", "academic-standing"),
    ("thi-ket-thuc-hoc-phan", "Kiểm tra, thi kết thúc học phần và cách tính điểm", [20, 21, 27], "student", "assessment"),
    ("phuc-khao-khieu-nai-diem", "Khiếu nại điểm và xem lại kết quả bài thi", [26], "student", "assessment"),
    ("tot-nghiep", "Thực tập cuối khoá, xét và công nhận tốt nghiệp", [29, 30, 31], "student", "graduation"),
    ("trach-nhiem-giang-vien", "Trách nhiệm của giảng viên trong ra đề, coi thi và chấm thi", [19, 22, 23, 37], "faculty", "assessment"),
    ("trach-nhiem-khoa-vien-phong", "Trách nhiệm của Khoa, Viện, Phòng QLĐT và cố vấn học tập", [35, 36, 38, 40], "staff", "administration"),
]


def yaml_scalar(value: str) -> str:
    """Emit a plain YAML scalar, quoting only when the value would need it.

    The CHECKPOINT 2 script compares front-matter values literally, so a quoted
    doc_id never equals the file stem. Plain scalars keep both checkers happy.
    """
    text = str(value)
    if not text or text != text.strip() or text[0] in "\"'{}[]&*!|>%@`#-?:," or ": " in text or " #" in text:
        return '"' + text.replace('"', '\\"') + '"'
    return text


def is_section_heading(line: str) -> bool:
    """True for 'Chuong IV' and the shouty chapter titles that follow it."""
    stripped = line.strip()
    if not stripped or CHUONG_RE.match(stripped):
        return bool(stripped)
    letters = [c for c in stripped if c.isalpha()]
    return len(letters) > 3 and stripped == stripped.upper()


def parse_articles(raw_text: str) -> dict[int, tuple[str, str]]:
    """Return {article_number: (title, body)} from the cleaned region of the page."""
    lines = raw_text.split("\n")
    end = next((i for i, l in enumerate(lines) if END_MARKER in l), len(lines))

    starts: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines[:end]):
        match = ARTICLE_RE.match(line.strip())
        if match:
            starts.append((index, int(match.group(1)), match.group(2).strip()))

    articles: dict[int, tuple[str, str]] = {}
    for position, (line_index, number, title) in enumerate(starts):
        stop = starts[position + 1][0] if position + 1 < len(starts) else end
        body_lines = lines[line_index + 1 : stop]
        while body_lines and is_section_heading(body_lines[-1]):
            body_lines.pop()
        body = re.sub(r"\n{3,}", "\n\n", "\n".join(body_lines)).strip()
        articles[number] = (title, body)
    return articles


def build() -> int:
    if not RAW_FILE.exists():
        print(f"Chua co {RAW_FILE}. Chay fetch_public_pages.py truoc.")
        return 2

    raw_text = RAW_FILE.read_text(encoding="utf-8")
    retrieved_at = re.search(r'retrieved_at:\s*"?([\d-]+)"?', raw_text).group(1)
    articles = parse_articles(raw_text)
    print(f"Doc duoc {len(articles)} Dieu tu trang nguon\n")

    manifest_rows = []
    for doc_id, title, numbers, audience, category in DOCUMENTS:
        missing = [n for n in numbers if n not in articles]
        if missing:
            print(f"[BO QUA] {doc_id}: khong tim thay Dieu {missing}")
            continue

        sections = [f"## Điều {n}. {articles[n][0]}\n\n{articles[n][1]}" for n in numbers]
        front_matter = {
            "doc_id": doc_id, "title": title, "source_url": SOURCE_URL,
            "retrieved_at": retrieved_at, "document_version": DOCUMENT_VERSION,
            "audience": audience, "department": DEPARTMENT, "category": category, "language": LANGUAGE,
        }
        yaml = "\n".join(f"{k}: {yaml_scalar(v)}" for k, v in front_matter.items())
        body = "\n\n".join(sections)
        output = CORPUS_DIR / f"{doc_id}.md"
        output.write_text(f"---\n{yaml}\n---\n\n# {title}\n\n{body}\n", encoding="utf-8")

        print(f"[OK] {doc_id:32} {audience:8} Dieu {numbers} -> {len(body):>6,} ky tu")
        manifest_rows.append({
            "doc_id": doc_id, "file_path": str(output), "title": title,
            "source_url": SOURCE_URL, "retrieved_at": retrieved_at,
            "document_version": DOCUMENT_VERSION, "license_or_permission": "public-source",
        })

    manifest_path = CORPUS_DIR / "sources.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "doc_id", "file_path", "title", "source_url", "retrieved_at",
            "document_version", "license_or_permission"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    RAW_FILE.unlink()
    print(f"\nDa ghi {len(manifest_rows)} tai lieu + {manifest_path}; xoa ban tho.")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
