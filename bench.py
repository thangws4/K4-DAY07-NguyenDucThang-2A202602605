#!/usr/bin/env python3
"""Benchmark harness for Phase 2 — not graded by pytest, this is the measuring tool.

    python bench.py

Chunking happens here, outside EmbeddingStore: one chunk becomes one Document,
so a strategy can be swapped without touching src/. Front matter is spread onto
every chunk, otherwise search_with_filter() would have nothing to filter on.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from src import (
    EmbeddingStore,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
    _mock_embed,
    load_corpus,
)
from src import make_embedder as _select_embedder

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Khoá API cho embedder/LLM nằm trong .env, nạp ngay khi import module này.
load_dotenv(override=False)

CORPUS_DIR = Path("data/quy-che-dao-tao-neu")


class HeadingChunker:
    """One article per chunk, using the structure the drafters already wrote.

    A regulation is authored as numbered articles, so "Điều 26. ..." is already
    a complete unit of meaning — better than any boundary we could guess. Two
    things this has to get right on this corpus:

    - The HTML-derived files mark articles as "## Điều N", the PDF-derived ones
      as bold "**Điều N**". Splitting on markdown headings alone leaves the two
      PDF regulations as one huge blob that falls back to recursive splitting,
      and they then take over every top-k.
    - A heading with no body of its own ("CHƯƠNG III. TỔ CHỨC ĐÀO TẠO") is not a
      retrieval unit. It is carried onto the next section instead of being
      emitted as its own contentless chunk.

    Sections longer than max_chars fall back to RecursiveChunker, and every
    piece gets the heading pasted back on: without that, the second piece
    onwards no longer says which article it belongs to.
    """

    SECTION_BREAK = re.compile(r"\n(?=#{1,6} |\*\*\s*(?:Điều|CHƯƠNG|Chương)\s)")

    def __init__(self, max_chars: int = 1200) -> None:
        self.max_chars = max_chars
        self._fallback = RecursiveChunker(chunk_size=max_chars)

    @staticmethod
    def _split_heading(section: str) -> tuple[str, str]:
        """Return (heading, body); body is empty when the section is a bare title."""
        lines = section.split("\n")
        taken = 0
        for line in lines:
            stripped = line.strip()
            if not stripped and taken:
                taken += 1
                continue
            is_heading = stripped.startswith("#") or (
                stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4
            )
            if not is_heading:
                break
            taken += 1

        heading = "\n".join(lines[:taken]).strip()
        body = "\n".join(lines[taken:]).strip()
        return heading, body

    def chunk(self, text: str) -> list[str]:
        chunks: list[str] = []
        carried = ""  # headings that had no body of their own

        for section in self.SECTION_BREAK.split(text):
            section = section.strip()
            if not section:
                continue

            heading, body = self._split_heading(section)
            if not body:
                carried = f"{carried}\n{heading}".strip() if carried else heading
                continue

            if carried:
                heading = f"{carried}\n{heading}".strip()
                carried = ""

            whole = f"{heading}\n\n{body}" if heading else body
            if len(whole) <= self.max_chars:
                chunks.append(whole)
                continue

            for piece in self._fallback.chunk(body):
                chunks.append(f"{heading}\n\n{piece}" if heading else piece)

        if carried:
            chunks.append(carried)
        return chunks


# === DOI DUNG MOT DONG NAY sang chien luoc cua ban, giu nguyen phan con lai ===
CHUNKER = HeadingChunker(max_chars=1200)
# CHUNKER = FixedSizeChunker(chunk_size=800, overlap=100)
# CHUNKER = RecursiveChunker(chunk_size=800)
# CHUNKER = SentenceChunker(max_sentences_per_chunk=5)

QUERIES = [
    {
        "id": 1,
        "dang": "tra so lieu",
        "query": "Trường tổ chức cho sinh viên đăng ký học muộn nhất bao lâu trước khi bắt đầu học kỳ?",
        "filter": None,
        "gold_doc": "dang-ky-hoc-phan",
    },
    {
        "id": 2,
        "dang": "hoi dieu kien",
        "query": "Học cải thiện điểm được tối đa bao nhiêu tín chỉ trong học kỳ 1?",
        "filter": None,
        "gold_doc": "dang-ky-hoc-phan",
    },
    {
        "id": 3,
        "dang": "hoi quy trinh (CAN FILTER)",
        # Câu hỏi không nêu ai là người hỏi, và "điểm thi" là từ vựng dùng chung
        # giữa Điều 26 (student) và Điều 19/23 (faculty). Không lọc thì
        # trach-nhiem-giang-vien chiếm hạng 1; lọc xong Điều 26 mới lên hạng 1.
        "query": "Khi không đồng ý với điểm thi thì làm gì?",
        "filter": {"audience": "student"},
        "gold_doc": "phuc-khao-khieu-nai-diem",
    },
    {
        "id": 4,
        "dang": "hoi dieu kien (khac chuong trinh)",
        "query": "Sinh viên được tuyển chọn vào chương trình Chất lượng cao như thế nào?",
        "filter": {"program": "chat-luong-cao"},
        "gold_doc": "dao-tao-chat-luong-cao",
    },
    {
        "id": 5,
        "dang": "liet ke",
        "query": "Điều kiện để được xét công nhận tốt nghiệp gồm những gì?",
        "filter": None,
        "gold_doc": "tot-nghiep",
    },
]


def make_embedder():
    """Chọn backend nhúng theo $EMBEDDING_PROVIDER trong .env (Phụ lục B của lab).

    Không chọn gì thì lab dùng mock, mà điểm số của mock là nhiễu MD5 — nên cảnh
    báo to khi rơi vào trường hợp đó, thay vì để người đọc tưởng đang đo chất
    lượng truy xuất thật.
    """
    embedder = _select_embedder()
    if embedder is _mock_embed:
        print(
            "!! Dang dung MockEmbedder (bam MD5): moi diem so duoi day chi la nhieu.\n"
            "   Dat EMBEDDING_PROVIDER=local (hoac openai/gemini) trong .env de do that.\n"
        )
    return embedder


def main() -> int:
    docs = load_corpus(CORPUS_DIR, chunker=CHUNKER)
    embedder = make_embedder()
    store = EmbeddingStore("bench", embedding_fn=embedder)
    store.add_documents(docs)

    files = len({d.metadata["doc_id"] for d in docs})
    lengths = [len(d.content) for d in docs]
    print(f"Embedder    : {getattr(embedder, '_backend_name', type(embedder).__name__)}")
    print(f"Chien luoc  : {type(CHUNKER).__name__}")
    print(f"Corpus      : {files} tai lieu -> {store.get_collection_size()} chunk")
    print(f"Do dai chunk: trung binh {sum(lengths) // len(lengths)}, min {min(lengths)}, max {max(lengths)}")
    print(f"Chunk < 120 ky tu: {sum(1 for n in lengths if n < 120)}\n")

    hits = 0
    for case in QUERIES:
        results = store.search_with_filter(case["query"], top_k=3, metadata_filter=case["filter"])
        found = [r["metadata"]["doc_id"] for r in results]
        ok = case["gold_doc"] in found
        hits += ok

        print(f'[{case["id"]}] {case["dang"]}  filter={case["filter"] or "-"}')
        print(f'    {case["query"]}')
        for rank, result in enumerate(results, 1):
            mark = "<==" if result["metadata"]["doc_id"] == case["gold_doc"] else "   "
            print(
                f'    {rank}. {result["score"]:+.3f} {result["metadata"]["doc_id"]:28}'
                f' #{result["metadata"]["chunk_index"]:<3} {mark}'
            )
        print(f'    gold_doc trong top-3: {"CO" if ok else "KHONG"}\n')

    print(f"Tom tat: {hits}/{len(QUERIES)} cau co gold_doc trong top-3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
