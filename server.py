#!/usr/bin/env python3
"""Máy chủ nhỏ cho giao diện HTML/CSS/JS trong web/.

    python server.py            # http://localhost:8000
    python server.py --port 9000

Chỉ dùng thư viện chuẩn: trình duyệt không chạy được mô hình nhúng lẫn LLM,
nên phần đó ở lại Python và frontend gọi qua JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv

from bench import CHUNKER, make_embedder
from src import EmbeddingStore, KnowledgeBaseAgent, load_corpus, make_llm

load_dotenv(override=False)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WEB_DIR = Path(__file__).parent / "web"
CORPUS_DIR = Path("data/quy-che-dao-tao-neu")

ROLES = {"student": "Sinh viên", "faculty": "Giảng viên", "staff": "Cán bộ / phòng ban"}
PROGRAMS = {"dai-tra": "Đại trà", "tien-tien": "Tiên tiến", "chat-luong-cao": "Chất lượng cao"}

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}


class Backend:
    """Nạp corpus, nhúng và LLM đúng một lần khi máy chủ khởi động."""

    def __init__(self) -> None:
        print("Đang nạp mô hình nhúng…")
        self.embedder = make_embedder()
        print(f"  Embedding backend: {getattr(self.embedder, '_backend_name', '?')}")
        self.llm = make_llm()
        print(f"  LLM backend      : {getattr(self.llm, '_backend_name', '?')}")

        print("Đang chia nhỏ và nhúng corpus…")
        docs = load_corpus(CORPUS_DIR, chunker=CHUNKER)
        self.store = EmbeddingStore("web", embedding_fn=self.embedder)
        self.store.add_documents(docs)
        self.agent = KnowledgeBaseAgent(self.store, llm_fn=self.llm)

        self.doc_count = len({d.metadata["doc_id"] for d in docs})
        self.chunk_count = self.store.get_collection_size()
        print(f"Sẵn sàng: {self.doc_count} tài liệu, {self.chunk_count} chunk.")

    def meta(self) -> dict:
        return {
            "docs": self.doc_count,
            "chunks": self.chunk_count,
            "embedder": getattr(self.embedder, "_backend_name", "mock"),
            "llm": getattr(self.llm, "_backend_name", "?"),
            "roles": ROLES,
            "programs": PROGRAMS,
        }

    def ask(self, question: str, audience: str | None, program: str | None, top_k: int) -> dict:
        metadata_filter = {}
        if audience:
            metadata_filter["audience"] = audience
        if program:
            metadata_filter["program"] = program

        results = self.store.search_with_filter(
            question, top_k=top_k, metadata_filter=metadata_filter or None
        )
        if not results:
            return {"answer": None, "error": None, "sources": [], "filter": metadata_filter}

        sources = [
            {
                "n": index,
                "doc_id": r["metadata"]["doc_id"],
                "title": r["metadata"].get("title", r["metadata"]["doc_id"]),
                "score": round(r["score"], 3),
                "content": r["content"],
                "url": r["metadata"].get("source_url", ""),
                "version": r["metadata"].get("document_version", ""),
                "audience": r["metadata"].get("audience", ""),
                "program": r["metadata"].get("program", ""),
            }
            for index, r in enumerate(results, 1)
        ]

        prompt = self.agent._build_prompt(question, results)
        try:
            answer, error = self.llm(prompt), None
        except Exception as exc:  # hạn mức, sai tên model, mất mạng…
            answer, error = None, str(exc)

        return {"answer": answer, "error": error, "sources": sources, "filter": metadata_filter}


class Handler(BaseHTTPRequestHandler):
    backend: Backend

    def log_message(self, fmt: str, *args) -> None:
        print(f"  {self.address_string()} {fmt % args}")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if self.path == "/api/meta":
            self._send_json(200, self.backend.meta())
            return

        name = "index.html" if self.path in ("/", "") else self.path.lstrip("/").split("?")[0]
        target = (WEB_DIR / name).resolve()
        # Chặn path traversal: chỉ phục vụ file nằm trong web/.
        if not target.is_file() or WEB_DIR.resolve() not in target.parents:
            self._send_json(404, {"error": "not found"})
            return

        self._send(200, target.read_bytes(), CONTENT_TYPES.get(target.suffix, "application/octet-stream"))

    def do_POST(self) -> None:
        if self.path != "/api/ask":
            self._send_json(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "JSON không hợp lệ"})
            return

        question = (payload.get("question") or "").strip()
        if not question:
            self._send_json(400, {"error": "Thiếu câu hỏi"})
            return

        try:
            top_k = max(1, min(10, int(payload.get("top_k", 3))))
        except (TypeError, ValueError):
            top_k = 3

        audience = payload.get("audience") if payload.get("audience") in ROLES else None
        program = payload.get("program") if payload.get("program") in PROGRAMS else None

        self._send_json(200, self.backend.ask(question, audience, program, top_k))


def main() -> int:
    parser = argparse.ArgumentParser(description="Máy chủ demo truy xuất quy chế NEU")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    Handler.backend = Backend()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\nMở http://{args.host}:{args.port} — Ctrl+C để dừng\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
