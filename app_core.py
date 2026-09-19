"""Logic dùng chung cho các trang Streamlit.

Trang chỉ lo phần hiển thị; nạp corpus, nhúng và gọi LLM nằm hết ở đây để hai
trang không dựng hai store khác nhau cho cùng một cấu hình.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from bench import HeadingChunker
from src import (
    EchoLLM,
    EmbeddingStore,
    FixedSizeChunker,
    KnowledgeBaseAgent,
    RecursiveChunker,
    SentenceChunker,
    _mock_embed,
    load_corpus,
    make_embedder,
    make_llm,
)

# Khoá API nằm trong .env, không bao giờ nằm trong mã nguồn.
load_dotenv(override=False)

CORPUS_DIR = Path("data/quy-che-dao-tao-neu")
DEFAULT_STRATEGY = "Theo tiêu đề"
DEFAULT_SIZE = 1200

STRATEGIES = {
    "Theo tiêu đề": lambda size: HeadingChunker(max_chars=size),
    "Đệ quy": lambda size: RecursiveChunker(chunk_size=size),
    "Kích thước cố định": lambda size: FixedSizeChunker(chunk_size=size, overlap=size // 10),
    "Theo câu": lambda _size: SentenceChunker(max_sentences_per_chunk=5),
}

# Nhãn người dùng hiểu được -> giá trị metadata trong front matter.
ROLES = {
    "Sinh viên": "student",
    "Giảng viên": "faculty",
    "Cán bộ / phòng ban": "staff",
}
PROGRAMS = {
    "Đại trà": "dai-tra",
    "Tiên tiến": "tien-tien",
    "Chất lượng cao": "chat-luong-cao",
}


@st.cache_resource(show_spinner="Đang nạp mô hình nhúng…")
def get_embedder():
    """Backend nhúng theo $EMBEDDING_PROVIDER trong .env (Phụ lục B của lab)."""
    return make_embedder()


def embedder_is_mock() -> bool:
    return get_embedder() is _mock_embed


@st.cache_resource(show_spinner=False)
def get_llm():
    return make_llm()


def llm_is_real() -> bool:
    return not isinstance(get_llm(), EchoLLM)


@st.cache_resource(show_spinner="Đang chia nhỏ và nhúng corpus…")
def get_store(strategy: str, size: int):
    chunker = STRATEGIES[strategy](size)
    docs = load_corpus(CORPUS_DIR, chunker=chunker)
    store = EmbeddingStore(f"{strategy}-{size}", embedding_fn=get_embedder())
    store.add_documents(docs)
    return store, docs


@st.cache_data(show_spinner=False)
def corpus_table() -> pd.DataFrame:
    rows = []
    for doc in load_corpus(CORPUS_DIR):
        meta = doc.metadata
        rows.append(
            {
                "doc_id": meta.get("doc_id", ""),
                "Tiêu đề": meta.get("title", ""),
                "audience": meta.get("audience", ""),
                "program": meta.get("program", ""),
                "category": meta.get("category", ""),
                "Số ký tự": len(doc.content),
                "Phiên bản": meta.get("document_version", ""),
                "Nguồn": meta.get("source_url", ""),
            }
        )
    return pd.DataFrame(rows)


def answer_question(store, question: str, top_k: int, metadata_filter: dict | None):
    """Truy xuất có lọc, dựng prompt, gọi LLM.

    KnowledgeBaseAgent.answer() luôn dùng search() nên không nhận bộ lọc; ở đây
    truy xuất trước rồi mới nhờ agent dựng prompt, để câu trả lời bám đúng tập
    tài liệu mà người dùng đã lọc.

    Trả về (results, prompt, answer, error). Lỗi gọi API được trả về chứ không
    ném ra: một lần quá hạn mức hay sai tên model không nên làm sập cả trang,
    và các điều khoản truy xuất được vẫn còn nguyên giá trị.
    """
    results = store.search_with_filter(question, top_k=top_k, metadata_filter=metadata_filter)
    if not results:
        return [], "", None, None

    agent = KnowledgeBaseAgent(store, llm_fn=get_llm())
    prompt = agent._build_prompt(question, results)
    try:
        return results, prompt, agent.llm_fn(prompt), None
    except Exception as error:
        return results, prompt, None, str(error)
