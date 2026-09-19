import pandas as pd
import streamlit as st

from app_core import (
    CORPUS_DIR,
    STRATEGIES,
    answer_question,
    corpus_table,
    get_embedder,
    get_llm,
    get_store,
)
from bench import QUERIES

st.title("Phòng thí nghiệm truy xuất")
st.caption("Công cụ đo cho Lab 7 · K4-L3A — đổi chiến lược chia nhỏ và xem nó đổi kết quả thế nào.")

with st.sidebar:
    st.subheader("Chiến lược")
    strategy = st.selectbox("Cách chia nhỏ", list(STRATEGIES), key="lab_strategy")
    size = st.slider(
        "Kích thước chunk (ký tự)",
        min_value=400,
        max_value=2000,
        value=1200,
        step=100,
        key="lab_size",
        help="Không áp dụng cho cách chia theo câu.",
        disabled=strategy == "Theo câu",
    )

    st.subheader("Bộ lọc siêu dữ liệu")
    audience = st.segmented_control(
        "Đối tượng", ["student", "faculty", "staff"], key="lab_audience", help="Bỏ chọn để không lọc."
    )
    program = st.segmented_control(
        "Chương trình", ["dai-tra", "tien-tien", "chat-luong-cao"], key="lab_program"
    )
    top_k = st.slider("Số kết quả (top-k)", 1, 10, 3, key="lab_top_k")

store, docs = get_store(strategy, size)

metadata_filter = {}
if audience:
    metadata_filter["audience"] = audience
if program:
    metadata_filter["program"] = program

with st.container(horizontal=True):
    st.metric("Tài liệu", len({d.metadata["doc_id"] for d in docs}))
    st.metric("Chunk", store.get_collection_size())
    st.metric("Dài trung bình", f"{sum(len(d.content) for d in docs) // len(docs)} ký tự")
    st.metric("Nhúng bằng", getattr(get_embedder(), "_backend_name", "mock").split("/")[-1])
    st.metric("LLM", getattr(get_llm(), "_backend_name", "?").split(":")[-1])

tra_cuu, so_sanh, corpus = st.tabs(["Tra cứu", "So sánh chiến lược", "Corpus"])

with tra_cuu:
    with st.form("search", border=False):
        question = st.text_input("Câu hỏi", value=QUERIES[2]["query"])
        submitted = st.form_submit_button("Tìm", icon=":material/search:", type="primary")

    if submitted and question.strip():
        results, prompt_text, answer, error = answer_question(
            store, question, top_k=top_k, metadata_filter=metadata_filter or None
        )

        if not results:
            st.warning("Không có chunk nào khớp bộ lọc. Nới bộ lọc ở thanh bên rồi thử lại.")
        else:
            st.subheader(f"Top-{len(results)} chunk")
            for index, result in enumerate(results, 1):
                meta = result["metadata"]
                with st.container(border=True):
                    with st.container(horizontal=True):
                        st.badge(f'[{index}]  {result["score"]:+.3f}', color="blue")
                        st.badge(meta["doc_id"])
                        st.badge(meta.get("audience", "?"), color="green")
                        st.badge(meta.get("program", "?"), color="orange")
                    st.markdown(result["content"])
                    st.caption(f'Nguồn: {meta.get("source_url", "?")} · phiên bản {meta.get("document_version", "?")}')

            st.subheader("Câu trả lời của agent")
            if error:
                st.error(f"Không gọi được LLM: {error}", icon=":material/cloud_off:")
            else:
                st.markdown(answer)
            with st.expander("Ngữ cảnh agent đã dựng (kiểm tra truy vết nguồn)"):
                st.code(prompt_text, language="text")

with so_sanh:
    st.markdown(
        "Chạy cùng 5 câu hỏi đánh giá qua từng chiến lược. "
        "Ô đánh dấu nghĩa là tài liệu chứa đáp án chuẩn nằm trong top-3."
    )
    if st.button("Chạy so sánh", icon=":material/play_arrow:", type="primary"):
        rows = []
        progress = st.progress(0.0, text="Đang đo…")
        for position, name in enumerate(STRATEGIES, 1):
            other_store, _ = get_store(name, size)
            row = {"Chiến lược": name, "Số chunk": other_store.get_collection_size()}
            hits = 0
            for case in QUERIES:
                found = [
                    r["metadata"]["doc_id"]
                    for r in other_store.search_with_filter(case["query"], top_k=3, metadata_filter=case["filter"])
                ]
                ok = case["gold_doc"] in found
                hits += ok
                row[f'Câu {case["id"]}'] = ok
            row["Đúng"] = f"{hits}/{len(QUERIES)}"
            rows.append(row)
            progress.progress(position / len(STRATEGIES), text=f"Xong {name}")
        progress.empty()
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Điểm số chỉ có ý nghĩa khi nhúng bằng mô hình thật, không phải mock.")

with corpus:
    st.dataframe(
        corpus_table(),
        hide_index=True,
        column_config={"Nguồn": st.column_config.LinkColumn("Nguồn", display_text="mở")},
    )
    st.caption(f"{CORPUS_DIR} · siêu dữ liệu đọc từ YAML front matter của từng file")
