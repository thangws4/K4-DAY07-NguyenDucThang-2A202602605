import streamlit as st

from app_core import (
    DEFAULT_SIZE,
    DEFAULT_STRATEGY,
    PROGRAMS,
    ROLES,
    answer_question,
    embedder_is_mock,
    get_store,
    llm_is_real,
)

SUGGESTIONS = {
    ":blue[:material/how_to_reg:] Đăng ký học phần": "Đăng ký học phần muộn nhất là bao lâu trước khi bắt đầu học kỳ?",
    ":green[:material/grade:] Cải thiện điểm": "Học cải thiện điểm được tối đa bao nhiêu tín chỉ trong một học kỳ?",
    ":orange[:material/gavel:] Phúc khảo": "Khi không đồng ý với điểm thi thì làm gì?",
    ":violet[:material/school:] Tốt nghiệp": "Điều kiện để được xét công nhận tốt nghiệp gồm những gì?",
}

st.title("Hỏi đáp quy chế đào tạo")
st.caption("Trả lời dựa trên văn bản quy chế của Đại học Kinh tế Quốc dân, có dẫn nguồn từng điều.")

with st.sidebar:
    st.subheader("Bạn là ai?")
    role = st.selectbox("Đối tượng", list(ROLES), key="role")
    program = st.selectbox("Chương trình đang học", ["Tất cả"] + list(PROGRAMS), key="program")
    st.caption("Chọn đúng để tránh nhận quy định dành cho đối tượng khác.")

    if st.button("Xoá hội thoại", icon=":material/delete:"):
        st.session_state.messages = []
        st.rerun()

st.session_state.setdefault("messages", [])

metadata_filter = {"audience": ROLES[role]}
if program != "Tất cả":
    metadata_filter["program"] = PROGRAMS[program]

if embedder_is_mock():
    st.warning(
        "Đang dùng mô hình nhúng giả lập nên thứ tự kết quả là ngẫu nhiên. "
        "Đặt EMBEDDING_PROVIDER=local (hoặc gemini) trong .env để tra cứu đúng.",
        icon=":material/science:",
    )

if not llm_is_real():
    st.warning(
        "Chưa cấu hình LLM nên phần trả lời chỉ là thông báo. "
        "Các điều khoản trích dẫn bên dưới vẫn là nội dung thật từ quy chế.",
        icon=":material/key_off:",
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f'Căn cứ ({len(message["sources"])} điều khoản)'):
                for index, source in enumerate(message["sources"], 1):
                    st.markdown(f'**[{index}] {source["title"]}**')
                    st.markdown(source["content"])
                    st.caption(f'{source["url"]} · phiên bản {source["version"]}')

prompt = None
if not st.session_state.messages:
    picked = st.pills("Câu hỏi gợi ý", list(SUGGESTIONS), label_visibility="collapsed")
    if picked:
        prompt = SUGGESTIONS[picked]

prompt = st.chat_input("Hỏi về quy chế đào tạo…", submit_mode="disable") or prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    store, _docs = get_store(DEFAULT_STRATEGY, DEFAULT_SIZE)

    with st.chat_message("assistant"):
        with st.status(":shimmer[Đang tra quy chế]", type="compact") as status:
            with st.status("Tìm điều khoản liên quan", type="step"):
                results, _prompt_text, answer, error = answer_question(
                    store, prompt, top_k=3, metadata_filter=metadata_filter
                )
                st.write(f"Lọc theo: {metadata_filter}")
                st.write(f"Tìm được {len(results)} đoạn.")
            status.update(label=f"Đã tra {len(results)} điều khoản", state="complete")

        if error:
            st.error(f"Không gọi được LLM: {error}", icon=":material/cloud_off:")
            answer = "Không sinh được câu trả lời, nhưng các điều khoản liên quan vẫn ở bên dưới."

        if not results:
            answer = (
                "Không tìm thấy điều khoản nào phù hợp với đối tượng và chương trình bạn chọn. "
                "Thử đổi lựa chọn ở thanh bên, hoặc hỏi theo cách khác."
            )
            sources = []
            st.markdown(answer)
        else:
            sources = [
                {
                    "title": r["metadata"].get("title", r["metadata"]["doc_id"]),
                    "content": r["content"],
                    "url": r["metadata"].get("source_url", "?"),
                    "version": r["metadata"].get("document_version", "?"),
                }
                for r in results
            ]
            st.markdown(answer)
            with st.expander(f"Căn cứ ({len(sources)} điều khoản)"):
                for index, source in enumerate(sources, 1):
                    st.markdown(f'**[{index}] {source["title"]}**')
                    st.markdown(source["content"])
                    st.caption(f'{source["url"]} · phiên bản {source["version"]}')

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
