"""Giao diện Lab 7 — truy xuất quy chế đào tạo NEU.

    streamlit run streamlit_app.py

Hai trang: "Hỏi đáp" cho người dùng cuối, "Phòng thí nghiệm" cho phần đo đạc của
bài lab. Cả hai dùng chung corpus, mô hình nhúng và LLM qua app_core.py.
"""

import streamlit as st

st.set_page_config(page_title="Quy chế đào tạo NEU", page_icon=":material/gavel:", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/hoi_dap.py", title="Hỏi đáp", icon=":material/chat:", default=True),
        st.Page("app_pages/phong_lab.py", title="Phòng thí nghiệm", icon=":material/science:"),
    ]
)

page.run()
