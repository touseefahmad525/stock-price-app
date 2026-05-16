from pathlib import Path

import streamlit as st


LOGO_PATH = Path(__file__).resolve().parents[1] / "assets" / "images" / "Logo.png"


PAGES = {
    "📈 Dashboard": "dashboard",
    "🧠 Model Evaluation": "model_comparison",
}


def render_sidebar():
    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = "dashboard"

    with st.sidebar:
        st.markdown(
            """
            <style>
            section[data-testid="stSidebar"] {
                background: #111827;
            }

            section[data-testid="stSidebar"] .stImage {
                margin-bottom: 12px;
            }

            section[data-testid="stSidebar"] .stButton {
                margin-bottom: 4px;
            }

            section[data-testid="stSidebar"] .stButton > button {
                width: 100%;
                justify-content: flex-start;
                border: 0;
                border-radius: 8px;
                padding: 9px 12px;
                background: transparent;
                color: #d1d5db;
                font-size: 18px;
                font-weight: 700;
                transition: background 0.18s ease, color 0.18s ease;
            }

            section[data-testid="stSidebar"] .stButton > button:hover {
                background: #1f2937;
                color: #ffffff;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.image(str(LOGO_PATH), use_container_width=True)

        for page_label, page_key in PAGES.items():
            if st.button(page_label, key=f"sidebar_{page_key}"):
                st.session_state["selected_page"] = page_key

        st.divider()
        st.caption("Analyze a stock from the dashboard, then view it across all pages.")

    return st.session_state["selected_page"]
