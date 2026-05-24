from pathlib import Path

import streamlit as st


LOGO_PATH = Path(__file__).resolve().parents[1] / "assets" / "images" / "Logo.png"


PAGES = {
    "Dashboard": {
        "key": "dashboard",
        "icon": "layout-dashboard",
    },
    "Model Evaluation": {
        "key": "model_comparison",
        "icon": "bar-chart-3",
    },
}


LUCIDE_ICONS = {
    "layout-dashboard": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/></svg>',
    "bar-chart-3": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>',
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
                margin-bottom: 8px;
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

            section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
                align-items: center;
            }

            section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
                margin: 0;
            }

            .sidebar-icon {
                color: #ffffff;
                height: 45px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-top: -8px;
                margin-bottom: 8px;
            }

            .sidebar-icon svg {
                display: block;
            }
            </style>

            """,
            unsafe_allow_html=True,
        )

        st.image(str(LOGO_PATH), use_container_width=True)

        for page_label, page_data in PAGES.items():
            icon_svg = LUCIDE_ICONS[page_data["icon"]]

            col1, col2 = st.columns([1, 5])

            with col1:
                st.markdown(
                    f'<div class="sidebar-icon">{icon_svg}</div>',
                    unsafe_allow_html=True,
                )

            with col2:
                if st.button(
                    page_label,
                    key=f"sidebar_{page_data['key']}",
                ):
                    st.session_state["selected_page"] = page_data["key"]

        st.divider()

        st.caption(
            "Analyze a stock from the dashboard, then view it across all pages."
        )

    return st.session_state["selected_page"]
