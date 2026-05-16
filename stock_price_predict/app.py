import streamlit as st

from components.sidebar import render_sidebar
from pages import dashboard, model_comparison


st.set_page_config(
    page_title="Stock ML Trading Dashboard",
    page_icon="📈",
    layout="wide",
)


PAGE_RENDERERS = {
    "dashboard": dashboard.render,
    "model_comparison": model_comparison.render,
}


def apply_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background: #0b1120;
            color: #e5e7eb;
        }
        div[data-testid="stMetricValue"] {
            color: #f8fafc;
        }
        div[data-testid="stSidebarNav"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    apply_theme()
    selected_page = render_sidebar()

    st.title("Stock ML Trading Dashboard")

    analysis = st.session_state.get("analysis")
    PAGE_RENDERERS[selected_page](analysis)


if __name__ == "__main__":
    main()
