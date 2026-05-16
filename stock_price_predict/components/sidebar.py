import streamlit as st


PAGES = {
    "Dashboard": "dashboard",
    "Model Comparison": "model_comparison",
}


def render_sidebar():
    with st.sidebar:
        st.title("Stock App")
        selected_page = st.radio("Navigation", list(PAGES.keys()))

        st.divider()
        st.caption("Analyze a stock from the dashboard, then view it across all pages.")

    return PAGES[selected_page]
