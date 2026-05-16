import streamlit as st

from pages.charts_analysis import render_chart_content
from pages.prediction import render_prediction_content
from utils.app_helpers import build_stock_analysis


def render_stock_search(current_analysis):
    if st.session_state.pop("clear_stock_input", False):
        st.session_state["stock_input"] = ""

    if "stock_input" not in st.session_state:
        st.session_state["stock_input"] = ""

    with st.form("stock_search_form"):
        st.text_input(
            "Enter Stock Symbol",
            key="stock_input",
            placeholder="AAPL, TSLA, MSFT",
        )
        analyze = st.form_submit_button("Analyze Stock")

    if not analyze:
        return current_analysis

    st.session_state["has_analyzed"] = True
    stock = st.session_state.get("stock_input", "").strip().upper()

    if not stock:
        st.warning("Please enter a stock symbol")
        st.stop()

    analysis = None
    error_message = None
    st.session_state["analysis"] = None

    with st.spinner("Fetching market data and training models..."):
        try:
            analysis = build_stock_analysis(stock)
        except ValueError as error:
            error_message = str(error)
        except Exception as error:
            error_message = f"Could not analyze this symbol: {error}"

    if error_message:
        st.session_state["analysis"] = None
        st.error(error_message)
        return None

    st.session_state["analysis"] = analysis
    st.session_state["stock_symbol"] = stock
    st.session_state["clear_stock_input"] = True
    st.rerun()


def render(analysis):
    analysis = render_stock_search(analysis)

    if analysis is None:
        if not st.session_state.get("has_analyzed", False):
            st.info("Enter a stock symbol and click Analyze Stock.")
        return

    errors = analysis["errors"]
    best_model_name = analysis["best_model_name"]

    st.subheader("Market Snapshot")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Current Price", f"${analysis['last_price']:.2f}")
    metric_cols[1].metric("Predicted Close", f"${analysis['prediction']:.2f}")
    metric_cols[2].metric("Best MSE", f"{errors[best_model_name]:.2f}")
    metric_cols[3].metric("Active Symbol", analysis["stock"])

    st.divider()
    render_prediction_content(analysis)

    st.divider()
    render_chart_content(analysis)
