import streamlit as st

from pages.charts_analysis import render_chart_content
from pages.prediction import render_prediction_content
from utils.app_helpers import build_stock_analysis


def render_stock_search(current_analysis):
    with st.form("stock_search_form"):
        stock = st.text_input(
            "Enter Stock Symbol",
            value=st.session_state.get("stock_symbol", ""),
            placeholder="AAPL, TSLA, MSFT",
        )
        analyze = st.form_submit_button("Analyze Stock")

    if not analyze:
        return current_analysis

    if not stock:
        st.warning("Please enter a stock symbol")
        st.stop()

    with st.spinner("Fetching market data and training models..."):
        try:
            analysis = build_stock_analysis(stock)
        except ValueError as error:
            st.error(str(error))
            st.stop()

    st.session_state["analysis"] = analysis
    st.session_state["stock_symbol"] = stock.upper().strip()
    return analysis


def render(analysis):
    st.header("Dashboard")
    analysis = render_stock_search(analysis)

    if analysis is None:
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
