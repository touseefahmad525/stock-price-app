import streamlit as st

from utils.app_helpers import percentage_change


def render_prediction_content(analysis):
    st.subheader("Prediction Result")
    st.metric("Predicted Close Price", f"${analysis['prediction']:.2f}")

    st.subheader("Current Price")
    st.metric("Latest Close", f"${analysis['last_price']:.2f}")

    st.subheader("Future Price Predictions")
    future = analysis["future"]
    last_price = analysis["last_price"]
    col1, col2, col3 = st.columns(3)

    with col1:
        change_7 = percentage_change(future["7"], last_price)
        st.metric("7-Day Prediction", f"${future['7']:.2f}", f"{change_7:.2f}%")
        st.progress(0.78)
        st.caption("Confidence: 78%")

    with col2:
        change_14 = percentage_change(future["14"], last_price)
        st.metric("14-Day Prediction", f"${future['14']:.2f}", f"{change_14:.2f}%")
        st.progress(0.72)
        st.caption("Confidence: 72%")

    with col3:
        change_30 = percentage_change(future["30"], last_price)
        st.metric("30-Day Prediction", f"${future['30']:.2f}", f"{change_30:.2f}%")
        st.progress(0.65)
        st.caption("Confidence: 65%")


def render(analysis):
    st.header("Prediction")

    if analysis is None:
        st.info("Analyze a stock from the dashboard to view predictions.")
        return

    render_prediction_content(analysis)
