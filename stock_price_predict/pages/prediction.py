import streamlit as st

from utils.app_helpers import percentage_change


def render_prediction_metric(label, future_price, last_price, confidence):
    change = percentage_change(future_price, last_price)

    st.metric(label, f"${future_price:.2f}", f"{change:.2f}%")
    st.progress(confidence / 100)
    st.caption(f"Confidence: {confidence:.2f}%")


def render_prediction_content(analysis):
    st.subheader("Future Price Predictions")
    future = analysis["future"]
    last_price = analysis["last_price"]
    confidence = analysis.get("confidence", {})
    col1, col2, col3 = st.columns(3)

    with col1:
        render_prediction_metric(
            "7-Day Prediction",
            future["7"],
            last_price,
            confidence.get("7", 0.0),
        )

    with col2:
        render_prediction_metric(
            "14-Day Prediction",
            future["14"],
            last_price,
            confidence.get("14", 0.0),
        )

    with col3:
        render_prediction_metric(
            "30-Day Prediction",
            future["30"],
            last_price,
            confidence.get("30", 0.0),
        )


def render(analysis):
    st.header("Prediction")

    if analysis is None:
        st.info("Analyze a stock from the dashboard to view predictions.")
        return

    render_prediction_content(analysis)
