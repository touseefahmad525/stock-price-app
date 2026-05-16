import streamlit as st

from utils.app_helpers import PLOTLY_CONFIG
from utils.visualizations import plot_candlestick_with_volume


def render(analysis):
    st.header("Charts / Analysis")

    if analysis is None:
        st.info("Analyze a stock from the dashboard to view charts.")
        return

    st.subheader("Trading View")
    st.plotly_chart(
        plot_candlestick_with_volume(
            analysis["data"],
            title=f"{analysis['stock']} OHLCV",
        ),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )
