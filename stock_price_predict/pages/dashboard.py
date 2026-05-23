import streamlit as st

from pages.charts_analysis import render_chart_content
from pages.prediction import render_prediction_content
from utils.app_helpers import build_stock_analysis
from utils.news_api import get_stock_news
from utils.sentiment import analyze_sentiment
from utils.recommendation import generate_recommendation


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

    # ✅ ONLY ONE CALL (FIXED)
    render_chart_content(analysis)

    # -----------------------------
    # 📰 NEWS SENTIMENT ANALYSIS
    # -----------------------------
    st.divider()
    st.subheader("📰 News Sentiment Analysis")

    stock = st.session_state.get("stock_symbol", analysis["stock"])

    try:
        news_list = get_stock_news(stock)

        if not news_list:
            st.info("No news found for this stock.")
        else:
            sentiment = analyze_sentiment(news_list)

            st.write("### Latest News")

            for item in sentiment["details"]:
                st.write(f"• {item['news']} → {item['sentiment']}")

            st.write("### Sentiment Summary")

            col1, col2, col3 = st.columns(3)

            col1.metric("👍 Positive", sentiment["positive"])
            col2.metric("👎 Negative", sentiment["negative"])
            col3.metric("😐 Neutral", sentiment["neutral"])

            # -----------------------------
            # 🤖 AI Recommendation Engine
            # -----------------------------
            recommendation_data = generate_recommendation(
                analysis["last_price"],
                analysis["prediction"],
                sentiment,
                analysis["confidence"]
            )

            recommendation = recommendation_data["recommendation"]
            score = recommendation_data["score_10"]

            st.write("### 🤖 AI Recommendation")

            if "Buy" in recommendation:
                st.success(f"{recommendation} | AI Score: {score}/10")

            elif "Sell" in recommendation:
                st.error(f"{recommendation} | AI Score: {score}/10")

            else:
                st.warning(f"{recommendation} | AI Score: {score}/10")

    except Exception as e:
        st.warning(f"Sentiment analysis failed: {e}")
