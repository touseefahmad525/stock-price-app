from html import escape
from textwrap import dedent

import streamlit as st

from pages.charts_analysis import render_chart_content
from pages.prediction import render_prediction_content
from utils.app_helpers import build_stock_analysis
from utils.news_api import get_stock_news
from utils.sentiment import analyze_sentiment
from utils.recommendation import generate_recommendation


SENTIMENT_THEME = {
    "Positive": {
        "class": "positive",
        "label": "Positive",
        "accent": "#22c55e",
    },
    "Negative": {
        "class": "negative",
        "label": "Negative",
        "accent": "#ef4444",
    },
    "Neutral": {
        "class": "neutral",
        "label": "Neutral",
        "accent": "#94a3b8",
    },
}


RECOMMENDATION_THEME = {
    "buy": {
        "class": "buy",
        "label": "Buy",
        "accent": "#22c55e",
    },
    "sell": {
        "class": "sell",
        "label": "Sell",
        "accent": "#ef4444",
    },
    "hold": {
        "class": "hold",
        "label": "Hold",
        "accent": "#f59e0b",
    },
}


def inject_news_sentiment_styles():
    st.markdown(
        """
        <style>
        .section-kicker {
            margin: 0 0 8px;
            color: #cbd5e1;
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: 0;
        }

        .news-card,
        .sentiment-tile,
        .recommendation-card {
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 8px;
            background: #111827;
            box-shadow: 0 10px 28px rgba(2, 6, 23, 0.24);
        }

        .news-card {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 14px;
            padding: 14px 16px;
            margin-bottom: 10px;
        }

        .news-title {
            color: #e5e7eb;
            font-size: 0.96rem;
            line-height: 1.45;
            margin: 0;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 78px;
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 0.76rem;
            font-weight: 800;
            line-height: 1;
            white-space: nowrap;
        }

        .badge-positive,
        .badge-buy {
            color: #bbf7d0;
            background: rgba(34, 197, 94, 0.16);
            border: 1px solid rgba(34, 197, 94, 0.38);
        }

        .badge-negative,
        .badge-sell {
            color: #fecaca;
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.38);
        }

        .badge-neutral {
            color: #e2e8f0;
            background: rgba(148, 163, 184, 0.15);
            border: 1px solid rgba(148, 163, 184, 0.34);
        }

        .badge-hold {
            color: #fde68a;
            background: rgba(245, 158, 11, 0.16);
            border: 1px solid rgba(245, 158, 11, 0.38);
        }

        .sentiment-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 18px;
        }

        .sentiment-tile {
            padding: 14px 16px;
            border-left: 4px solid var(--accent);
        }

        .sentiment-label {
            margin-bottom: 8px;
        }

        .sentiment-count {
            color: #f8fafc;
            font-size: 1.75rem;
            font-weight: 800;
            line-height: 1;
        }

        .recommendation-card {
            border-left: 5px solid var(--accent);
            padding: 18px 20px;
        }

        .recommendation-topline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 14px;
        }

        .recommendation-title {
            color: #f8fafc;
            font-size: 1.15rem;
            font-weight: 800;
            margin: 0;
        }

        .score-row {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            color: #cbd5e1;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .score-value {
            color: #f8fafc;
            font-size: 1.35rem;
            font-weight: 900;
        }

        .score-track {
            height: 9px;
            overflow: hidden;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.18);
        }

        .score-fill {
            height: 100%;
            width: var(--score-width);
            border-radius: 999px;
            background: var(--accent);
        }

        @media (max-width: 760px) {
            .news-card,
            .recommendation-topline {
                align-items: flex-start;
                flex-direction: column;
            }

            .sentiment-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_recommendation_action(recommendation):
    recommendation = str(recommendation).lower()
    for action in RECOMMENDATION_THEME:
        if action in recommendation:
            return action
    return "hold"


def render_badge(label, badge_class):
    return f'<span class="badge badge-{badge_class}">{escape(label)}</span>'


def render_latest_news(sentiment_details):
    st.markdown('<p class="section-kicker">Latest News</p>', unsafe_allow_html=True)

    for item in sentiment_details:
        theme = SENTIMENT_THEME.get(item["sentiment"], SENTIMENT_THEME["Neutral"])
        score = item.get("score", 0)
        st.markdown(
            dedent(f"""
            <div class="news-card">
                <p class="news-title">{escape(item["news"])}</p>
                <div>
                    {render_badge(theme["label"], theme["class"])}
                    <div style="color:#64748b;font-size:0.72rem;font-weight:700;margin-top:7px;text-align:center;">
                        {score:+.2f}
                    </div>
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )


def render_sentiment_summary(sentiment):
    st.markdown('<p class="section-kicker">Sentiment Summary</p>', unsafe_allow_html=True)

    columns = st.columns(3)
    for column, sentiment_name in zip(columns, ("Positive", "Negative", "Neutral")):
        theme = SENTIMENT_THEME[sentiment_name]
        count = sentiment.get(sentiment_name.lower(), 0)
        tile_html = (
            f'<div class="sentiment-tile" style="--accent:{theme["accent"]};">'
            f'<div class="sentiment-label">{render_badge(theme["label"], theme["class"])}</div>'
            f'<div class="sentiment-count">{count}</div>'
            '</div>'
        )
        column.markdown(tile_html, unsafe_allow_html=True)


def render_ai_recommendation(recommendation_data):
    recommendation = recommendation_data["recommendation"]
    score = recommendation_data["score_10"]
    action = get_recommendation_action(recommendation)
    theme = RECOMMENDATION_THEME[action]
    score_width = max(0, min(score * 10, 100))

    st.subheader("AI Recommendation")
    st.markdown(
        dedent(f"""
        <div class="recommendation-card" style="--accent:{theme['accent']};--score-width:{score_width}%;">
            <div class="recommendation-topline">
                <p class="recommendation-title">{escape(theme["label"])} signal</p>
                {render_badge(theme["label"], theme["class"])}
            </div>
            <div class="score-row">
                <span>AI Score</span>
                <span class="score-value">{score}/10</span>
            </div>
            <div class="score-track">
                <div class="score-fill"></div>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )


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

    st.divider()
    st.subheader("News Sentiment Analysis")
    inject_news_sentiment_styles()

    stock = st.session_state.get("stock_symbol", analysis["stock"])

    try:
        news_list = get_stock_news(stock)

        if not news_list:
            st.info("No news found for this stock.")
        else:
            sentiment = analyze_sentiment(news_list)

            render_latest_news(sentiment["details"])
            render_sentiment_summary(sentiment)

            recommendation_data = generate_recommendation(
                analysis["last_price"],
                analysis["prediction"],
                sentiment,
                analysis["confidence"],
            )
            render_ai_recommendation(recommendation_data)

    except Exception as e:
        st.warning(f"Sentiment analysis failed: {e}")
