import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# -----------------------------
#  Auto download (IMPORTANT)
# -----------------------------
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()


def analyze_sentiment(news_list):
    results = {
        "positive": 0,
        "negative": 0,
        "neutral": 0,
        "details": [],
    }

    for news in news_list:
        news = str(news).strip()
        if not news:
            continue

        score = sia.polarity_scores(news)["compound"]

        if score >= 0.05:
            sentiment = "Positive"
            results["positive"] += 1
        elif score <= -0.05:
            sentiment = "Negative"
            results["negative"] += 1
        else:
            sentiment = "Neutral"
            results["neutral"] += 1

        results["details"].append(
            {
                "news": news,
                "sentiment": sentiment,
                "score": score,
            }
        )

    return results
