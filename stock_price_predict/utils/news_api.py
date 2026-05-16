import requests
import os
from dotenv import load_dotenv

# load .env file
load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")


def get_stock_news(symbol):

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": symbol,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        news_list = []

        if data.get("status") == "ok":
            for article in data.get("articles", [])[:5]:
                title = article.get("title")
                if title:
                    news_list.append(title)

        return news_list

    except Exception as e:
        print("News API error:", e)
        return []