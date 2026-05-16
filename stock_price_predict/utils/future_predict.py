import numpy as np


def predict_future_prices(last_price, data=None):
    """
    Estimate future prices for 7, 14, and 30 days from recent price movement.
    """
    last_price = float(np.asarray(last_price).squeeze())

    if data is None or "Close" not in data.columns:
        return {
            "7": float(round(last_price, 2)),
            "14": float(round(last_price, 2)),
            "30": float(round(last_price, 2)),
        }

    close_prices = data["Close"].astype(float).dropna()
    daily_returns = close_prices.pct_change().dropna()

    if daily_returns.empty:
        return {
            "7": float(round(last_price, 2)),
            "14": float(round(last_price, 2)),
            "30": float(round(last_price, 2)),
        }

    recent_returns = daily_returns.tail(30)
    average_daily_return = float(recent_returns.mean())

    def forecast_price(days):
        expected_return = np.clip(average_daily_return * days, -0.5, 0.5)
        return last_price * (1 + expected_return)

    return {
        "7": float(round(forecast_price(7), 2)),
        "14": float(round(forecast_price(14), 2)),
        "30": float(round(forecast_price(30), 2)),
    }
