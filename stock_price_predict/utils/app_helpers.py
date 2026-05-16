import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.fetch_data import get_stock_data
from model.train_models import train_models
from utils.future_predict import predict_future_prices
from utils.preprocess import prepare_data


PLOTLY_CONFIG = {
    "displayModeBar": True,
    "scrollZoom": True,
    "responsive": True,
}


def to_float(value, default=0.0):
    """
    Convert Pandas, NumPy, and sklearn outputs into a plain Python float.
    """
    if value is None:
        return default

    if isinstance(value, (pd.DataFrame, pd.Series)):
        value = value.squeeze()

    if isinstance(value, np.ndarray):
        value = value.squeeze()
        if value.ndim == 0:
            value = value.item()

    if isinstance(value, (pd.Series, pd.Index, list, tuple, np.ndarray)):
        if len(value) == 0:
            return default
        value = np.asarray(value, dtype="float64").reshape(-1)[0]

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return default

    if not np.isfinite(numeric_value):
        return default

    return numeric_value


def normalize_stock_data(data, stock):
    """
    Flatten yfinance output so columns are always Open/High/Low/Close/Volume.
    """
    data = data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        ticker = stock.upper().strip()

        if ticker in data.columns.get_level_values(-1):
            data = data.xs(ticker, axis=1, level=-1, drop_level=True)
        else:
            data.columns = [
                next((part for part in col if part), str(col))
                for col in data.columns.to_flat_index()
            ]

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    data = data.loc[:, [col for col in required_columns if col in data.columns]]

    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data.dropna()


def percentage_change(future_price, current_price):
    current_price = to_float(current_price)
    future_price = to_float(future_price)

    if current_price == 0:
        return 0.0

    return to_float(((future_price - current_price) / current_price) * 100)


def calculate_confidence(mse):
    """
    Convert model MSE into a 0-100 confidence score.
    Lower MSE produces higher confidence.
    """
    mse = max(to_float(mse), 0.0)
    confidence = 100 / (1 + mse)
    return max(0.0, min(to_float(confidence), 100.0))


def calculate_horizon_confidences(base_confidence):
    """
    Scale base model confidence by forecast horizon.
    Longer horizons receive a larger penalty.
    """
    base_confidence = max(0.0, min(to_float(base_confidence), 100.0))
    horizon_scales = {
        "7": 1.0,
        "14": 0.85,
        "30": 0.70,
    }

    return {
        horizon: max(0.0, min(base_confidence * scale, 100.0))
        for horizon, scale in horizon_scales.items()
    }


def build_stock_analysis(stock):
    data = get_stock_data(stock)

    if data.empty:
        raise ValueError("Invalid stock symbol")

    data = normalize_stock_data(data, stock)

    if data.empty or "Close" not in data.columns:
        raise ValueError("Could not load valid OHLCV data for this symbol")

    X, y = prepare_data(data)

    if X.empty or y.empty:
        raise ValueError("Not enough valid data to train a model")

    lr_model, rf_model, dt_model, lr_error, rf_error, dt_error = train_models(X, y)

    errors = {
        "Linear Regression": to_float(lr_error),
        "Random Forest": to_float(rf_error),
        "Decision Tree": to_float(dt_error),
    }
    models = {
        "Linear Regression": lr_model,
        "Random Forest": rf_model,
        "Decision Tree": dt_model,
    }

    best_model_name = min(errors, key=errors.get)
    best_model = models[best_model_name]
    latest_data = X.tail(1)
    prediction = to_float(best_model.predict(latest_data))
    last_price = to_float(data["Close"].iloc[-1])

    future = predict_future_prices(last_price, data)
    future = {
        "7": to_float(future.get("7")),
        "14": to_float(future.get("14")),
        "30": to_float(future.get("30")),
    }
    base_confidence = calculate_confidence(errors[best_model_name])
    confidence = calculate_horizon_confidences(base_confidence)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    y_pred = best_model.predict(X_test)

    return {
        "stock": stock.upper().strip(),
        "data": data,
        "X": X,
        "y": y,
        "models": models,
        "errors": errors,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "last_price": last_price,
        "prediction": prediction,
        "future": future,
        "base_confidence": base_confidence,
        "confidence": confidence,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }
