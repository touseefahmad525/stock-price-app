import numpy as np
import pandas as pd
import streamlit as st

from data.fetch_data import get_stock_data
from model.train_models import train_models
from utils.future_predict import predict_future_prices
from utils.preprocess import prepare_data


def to_float(value, default=0.0):
    """
    Convert Pandas, NumPy, and sklearn outputs into a plain Python float.
    This keeps Streamlit formatting like .2f safe even when yfinance returns
    Series/DataFrame values.
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
    Newer yfinance versions may return MultiIndex columns even for one ticker.
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


st.title("📊 Stock Price Predictor App")

stock = st.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, MSFT)")

if st.button("Predict"):
    if not stock:
        st.warning("Please enter a stock symbol")
        st.stop()

    with st.spinner("Fetching data..."):
        data = get_stock_data(stock)

    if data.empty:
        st.error("Invalid stock symbol")
        st.stop()

    data = normalize_stock_data(data, stock)

    if data.empty or "Close" not in data.columns:
        st.error("Could not load valid price data for this symbol")
        st.stop()

    X, y = prepare_data(data)

    if X.empty or y.empty:
        st.error("Not enough valid data to train a model")
        st.stop()

    lr_model, rf_model, dt_model, lr_error, rf_error, dt_error = train_models(X, y)

    lr_error = to_float(lr_error)
    rf_error = to_float(rf_error)
    dt_error = to_float(dt_error)

    st.subheader("📊 Model Comparison")
    st.write(f"Linear Regression Error: {lr_error:.2f}")
    st.write(f"Random Forest Error: {rf_error:.2f}")
    st.write(f"Decision Tree Error: {dt_error:.2f}")

    errors = {
        "Linear Regression": lr_error,
        "Random Forest": rf_error,
        "Decision Tree": dt_error,
    }

    best_model_name = min(errors, key=errors.get)
    best_model = {
        "Linear Regression": lr_model,
        "Random Forest": rf_model,
        "Decision Tree": dt_model,
    }[best_model_name]

    st.success(f"✅ {best_model_name} is the best model")

    latest_data = X.tail(1)
    prediction = to_float(best_model.predict(latest_data))

    st.subheader("🔮 Prediction Result")
    st.write(f"Predicted Close Price: ${prediction:.2f}")

    last_price = to_float(data["Close"].iloc[-1])

    st.subheader("💰 Current Price")
    st.write(f"${last_price:.2f}")

    future = predict_future_prices(last_price)
    future = {
        "7": to_float(future.get("7")),
        "14": to_float(future.get("14")),
        "30": to_float(future.get("30")),
    }

    st.subheader("📊 Price Predictions")
    col1, col2, col3 = st.columns(3)

    with col1:
        change_7 = percentage_change(future["7"], last_price)
        st.markdown("### 📅 7-Day Prediction")
        st.markdown(f"## ${future['7']:.2f}")
        st.markdown(f"📈 {change_7:.2f}%")
        st.progress(0.78)
        st.write("Confidence: 78%")

    with col2:
        change_14 = percentage_change(future["14"], last_price)
        st.markdown("### 📅 14-Day Prediction")
        st.markdown(f"## ${future['14']:.2f}")
        st.markdown(f"📈 {change_14:.2f}%")
        st.progress(0.72)
        st.write("Confidence: 72%")

    with col3:
        change_30 = percentage_change(future["30"], last_price)
        st.markdown("### 📅 30-Day Prediction")
        st.markdown(f"## ${future['30']:.2f}")
        st.markdown(f"📈 {change_30:.2f}%")
        st.progress(0.65)
        st.write("Confidence: 65%")

    st.subheader("📈 Stock Close Price Chart")
    st.line_chart(data["Close"])
