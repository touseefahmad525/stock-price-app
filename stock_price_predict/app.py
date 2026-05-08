import streamlit as st
import yfinance as yf

from data.fetch_data import get_stock_data
from utils.preprocess import prepare_data
from model.train_models import train_models
from model.load_model import load_model

st.title("📊 Stock Price Predictor App")

# Input stock symbol
stock = st.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, MSFT)")

if st.button("Predict"):

    if stock:

        # Step 1: Fetch data
        with st.spinner("Fetching data..."):
            data = get_stock_data(stock)
        
        if data.empty:
            st.error("Invalid stock symbol")
            st.stop()

        # Step 2: Preprocess
        X, y = prepare_data(data)

        # Step 3: Train models (Linear + Random Forest)
        lr_model, rf_model, dt_model,  lr_error, rf_error, dt_error = train_models(X, y)

        # Show comparison
        st.subheader("📊 Model Comparison")
        st.write(f"Linear Regression Error: {lr_error:.2f}")
        st.write(f"Random Forest Error: {rf_error:.2f}")
        st.write(f"Decision Tree Error: {dt_error:.2f}")

        # Step 4: Compare all models and pick best one
        errors = {
            "Linear Regression": lr_error,
            "Random Forest": rf_error,
            "Decision Tree": dt_error
        }

        # find model with minimum error
        best_model_name = min(errors, key=errors.get)

        # assign best model
        if best_model_name == "Linear Regression":
            best_model = lr_model
        elif best_model_name == "Random Forest":
            best_model = rf_model
        else:
            best_model = dt_model

        st.success(f"✅ {best_model_name} is the best model")

        # Step 5: Prediction
        latest_data = X.tail(1)
        prediction = best_model.predict(latest_data)

        # Show result
        st.subheader("🔮 Prediction Result")
        st.write(f"Predicted Close Price: {prediction[0][0]:.2f}")

        st.subheader("💰 Current Price")
        st.write(data["Close"].iloc[-1])

        # Show chart
        st.subheader("📈 Stock Close Price Chart")
        st.line_chart(data["Close"])

    else:
        st.warning("Please enter a stock symbol")