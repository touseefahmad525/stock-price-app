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
        lr_model, rf_model, lr_error, rf_error = train_models(X, y)

        # Show comparison
        st.subheader("📊 Model Comparison")
        st.write(f"Linear Regression Error: {lr_error:.2f}")
        st.write(f"Random Forest Error: {rf_error:.2f}")

        # Step 4: Choose best model
        if rf_error < lr_error:
            best_model = rf_model
            st.success("✅ Random Forest is better")
        else:
            best_model = lr_model
            st.success("✅ Linear Regression is better")

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