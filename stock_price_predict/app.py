import streamlit as st
import yfinance as yf

from data.fetch_data import get_stock_data
from utils.preprocess import prepare_data
from model.train_model import train_model
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

        # Step 3: Load or Train model (THIS PART 👇)
        try:
            model = load_model()
        except:
            model = train_model(X, y)

        # Step 4: Make prediction (latest row)
        latest_data = X.tail(1)
        prediction = model.predict(latest_data)

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