import yfinance as yf

def get_stock_data(stock="AAPL", period="1mo"):
    """
    Fetch stock data from Yahoo Finance
    """
    data = yf.download(stock, period=period)
    return data