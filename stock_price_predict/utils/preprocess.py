def prepare_data(data):
    """
    Convert raw stock data into ML-ready format
    """

    # Remove missing values (safety step)
    data = data.dropna()

    # Features (input)
    X = data[['Open', 'High', 'Low', 'Volume']]

    # Target (output)
    y = data['Close']

    return X, y