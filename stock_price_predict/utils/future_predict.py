import numpy as np

def predict_future_prices(last_price):
    """
    Simulate future stock predictions (7, 14, 30 days)
    """

    np.random.seed
    last_price = float(np.asarray(last_price).squeeze())

    # small random growth simulation
    pred_7 = last_price * (1 + np.random.uniform(0.01, 0.03))
    pred_14 = last_price * (1 + np.random.uniform(0.02, 0.05))
    pred_30 = last_price * (1 + np.random.uniform(0.03, 0.08))

    return {
        "7": float(round(pred_7, 2)),
        "14": float(round(pred_14, 2)),
        "30": float(round(pred_30, 2))
    }