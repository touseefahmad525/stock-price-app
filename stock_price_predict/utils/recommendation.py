def _clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_confidence(confidence):
    if isinstance(confidence, dict):
        confidence_values = [
            _to_float(value)
            for value in confidence.values()
            if value is not None
        ]
        if not confidence_values:
            return 0.0
        return sum(confidence_values) / len(confidence_values)

    return _to_float(confidence)


def generate_recommendation(
    current_price,
    predicted_price,
    sentiment,
    confidence,
):
    current_price = _to_float(current_price)
    predicted_price = _to_float(predicted_price)
    confidence = _clamp(_normalize_confidence(confidence), 0.0, 100.0)

    positive = int(sentiment.get("positive", 0))
    negative = int(sentiment.get("negative", 0))
    neutral = int(sentiment.get("neutral", 0))
    total_sentiment = positive + negative + neutral

    if current_price:
        trend_change = ((predicted_price - current_price) / current_price) * 100
    else:
        trend_change = 0.0

    trend_signal = _clamp(trend_change / 5, -1.0, 1.0)
    sentiment_signal = (
        (positive - negative) / total_sentiment
        if total_sentiment
        else 0.0
    )
    confidence_signal = (confidence - 50) / 50

    trend_score = trend_signal * 50
    sentiment_score = sentiment_signal * 30
    confidence_score = confidence_signal * 20
    raw_score = trend_score + sentiment_score + confidence_score
    score_10 = round(_clamp((raw_score + 100) / 20, 0.0, 10.0), 2)

    if score_10 >= 7:
        recommendation = "Buy 📈"
    elif score_10 <= 3:
        recommendation = "Sell 📉"
    else:
        recommendation = "Hold 🤝"

    return {
        "recommendation": recommendation,
        "score_10": score_10,
        "details": {
            "trend_score": round(trend_score, 2),
            "sentiment_score": round(sentiment_score, 2),
            "confidence_score": round(confidence_score, 2),
            "raw_score": round(raw_score, 2),
            "trend_change_percent": round(trend_change, 2),
            "confidence": round(confidence, 2),
        },
    }
