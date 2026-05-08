import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import auc, confusion_matrix, roc_curve


PLOTLY_DARK_TEMPLATE = "plotly_dark"
BACKGROUND_COLOR = "#0f172a"
PAPER_COLOR = "#111827"
GRID_COLOR = "#243044"
TEXT_COLOR = "#e5e7eb"
GREEN = "#22c55e"
RED = "#ef4444"
BLUE = "#38bdf8"
PURPLE = "#a78bfa"


def _dark_layout(fig, title, height=420):
    fig.update_layout(
        title=title,
        height=height,
        template=PLOTLY_DARK_TEMPLATE,
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font={"color": TEXT_COLOR},
        margin={"l": 24, "r": 24, "t": 64, "b": 32},
        hovermode="x unified",
        dragmode="zoom",
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig


def plot_candlestick_with_volume(data, title="Stock Price"):
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    chart_data = data.copy()
    chart_data = chart_data.dropna(subset=required_columns)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
        specs=[[{"type": "candlestick"}], [{"type": "bar"}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=chart_data.index,
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"],
            name="OHLC",
            increasing={"line": {"color": GREEN}, "fillcolor": GREEN},
            decreasing={"line": {"color": RED}, "fillcolor": RED},
            hoverlabel={"namelength": -1},
        ),
        row=1,
        col=1,
    )

    volume_colors = np.where(chart_data["Close"] >= chart_data["Open"], GREEN, RED)
    fig.add_trace(
        go.Bar(
            x=chart_data.index,
            y=chart_data["Volume"],
            name="Volume",
            marker={"color": volume_colors, "opacity": 0.65},
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Volume: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    _dark_layout(fig, title, height=720)
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_yaxes(title_text="Price", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(
        rangeselector={
            "buttons": [
                {"count": 7, "label": "7D", "step": "day", "stepmode": "backward"},
                {"count": 14, "label": "14D", "step": "day", "stepmode": "backward"},
                {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
                {"step": "all", "label": "All"},
            ],
            "bgcolor": "#1f2937",
            "activecolor": "#334155",
        },
        row=1,
        col=1,
    )
    return fig


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    if labels is None:
        labels = list(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)])))

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text}",
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    _dark_layout(fig, title, height=480)
    fig.update_xaxes(title="Predicted Label")
    fig.update_yaxes(title="Actual Label", autorange="reversed")
    return fig


def plot_roc_curve(y_true, y_score, title="ROC Curve"):
    y_score = np.asarray(y_score)
    if y_score.ndim == 2 and y_score.shape[1] > 1:
        y_score = y_score[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC AUC = {auc_score:.3f}",
            line={"color": BLUE, "width": 3},
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line={"color": "#64748b", "dash": "dash"},
            hoverinfo="skip",
        )
    )
    _dark_layout(fig, f"{title} - AUC {auc_score:.3f}", height=480)
    fig.update_xaxes(title="False Positive Rate", range=[0, 1])
    fig.update_yaxes(title="True Positive Rate", range=[0, 1])
    return fig


def plot_feature_importance(model, feature_names, title="Feature Importance"):
    if not hasattr(model, "feature_importances_"):
        return None

    importances = pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance": np.asarray(model.feature_importances_, dtype=float),
        }
    )
    importances = importances.sort_values("importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=importances["importance"],
            y=importances["feature"],
            orientation="h",
            marker={"color": importances["importance"], "colorscale": "Viridis"},
            hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    _dark_layout(fig, title, height=max(360, 70 + len(importances) * 42))
    fig.update_xaxes(title="Importance")
    fig.update_yaxes(title="")
    return fig


def plot_clusters(X, labels, centroids=None, title="Cluster Visualization"):
    data = pd.DataFrame(X).copy()
    if data.shape[1] < 2:
        raise ValueError("Cluster visualization requires at least two feature columns")

    data["cluster"] = labels
    x_col = data.columns[0]
    y_col = data.columns[1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data[x_col],
            y=data[y_col],
            mode="markers",
            marker={"color": data["cluster"], "colorscale": "Turbo", "size": 9, "opacity": 0.85},
            text=data["cluster"],
            name="Clusters",
            hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Cluster: %{text}<extra></extra>",
        )
    )

    if centroids is not None:
        centroids = np.asarray(centroids)
        fig.add_trace(
            go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode="markers",
                marker={"color": "#facc15", "size": 16, "symbol": "x", "line": {"width": 2}},
                name="Centroids",
                hovertemplate="Centroid X: %{x:.3f}<br>Centroid Y: %{y:.3f}<extra></extra>",
            )
        )

    _dark_layout(fig, title, height=520)
    fig.update_xaxes(title=str(x_col))
    fig.update_yaxes(title=str(y_col))
    return fig


def plot_neural_network_history(history, title="Neural Network Training"):
    history_data = history.history if hasattr(history, "history") else history
    fig = go.Figure()

    for key, color in [("loss", RED), ("val_loss", "#fb7185"), ("accuracy", GREEN), ("val_accuracy", BLUE)]:
        if key in history_data:
            fig.add_trace(
                go.Scatter(
                    y=history_data[key],
                    mode="lines+markers",
                    name=key.replace("_", " ").title(),
                    line={"color": color, "width": 2.5},
                    hovertemplate="Epoch: %{x}<br>Value: %{y:.4f}<extra></extra>",
                )
            )

    _dark_layout(fig, title, height=500)
    fig.update_xaxes(title="Epoch")
    fig.update_yaxes(title="Metric Value")
    return fig


def plot_regression_predictions(y_true, y_pred, title="Actual vs Predicted"):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=y_true,
            mode="lines+markers",
            name="Actual",
            line={"color": BLUE, "width": 2},
            hovertemplate="Index: %{x}<br>Actual: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=y_pred,
            mode="lines+markers",
            name="Predicted",
            line={"color": PURPLE, "width": 2},
            hovertemplate="Index: %{x}<br>Predicted: $%{y:,.2f}<extra></extra>",
        )
    )
    _dark_layout(fig, title, height=460)
    fig.update_yaxes(title="Close Price", tickprefix="$")
    return fig
