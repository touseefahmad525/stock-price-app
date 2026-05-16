import streamlit as st

from utils.app_helpers import PLOTLY_CONFIG
from utils.visualizations import plot_feature_importance, plot_regression_predictions


def render(analysis):
    st.header("Model Evaluation")

    if analysis is None:
        st.info("Analyze a stock from the dashboard to evaluate models.")
        return

    errors = analysis["errors"]

    st.subheader("ML Model Evaluation")
    eval_cols = st.columns(4)
    eval_cols[0].metric("Best Model", analysis["best_model_name"])
    eval_cols[1].metric("Linear Regression MSE", f"{errors['Linear Regression']:.2f}")
    eval_cols[2].metric("Random Forest MSE", f"{errors['Random Forest']:.2f}")
    eval_cols[3].metric("Decision Tree MSE", f"{errors['Decision Tree']:.2f}")

    st.plotly_chart(
        plot_regression_predictions(
            analysis["y_test"],
            analysis["y_pred"],
            title=f"{analysis['best_model_name']}: Actual vs Predicted Close",
        ),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    feature_col1, feature_col2 = st.columns(2)
    with feature_col1:
        rf_importance = plot_feature_importance(
            analysis["models"]["Random Forest"],
            analysis["X"].columns,
            title="Random Forest Feature Importance",
        )
        if rf_importance is not None:
            st.plotly_chart(rf_importance, use_container_width=True, config=PLOTLY_CONFIG)

    with feature_col2:
        dt_importance = plot_feature_importance(
            analysis["models"]["Decision Tree"],
            analysis["X"].columns,
            title="Decision Tree Feature Importance",
        )
        if dt_importance is not None:
            st.plotly_chart(dt_importance, use_container_width=True, config=PLOTLY_CONFIG)
