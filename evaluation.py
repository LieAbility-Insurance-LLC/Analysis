import logging
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Prints evaluation metrics for a classification model:
     - Confusion matrix
     - Classification report
     - ROC AUC
    """
    logging.info(f"--- {model_name} Evaluation ---")
    cm = confusion_matrix(y_true, y_pred)
    logging.info(f"Confusion Matrix:\n{cm}")
    cr = classification_report(y_true, y_pred, digits=4)
    logging.info(f"Classification Report:\n{cr}")

    roc_auc = 0
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except ValueError as e:
        logging.warning(f"Cannot compute ROC-AUC for {model_name}: {e}")
    logging.info(f"{model_name} ROC-AUC: {roc_auc:.4f}")


def explain_model_shap(model, X_test):
    """
    Generates SHAP explanation plots for a given model and test set.
    """
    import shap
    import matplotlib.pyplot as plt
    if not hasattr(shap, "TreeExplainer"):
        logging.warning("SHAP is not available or not imported. Skipping.")
        return

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # For multi-output (binary classification), use the 0-index
        if isinstance(shap_values, list):
            shap_val_to_plot = shap_values[0]
            expected_value = explainer.expected_value[0]
        else:
            shap_val_to_plot = shap_values
            expected_value = explainer.expected_value

        # Create a summary bar plot of feature importances
        shap.summary_plot(shap_val_to_plot, X_test, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (Bar)")
        plt.show()

        # Generate force plot for a single prediction using the new API
        if len(X_test) > 0:
            sample_index = 0
            shap.plots.force(expected_value,
                             shap_val_to_plot[sample_index, :],
                             X_test.iloc[sample_index, :])
    except Exception as e:
        logging.error(f"Error generating SHAP plots: {e}")