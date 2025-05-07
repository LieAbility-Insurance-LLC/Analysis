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


# ──────────────────────────────────────────────────────────
# Extra visual diagnostics · masters edition
# ──────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE

def plot_pr_curve(y_true, y_score, model_name="Model"):
    """
    Precision–Recall curve + Average Precision.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve · {model_name}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


def plot_calibration_curve(model, X_test, y_test, n_bins=10, model_name="Model"):
    """
    Reliability diagram (calibration curve).
    """
    if not hasattr(model, "predict_proba"):
        return  # skip models without probabilistic output
    prob_pos = model.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, prob_pos, n_bins=n_bins)
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration Curve · {model_name}")
    plt.tight_layout()
    plt.show()


def plot_cumulative_gain(y_true, y_score, model_name="Model"):
    """
    Cumulative gain & lift: what % of frauds caught vs % of population reviewed.
    """
    # Sort by score descending
    sort_idx = np.argsort(-y_score)
    y_true_sorted = np.array(y_true)[sort_idx]

    cum_fraud = np.cumsum(y_true_sorted)
    total_fraud = cum_fraud[-1]
    perc_population = np.arange(1, len(y_true_sorted) + 1) / len(y_true_sorted)
    gain = cum_fraud / total_fraud

    plt.figure()
    plt.plot(perc_population, gain, label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("% of claims reviewed")
    plt.ylabel("% of frauds captured")
    plt.title(f"Cumulative Gain · {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_tsne_embedding(X, y, model_name="Data"):
    """
    2‑D t‑SNE embedding coloured by class label.
    """
    tsne = TSNE(n_components=2, init="pca", random_state=42, perplexity=30)
    X_emb = tsne.fit_transform(X)
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        X_emb[:, 0], X_emb[:, 1], c=y, cmap="coolwarm", alpha=0.6, s=12
    )
    plt.legend(*scatter.legend_elements(), title="Fraud")
    plt.title(f"t‑SNE projection · {model_name}")
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(df, top_features, target="fraud_reported"):
    """
    Violin plots of top numeric predictors split by target class.
    """
    import seaborn as sns

    n = len(top_features)
    cols = 2
    rows = (n + 1) // cols
    plt.figure(figsize=(cols * 5, rows * 3.5))
    for i, col in enumerate(top_features, 1):
        plt.subplot(rows, cols, i)
        sns.violinplot(x=target, y=col, data=df, inner="quartile")
        plt.title(col)
        plt.xlabel("")
    plt.tight_layout()
    plt.show()
