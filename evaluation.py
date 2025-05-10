from __future__ import annotations

import logging
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve

# ──────────────────────────────────────────────────────────
# Pretty defaults
# ──────────────────────────────────────────────────────────
sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.05,
    rc={
        "figure.dpi": 120,
        "axes.titleweight": "bold",
        "axes.labelweight": "semibold",
        "axes.edgecolor": "#BBBBBB",
    },
)
PALETTE = sns.color_palette("Set2")
sns.set_palette(PALETTE)


# ──────────────────────────────────────────────────────────
# Core metrics + classic plots
# ──────────────────────────────────────────────────────────

def _plot_confusion_heatmap(cm: np.ndarray, model_name: str) -> None:
    """Nicely annotated confusion-matrix heat-map."""
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        linecolor="#EEEEEE",
    )
    plt.title(f"Confusion Matrix · {model_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()


def evaluate_model(y_true, y_pred, model_name: str = "Model") -> None:
    """
    Print classic numeric metrics **and** auto-draw a confusion-matrix heat-map.
    Signature unchanged - down-stream calls stay intact.
    """
    logging.info(f"── {model_name} Evaluation ──")
    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion_heatmap(cm, model_name)

    cr = classification_report(y_true, y_pred, digits=4)
    logging.info(f"Classification Report:\n{cr}")

    try:
        roc_auc = roc_auc_score(y_true, y_pred)
        logging.info(f"{model_name} ROC-AUC ≈{roc_auc:.4f}")
    except ValueError as exc:
        logging.warning(f"ROC-AUC unavailable → {exc}")


# ──────────────────────────────────────────────────────────
# SHAP explainability (unchanged API, nicer defaults)
# ──────────────────────────────────────────────────────────

def explain_model_shap(model, X_test):
    """
    SHAP summary-bar + force plot for the first observation.
    """
    if not hasattr(shap, "TreeExplainer"):
        logging.warning("SHAP unavailable - skipping explainability.")
        return

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, Sequence):
            shap_val = shap_values[0]
            expected = explainer.expected_value[0]
        else:
            shap_val = shap_values
            expected = explainer.expected_value

        # Summary (beeswarm is prettier but slower – keep bar)
        shap.summary_plot(shap_val, X_test, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.show()

        # Force plot – static PNG
        if len(X_test) > 0:
            shap.plots.force(expected, shap_val[0, :], X_test.iloc[0, :])
    except Exception as exc:
        logging.error(f"SHAP plotting error: {exc}")


# ──────────────────────────────────────────────────────────
# ROC / PR / Calibration / Gain curves
# ──────────────────────────────────────────────────────────

def plot_pr_curve(y_true, y_score, model_name: str = "Model"):
    """Precision–Recall curve with shaded area = Average Precision."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure()
    plt.fill_between(recall, precision, alpha=0.15)
    plt.plot(recall, precision, linewidth=2, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve · {model_name}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


def plot_calibration_curve(
    model, X_test, y_test, n_bins: int = 10, model_name: str = "Model"
):
    """Reliability diagram with ± 1 σ error bars."""
    if not hasattr(model, "predict_proba"):
        return
    prob_pos = model.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(
        y_test, prob_pos, n_bins=n_bins, strategy="quantile"
    )

    plt.figure()
    plt.errorbar(
        mean_pred,
        frac_pos,
        yerr=np.sqrt(frac_pos * (1 - frac_pos) / len(y_test)),
        fmt="o",
        capsize=3,
    )
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration Curve · {model_name}")
    plt.tight_layout()
    plt.show()


def plot_cumulative_gain(y_true, y_score, model_name: str = "Model"):
    """Cumulative gain curve + shaded lift over random."""
    order = np.argsort(-y_score)
    y_true_sorted = np.array(y_true)[order]

    cum_fraud = np.cumsum(y_true_sorted)
    total_fraud = cum_fraud[-1]
    perc_population = np.arange(1, len(y_true_sorted) + 1) / len(y_true_sorted)
    gain = cum_fraud / total_fraud

    plt.figure()
    plt.fill_between(perc_population, gain, perc_population, alpha=0.15)
    plt.plot(perc_population, gain, linewidth=2, label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("% of population reviewed")
    plt.ylabel("% of frauds captured")
    plt.title(f"Cumulative Gain · {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────
# NEW diagnostics
# ──────────────────────────────────────────────────────────

def plot_lift_curve(y_true, y_score, model_name: str = "Model"):
    """Lift curve = model gain / random gain."""
    order = np.argsort(-y_score)
    y_true_sorted = np.array(y_true)[order]

    cum_fraud = np.cumsum(y_true_sorted)
    total_fraud = cum_fraud[-1]
    perc_population = np.arange(1, len(y_true_sorted) + 1) / len(y_true_sorted)
    gain = cum_fraud / total_fraud
    lift = gain / perc_population

    plt.figure()
    plt.plot(perc_population, lift, linewidth=2)
    plt.xlabel("% of population reviewed")
    plt.ylabel("Lift")
    plt.title(f"Lift Curve · {model_name}")
    plt.axhline(1, color="k", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_probability_distribution(y_true, y_score, model_name: str = "Model"):
    """Histogram / KDE of predicted probabilities broken out by class."""
    plt.figure(figsize=(6, 4))
    sns.kdeplot(y_score[y_true == 0], label="Not Fraud", fill=True, alpha=0.3)
    sns.kdeplot(y_score[y_true == 1], label="Fraud", fill=True, alpha=0.3)
    plt.xlabel("Predicted probability")
    plt.title(f"Probability Distribution · {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────
# t-SNE & feature-distribution plots
# ──────────────────────────────────────────────────────────
from sklearn.manifold import TSNE


def plot_tsne_embedding(X, y, model_name: str = "Data"):
    """2-D t-SNE embedding coloured by class."""
    tsne = TSNE(n_components=2, init="pca", random_state=42, perplexity=30)
    emb = tsne.fit_transform(X)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=y, cmap="coolwarm", s=14, alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Fraud", loc="best")
    plt.title(f"t-SNE Projection · {model_name}")
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(df, top_features, target: str = "fraud_reported"):
    """Violin plots of numeric predictors split by class label."""
    n = len(top_features)
    cols = 2
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(cols * 5.2, rows * 3.5))
    for i, col in enumerate(top_features, 1):
        plt.subplot(rows, cols, i)
        sns.violinplot(
            x=target,
            y=col,
            data=df,
            inner="quartile",
            linewidth=0.8,
            palette=PALETTE,
        )
        plt.title(col, fontsize=11)
        plt.xlabel("")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────
#  Cluster visualisation helpers
# ──────────────────────────────────────────────────────────

def plot_cluster_embedding(
    X,
    cluster_labels: Sequence[int],
    algorithm_name: str = "Clustering",
    method: str = "tsne",
):
    """2-D embedding (t-SNE or PCA) coloured by cluster labels / noise."""

    # 1 · Sanity + to-array ----------------------------------------------------
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X_vals = X.values
    else:
        X_vals = np.asarray(X)

    # 2 · Low-dim embedding ----------------------------------------------------
    if method.lower() == "pca":
        from sklearn.decomposition import PCA

        emb = PCA(n_components=2, random_state=42).fit_transform(X_vals)
        title = "PCA"
    else:
        tsne = TSNE(n_components=2, init="pca", random_state=42, perplexity=30)
        emb = tsne.fit_transform(X_vals)
        title = "t-SNE"

    # 3 · Colour palette -------------------------------------------------------
    labels = np.asarray(cluster_labels)
    unique = np.unique(labels)
    n_clusters = len(unique[unique != -1])
    palette = sns.color_palette("husl", max(n_clusters, 2))

    def _label_colour(lbl):
        return "#999999" if lbl == -1 else palette[int(lbl) % n_clusters]

    colours = list(map(_label_colour, labels))

    # 4 · Plot ---------------------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.scatter(emb[:, 0], emb[:, 1], c=colours, s=18, alpha=0.8, linewidths=0)
    plt.title(f"{title} Cluster Map · {algorithm_name}")

    # 5 · Legend -------------------------------------------------------------
    handles = []
    for lbl in unique:
        colour = _label_colour(lbl)
        name = "Noise" if lbl == -1 else f"Cluster {lbl}"
        handles.append(plt.Line2D([], [], marker="o", color=colour, linestyle="", label=name))
    plt.legend(handles=handles, title="Label", loc="best", frameon=True, fontsize="small")

    plt.tight_layout()
    plt.show()
