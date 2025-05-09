from __future__ import annotations
import logging, re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

# ────────────────────────────────────────────────────────────────
# 1 · Domain‑aware feature additions
# ────────────────────────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple, interpretable columns."""
    if "incident_hour_of_the_day" in df.columns:
        df["is_rush_hour"] = df["incident_hour_of_the_day"].apply(
            lambda h: 1 if 7 <= h <= 9 or 16 <= h <= 18 else 0
        )

    logging.info("Feature engineering complete.")
    return df


# ────────────────────────────────────────────────────────────────
# 2 · Helpers for auto‑encoding mixed‑type frames
# ────────────────────────────────────────────────────────────────
_DATE_RE = re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}")

def _encode_mixed_df(X: pd.DataFrame) -> pd.DataFrame:
    """
    ▸ Dates  → ordinal integers (1970‑01‑01 = 719163, etc.)  
    ▸ Strings→ ordinal‑encoded (fast, no sparse matrices)
    """
    X_enc = X.copy()

    # 2.1 Try to parse obvious date strings
    obj_cols = X_enc.select_dtypes("object")
    for col in obj_cols:
        if obj_cols[col].str.match(_DATE_RE).all():
            X_enc[col] = pd.to_datetime(obj_cols[col], errors="coerce").map(
                pd.Timestamp.toordinal
            )

    # 2.2 Ordinal‑encode any remaining object columns
    cat_cols = X_enc.select_dtypes("object").columns
    if len(cat_cols):
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_enc[cat_cols] = oe.fit_transform(X_enc[cat_cols].astype(str))

    return X_enc


# ────────────────────────────────────────────────────────────────
# 3 · Top‑N feature selector that **never** breaks on strings/dates
# ────────────────────────────────────────────────────────────────
def select_top_features(
    df: pd.DataFrame,
    target: str,
    top_n: int = 20,
    log_importances: bool = True,
) -> List[str]:
    """Return list⇢top‑N predictors ranked by Random Forest Gini importance."""
    if df.empty:
        logging.warning("Feature selection skipped – DataFrame empty.")
        return []

    if target not in df.columns:
        logging.error(f"Target '{target}' not found in DataFrame.")
        return []

    X_raw = df.drop(columns=[target])
    y = df[target]

    # Ensure everything is numeric **before** the forest sees it
    X = _encode_mixed_df(X_raw)

    try:
        rf = RandomForestClassifier(
            n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced"
        )
        rf.fit(X, y)
    except Exception as exc:
        logging.error(f"Error in feature selection: {exc}")
        return []

    importances = rf.feature_importances_
    feat_imp = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)

    if log_importances:
        logging.info("All feature importances (descending):")
        for f, imp in feat_imp:
            logging.info(f"  {f:<35} {imp:.5f}")

    top_feats = [f for f, _ in feat_imp[:top_n]]
    logging.info(f"Selected top {len(top_feats)} features.")
    return top_feats
