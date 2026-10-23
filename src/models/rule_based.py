import pandas as pd


def rule_based_detector(
    features,
    min_roc=0.5,
    min_auc=300,
):
    """
    Rule-based retrospective meal detection.

    A timestamp is flagged when:
    - Average glucose rise over 60 minutes >= min_roc
    - Glucose AUC over 60 minutes >= min_auc

    Uses only glucose-derived features.
    Does not use reported carbohydrate entries or labels.

    Returns
    -------
    pd.DataFrame
        sid, timestamp, meal_probability
    """
    predictions = features[["sid", "timestamp"]].copy()

    detected = (
        (features["glucose_roc_60"] >= min_roc)
        & (features["glucose_auc_60"] >= min_auc)
    )

    predictions["meal_probability"] = detected.astype(float)

    # Missing required features mean we cannot apply the rule.
    missing = features[
        ["glucose_roc_60", "glucose_auc_60"]
    ].isna().any(axis=1)

    predictions.loc[missing, "meal_probability"] = float("nan")

    return predictions