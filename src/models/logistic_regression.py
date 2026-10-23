import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from data_processing import FEATURE_COLUMNS


def logistic_regression_detector(train, val):
    """
    Train logistic regression on training participants and predict
    meal probabilities for validation participants.

    Rows with ambiguous labels or missing features are excluded
    from training. Validation rows with missing features receive NaN.

    Returns
    -------
    pd.DataFrame
        Columns: sid, timestamp, meal_probability
    """

    # Select valid training rows
    train_mask = (
        train["label"].isin([0, 1])
        & train[FEATURE_COLUMNS].notna().all(axis=1)
    )

    X_train = train.loc[train_mask, FEATURE_COLUMNS]
    y_train = train.loc[train_mask, "label"].astype(int)

    # Build pipeline: scaling is fitted on training data only
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    # Prepare validation predictions
    predictions = val[["sid", "timestamp"]].copy()
    predictions["meal_probability"] = np.nan

    val_mask = val[FEATURE_COLUMNS].notna().all(axis=1)

    if val_mask.any():
        X_val = val.loc[val_mask, FEATURE_COLUMNS]

        predictions.loc[val_mask, "meal_probability"] = (
            model.predict_proba(X_val)[:, 1]
        )

    return predictions