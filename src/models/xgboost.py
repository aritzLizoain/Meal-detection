import numpy as np

from xgboost import XGBClassifier

from data_processing.get_features import FEATURE_COLUMNS


def xgboost_detector(train, val):
    """
    Train XGBoost on training participants and predict
    meal scores for validation participants.

    Returns
    -------
    pd.DataFrame
        Columns: sid, timestamp, meal_probability
    """

    # Use only labeled rows with complete features
    train_mask = (
        train["label"].isin([0, 1])
        & train[FEATURE_COLUMNS].notna().all(axis=1)
    )

    X_train = train.loc[train_mask, FEATURE_COLUMNS]
    y_train = train.loc[train_mask, "label"].astype(int)

    # Handle class imbalance using training data only
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()

    scale_pos_weight = n_negative / n_positive

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

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