import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def match_events(predicted_events, meals, min_carbs=5, tolerance_min=10):
    """
    Match predicted meal events to reported meals, one-to-one.

    Prioritizes:
    1. Maximum number of matches within the tolerance.
    2. Minimum total absolute timing error.

    Returns
    -------
    matches : pd.DataFrame
        Matched reported and predicted events.
    unmatched_predictions : pd.DataFrame
        Predictions without a matching reported meal.
    unmatched_meals : pd.DataFrame
        Eligible reported meals without a matching prediction.
    """
    references = meals[meals["carbs"] >= min_carbs].copy()

    references["timestamp"] = pd.to_datetime(references["timestamp"])
    predictions = predicted_events.copy()
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])

    matched_rows = []
    matched_ref_idx = set()
    matched_pred_idx = set()

    for sid, ref_group in references.groupby("sid"):
        pred_group = predictions[predictions["sid"] == sid]

        if pred_group.empty:
            continue

        ref_times = ref_group["timestamp"].to_numpy()
        pred_times = pred_group["timestamp"].to_numpy()

        # Pairwise signed timing errors: prediction - reference.
        errors = (
            pred_times[None, :] - ref_times[:, None]
        ) / np.timedelta64(1, "m")

        absolute_errors = np.abs(errors)
        valid = absolute_errors <= tolerance_min

        if not valid.any():
            continue

        # Dummy columns allow reference meals to remain unmatched.
        # The unmatched penalty is larger than any valid matching cost.
        # Invalid pairs cost more than leaving a meal unmatched.
        unmatched_cost = tolerance_min + 1
        invalid_cost = 2 * unmatched_cost + 1

        cost = np.full(
            (len(ref_group), len(pred_group) + len(ref_group)),
            invalid_cost,
            dtype=float,
        )

        cost[:, :len(pred_group)] = np.where(
            valid, absolute_errors, invalid_cost
        )

        for i in range(len(ref_group)):
            cost[i, len(pred_group) + i] = unmatched_cost

        ref_positions, pred_positions = linear_sum_assignment(cost)

        for i, j in zip(ref_positions, pred_positions):
            if j >= len(pred_group) or not valid[i, j]:
                continue

            ref = ref_group.iloc[i]
            pred = pred_group.iloc[j]

            matched_ref_idx.add(ref.name)
            matched_pred_idx.add(pred.name)

            matched_rows.append({
                "sid": sid,
                "meal_timestamp": ref["timestamp"],
                "predicted_timestamp": pred["timestamp"],
                "carbs": ref["carbs"],
                "timing_error_min": errors[i, j],
                "absolute_error_min": absolute_errors[i, j],
            })

    matches = pd.DataFrame(
        matched_rows,
        columns=[
            "sid",
            "meal_timestamp",
            "predicted_timestamp",
            "carbs",
            "timing_error_min",
            "absolute_error_min",
        ],
    )

    unmatched_predictions = predictions.loc[
        ~predictions.index.isin(matched_pred_idx)
    ].copy()

    unmatched_meals = references.loc[
        ~references.index.isin(matched_ref_idx)
    ].copy()

    return matches, unmatched_predictions, unmatched_meals